from train_args import make_args
import os
args = make_args()
if str(args.cuda) == -1:
    cuda = False
else:
    cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader, Batch
from Model.models import Generator, DiscriminatorVoxel
from Model.losses import VolumetricDesignLoss_D, VolumetricDesignLoss_G, compute_gradient_penalty
from util_eval import *
from util_graph import *
from Data.LargeFilenameDataset import LargeFilenameDataset

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
run_id = current_time + "_" + args.comment
device_ids = list(range(torch.cuda.device_count())) if cuda else []
device = torch.device('cuda:'+str(device_ids[0]) if cuda else 'cpu')
print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if cuda else "Using CPU")
print([torch.cuda.get_device_name(device_id) for device_id in device_ids] if cuda else "Using CPU")
print(run_id)
print(args)


run_dir = "runs"
viz_dir = os.path.join(run_dir, run_id, "output")
var_viz_dir1 = os.path.join(run_dir, run_id, "var_output1")
var_viz_dir2 = os.path.join(run_dir, run_id, "var_output2")
model_dir = os.path.join(run_dir, run_id, "checkpoints")
mkdirs = [os.path.join(run_dir, run_id), viz_dir, var_viz_dir1, var_viz_dir2, model_dir]
for mkdir in mkdirs:
    os.mkdir(mkdir)
writer_train = SummaryWriter(log_dir=os.path.join(run_dir, run_id))


# Load Data
data_fname_list = sorted(os.listdir(args.train_data_dir))
follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']
train_size = args.train_size//args.batch_size * args.batch_size
test_size = args.test_size//args.batch_size * args.batch_size
# print("Total %d data: %d train / %d test" % (len(data_fname_list), train_size, len(data_fname_list)-train_size))
print("Total %d data: %d train / %d test" % (len(data_fname_list), train_size, test_size))

train_data_list = LargeFilenameDataset(args.train_data_dir, data_fname_list[:train_size])
train_data_loader = DataLoader(train_data_list,  follow_batch=follow_batch, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_data_list = [torch.load(os.path.join(args.train_data_dir, fname)) for fname in data_fname_list[train_size:train_size + test_size]]
test_data_loader = DataLoader(test_data_list, follow_batch=follow_batch, batch_size=20, shuffle=False, num_workers=args.n_cpu)

variation_test_data1 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id1]))
variation_test_data2 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id2]))
variation_test_batch1 = Batch.from_data_list([variation_test_data1 for _ in range(args.variation_num)], follow_batch)
variation_test_batch2 = Batch.from_data_list([variation_test_data2 for _ in range(args.variation_num)], follow_batch)

# Load Model
program_input_dim = test_data_list[0].program_class_feature.size(-1) + 1
voxel_input_dim = test_data_list[0].voxel_feature.size(-1)
voxel_label_dim = test_data_list[0].voxel_label.size(-1)

generator = Generator(program_input_dim, voxel_input_dim, args.latent_dim, args.noise_dim, args.program_layer, args.voxel_layer, device).to(device)
discriminator = DiscriminatorVoxel(voxel_input_dim, voxel_label_dim, args.latent_dim, args.voxel_layer, act="sigmoid" if args.gan_loss == "NSGAN" else "").to(device)
d_loss_func = VolumetricDesignLoss_D(gan_loss=args.gan_loss, gp_lambda=args.gp_lambda).to(device)
g_loss_func = VolumetricDesignLoss_G(lp_weight=args.lp_weight, tr_weight=args.tr_weight, far_weight=args.far_weight, embedding_dim=args.latent_dim, sample_size=args.lp_sample_size, similarity_fun=args.lp_similarity_fun,
                                     gan_loss=args.gan_loss, lp_loss=args.lp_loss_fun, hinge_margin=args.lp_hinge_margin).to(device)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.b1, args.b2))


for epoch in range(args.n_epochs):
    for i, batch in enumerate(train_data_loader):
        batch = batch.to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in list(discriminator.parameters()):
            p.requires_grad = True
        optimizer_D.zero_grad()

        # random variables
        program_z = torch.rand(tuple([batch.program_class_feature.shape[0], args.noise_dim])).to(device)
        voxel_z = torch.rand(tuple([batch.voxel_feature.shape[0], args.noise_dim])).to(device)
        out, soft_out, mask, att, max_out_program_index = generator(batch, program_z, voxel_z)

        if i % args.n_critic_d == 0:
            fake_validity_voxel = discriminator(detach_batch(batch), out.detach())
            real_validity_voxel = discriminator(batch, batch.voxel_label)

            if args.gan_loss == "WGANGP" and d_loss_func.gp_lambda != 0:
                gp, gp_b, gp_s = compute_gradient_penalty(discriminator, detach_batch(batch), batch.voxel_label.data, soft_out.data)
            else:
                gp, gp_b, gp_s = torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)

            d_loss = d_loss_func(real_validity_voxel, fake_validity_voxel, gp)
            d_loss.backward()
            if i % args.n_critic_d == 0:
                optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        for p in list(discriminator.parameters()):
            p.requires_grad = False
        optimizer_G.zero_grad()

        if i % args.n_critic_g == 0:
            validity_G_voxel = discriminator(batch, out)
            g_loss, gan_loss, tr_loss, far_loss = g_loss_func(validity_G_voxel, batch, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
            g_loss.backward()
            optimizer_G.step()

    if epoch % args.plot_period == 0:
        avg_fake_validity_b, avg_fake_validity_s = torch.mean(fake_validity_voxel[0]), torch.mean(fake_validity_voxel[1])
        avg_real_validity_b, avg_real_validity_s = torch.mean(real_validity_voxel[0]), torch.mean(real_validity_voxel[1])
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, gp: %.4f] [G loss: %.4f (%.4f, %.4f, %.4f)] [Validity Real: (%.4f, %.4f), Fake (%.4f, %.4f)]" %
              (epoch, args.n_epochs, i + 1, len(train_data_loader), d_loss.item(), gp.item(), g_loss.item(), gan_loss.item(), tr_loss.item(), far_loss.item(),
               avg_real_validity_b.item(), avg_real_validity_s.item(),  avg_fake_validity_b.item(), avg_fake_validity_s.item()))
        writer_train.add_scalar('d_loss', d_loss, epoch)
        writer_train.add_scalar('_gp', gp.item(), epoch)
        writer_train.add_scalar('g_loss', g_loss, epoch)
        writer_train.add_scalar('gan_loss', gan_loss, epoch)
        writer_train.add_scalar('target_ratio_loss', tr_loss, epoch)
        writer_train.add_scalar('D(fake)', avg_fake_validity_b, epoch)
        writer_train.add_scalar('D(real)', avg_real_validity_b, epoch)
        writer_train.add_scalar('D(fake)_story', avg_fake_validity_s, epoch)
        writer_train.add_scalar('D(real)_story', avg_real_validity_s, epoch)
        writer_train.close()

    if epoch % args.eval_period == 0:
        os.mkdir(os.path.join(viz_dir, str(epoch)))
        os.mkdir(os.path.join(var_viz_dir1, str(epoch)))
        os.mkdir(os.path.join(var_viz_dir2, str(epoch)))
        evaluate(test_data_loader, generator, args.raw_dir, os.path.join(viz_dir, str(epoch)), follow_batch, device_ids, number_of_batches=1)
        generate_multiple_outputs_from_batch(variation_test_batch1, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir1, str(epoch)), follow_batch, device_ids)
        generate_multiple_outputs_from_batch(variation_test_batch2, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir2, str(epoch)), follow_batch, device_ids)
        save_model = os.path.join(model_dir, '{}.ckpt'.format(epoch))
        torch.save(generator.state_dict(), save_model)

