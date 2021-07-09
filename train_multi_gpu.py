from train_args import make_args
args = make_args()
# gpu_pci_str = "1, 2"  # multi-gpu version
gpu_pci_str = str(args.cuda)  # single-gpu version
import os
gpu_pci_ids = [int(i) for i in gpu_pci_str.split(',')]
cuda = len(gpu_pci_ids) > 0
if_multi_gpu = len(gpu_pci_ids) > 1
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_pci_str

from datetime import datetime
import numpy as np
import shutil, psutil, sys
import time
import gc

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torch_geometric.data import DataLoader, Batch
from Model.models import Generator, Discriminator_Voxel  #, Discriminator_Program
from Model.losses import VolumetricDesignLoss_D, VolumetricDesignLoss_G, compute_gradient_penalty, compute_gradient_penalty_room_dis
from util_eval import *
from util_graph import *
from Data.LargeFilenameDataset import LargeFilenameDataset
from CurriculumDataIterator import CurriculumDataIterator, CurriculumFilenameDataIterator

# process = psutil.Process(os.getpid())
# def debug_memory(x):
#     for var, obj in list(x.items()):
#         print(var, sys.getsizeof(obj))
#     print(process.memory_info().rss / 1024 / 1024 / 1024)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
run_id = current_time + "_" + args.comment
print(run_id)
device_ids = list(range(torch.cuda.device_count())) if cuda else []
assert(args.batch_size % len(device_ids) == 0)
print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if cuda else "Using CPU")
if cuda:
    print([torch.cuda.get_device_name(device_id) for device_id in device_ids])
print(args)
device = torch.device('cuda:'+str(device_ids[0]) if cuda else 'cpu')


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
follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']
# if not os.path.exists(args.preload_dir):  # preload isn't faster by a lot (170s vs 187s). Don't waste memory
#     data_list = []
#     for filename in sorted(os.listdir(args.train_data_dir)):
#         data_list.append(torch.load(os.path.join(args.train_data_dir, filename)))  # , map_location=device  -> if you use multithreading data-loader, can't map to GPU here
#     torch.save(data_list, args.preload_dir)
# else:
#     data_list = torch.load(args.preload_dir)

train_size = args.train_size//args.batch_size * args.batch_size
# print("Total %d data: %d train / %d test" % (len(data_list), train_size, len(data_list)-train_size))
# variation_test_batch1 = Batch.from_data_list([data_list[args.variation_eval_id1] for _ in range(args.variation_num)], follow_batch)
# variation_test_batch2 = Batch.from_data_list([data_list[args.variation_eval_id2] for _ in range(args.variation_num)], follow_batch)


# if args.if_curriculum:
#     # story_ticks, iter_ticks = [1, 2, 4, 7], [40, 80, 120, 160]  # equal
#     # story_ticks, iter_ticks = [1, 2, 4, 7], [20, 40, 80, 140]   # more on high story
#     # story_ticks, iter_ticks = [1, 2, 4, 7], [60, 120, 160, 180]  # more on low story
#     # story_ticks, iter_ticks = [1, 2, 4, 7], [100, 150, 200, 250]  # more on low story 2
#     story_ticks, iter_ticks = [1, 2, 4, 7], [5, 10, 15, 20]  # more on low story 3
#     # story_ticks, iter_ticks = [1, 2, 4, 7], [10, 20, 30, 40]  # more on low story 4
#     print("Story ticks and Iter ticks")
#     print(story_ticks, iter_ticks)
#     curriculum_iter = CurriculumDataIterator(data_list[:train_size], story_ticks, follow_batch, args.batch_size, True, args.n_cpu)
#     train_data_loader = None
# else:
#     train_data_loader = DataLoader(data_list[:train_size], follow_batch=follow_batch, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
#     curriculum_iter, iter_ticks = None, None
#
# test_data_loader = DataLoader(data_list[train_size:], follow_batch=follow_batch, batch_size=20, shuffle=False, num_workers=args.n_cpu)
#
# program_input_dim = data_list[0].program_class_feature.size(-1) + 1  # data_list[0].story_level_feature.size(-1)
# voxel_input_dim = data_list[0].voxel_feature.size(-1)
# voxel_label_dim = data_list[0].voxel_label.size(-1)
#
# del data_list
# gc.collect()


data_fname_list = sorted(os.listdir(args.train_data_dir))
print("Total %d data: %d train / %d test" % (len(data_fname_list), train_size, len(data_fname_list)-train_size))

variation_test_data1 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id1]))
variation_test_data2 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id2]))
variation_test_batch1 = Batch.from_data_list([variation_test_data1 for _ in range(args.variation_num)], follow_batch)
variation_test_batch2 = Batch.from_data_list([variation_test_data2 for _ in range(args.variation_num)], follow_batch)

if args.if_curriculum:
    # story_ticks, iter_ticks = [1, 2, 4, 7], [40, 80, 120, 160]  # equal
    # story_ticks, iter_ticks = [1, 2, 4, 7], [20, 40, 80, 140]   # more on high story
    # story_ticks, iter_ticks = [1, 2, 4, 7], [60, 120, 160, 180]  # more on low story
    # story_ticks, iter_ticks = [1, 2, 4, 7], [100, 150, 200, 250]  # more on low story 2

    story_ticks, iter_ticks = [1, 2, 4, 7], [5, 10, 15, 20]  # more on low story 3
    # story_ticks, iter_ticks = [1, 2, 4, 7], [1, 2, 4, 8]

    # story_ticks, iter_ticks = [1, 2, 4, 7], [10, 20, 30, 40]  # more on low story 4
    print("Story ticks and Iter ticks")
    print(story_ticks, iter_ticks)
    curriculum_iter = CurriculumFilenameDataIterator(args.train_data_dir, data_fname_list[:train_size], story_ticks, follow_batch, args.batch_size, True, args.n_cpu)
    train_data_loader = None
else:
    train_data_loader = DataLoader(LargeFilenameDataset(args.train_data_dir, data_fname_list[:train_size]),  follow_batch=follow_batch, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
    curriculum_iter, iter_ticks = None, None

test_data_list = []
for fname in data_fname_list[train_size:]:
    test_data_list.append(torch.load(os.path.join(args.train_data_dir, fname)))
test_data_loader = DataLoader(test_data_list, follow_batch=follow_batch, batch_size=20, shuffle=False, num_workers=args.n_cpu)
program_input_dim = test_data_list[0].program_class_feature.size(-1) + 1  # data_list[0].story_level_feature.size(-1)
voxel_input_dim = test_data_list[0].voxel_feature.size(-1)
voxel_label_dim = test_data_list[0].voxel_label.size(-1)

# Load Model
generator = Generator(program_input_dim, voxel_input_dim, args.latent_dim, args.noise_dim, args.program_layer, args.voxel_layer, device).to(device)
# pretrain_path = "480.ckpt"  # Fine-tune
# generator.load_state_dict(torch.load(pretrain_path, map_location=device))
discriminator_voxel = Discriminator_Voxel(voxel_input_dim, voxel_label_dim, args.latent_dim, args.voxel_layer, act="sigmoid" if args.gan_loss == "NSGAN" else "").to(device)
# discriminator_program = Discriminator_Program(voxel_input_dim, voxel_label_dim, args.latent_dim, args.voxel_layer).to(device)

d_loss_func = VolumetricDesignLoss_D(gan_loss=args.gan_loss, gp_lambda=args.gp_lambda).to(device)
g_loss_func = VolumetricDesignLoss_G(lp_weight=args.lp_weight, tr_weight=args.tr_weight, far_weight=args.far_weight, embedding_dim=args.latent_dim, sample_size=args.lp_sample_size, similarity_fun=args.lp_similarity_fun,
                                     gan_loss=args.gan_loss, lp_loss=args.lp_loss_fun, hinge_margin=args.lp_hinge_margin).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr, betas=(args.b1, args.b2))
optimizer_Dv = torch.optim.Adam(discriminator_voxel.parameters(), lr=args.d_lr, betas=(args.b1, args.b2))
# optimizer_Dp = torch.optim.Adam(discriminator_program.parameters(), lr=args.d_lr, betas=(args.b1, args.b2))


iter_ticks_index = 0
train_data_loader = curriculum_iter.get_next() if args.if_curriculum else train_data_loader

for epoch in range(args.n_epochs):
    if args.if_curriculum and iter_ticks_index < len(iter_ticks) and epoch == iter_ticks[iter_ticks_index]:
        train_data_loader = curriculum_iter.get_next()
        iter_ticks_index += 1

    for i, batch in enumerate(train_data_loader):
        # torch.cuda.empty_cache()
        batch = batch.to(device)  # multi-processing data loader can only take CPU tensors

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in list(discriminator_voxel.parameters()):  # + list(discriminator_program.parameters()):
            p.requires_grad = True
        optimizer_Dv.zero_grad()
        # optimizer_Dp.zero_grad()

        z_shape = [batch.program_class_feature.shape[0], args.noise_dim]
        z = torch.rand(tuple(z_shape)).to(device)
        if if_multi_gpu:
            out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = data_parallel(generator, batch, tuple([z]), follow_batch, device_ids)
        else:
            out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = generator(batch, z)

        if i % args.n_critic_d == 0:  # or i % args.n_critic_p == 0:
            if if_multi_gpu:
                fake_validity_voxel = data_parallel(discriminator_voxel, detach_batch(batch), (out.detach(), max_out_program_index.detach(), mask["hard"].detach()), follow_batch, device_ids)
                # fake_validity_program = data_parallel(discriminator_program, detach_batch(batch), (out.detach(), max_out_program_index.detach(), mask["hard"].detach()), follow_batch, device_ids)
                real_validity_voxel = data_parallel(discriminator_voxel, batch, (batch.voxel_label, batch.voxel_label_program_node_index, batch.voxel_bool.view(-1, 1)), follow_batch, device_ids)
                # real_validity_program = data_parallel(discriminator_program, batch, (batch.voxel_label, batch.voxel_label_program_node_index, batch.voxel_bool.view(-1, 1)), follow_batch, device_ids)
            else:
                fake_validity_voxel = discriminator_voxel(detach_batch(batch), out.detach(), max_out_program_index.detach(), mask["hard"].detach())
                # fake_validity_program = discriminator_program(detach_batch(batch), out.detach(), max_out_program_index.detach(), mask["hard"].detach())
                real_validity_voxel = discriminator_voxel(batch, batch.voxel_label, batch.voxel_label_program_node_index, batch.voxel_bool.view(-1, 1))
                # real_validity_program = discriminator_program(batch, batch.voxel_label, batch.voxel_label_program_node_index, batch.voxel_bool.view(-1, 1))

            # torch.cuda.empty_cache()
            # gc.collect()

            if args.gan_loss == "WGANGP":
                gp, gp_b, gp_s, _ = compute_gradient_penalty(discriminator_voxel, None, detach_batch(batch), batch.voxel_label.data, soft_out.data, follow_batch, device_ids) if d_loss_func.gp_lambda != 0 else None

                # # 0301 for room dis
                # gp, gp_b, gp_s, gp_r, _ = compute_gradient_penalty_room_dis(discriminator_voxel, None, detach_batch(batch),
                #                                              att["soft"].data, mask["soft"].data, follow_batch,
                #                                              device_ids) if d_loss_func.gp_lambda != 0 else None
            else:
                gp, gp_b, gp_s = torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
            d_loss = d_loss_func(real_validity_voxel, None, fake_validity_voxel, None, gp)
            d_loss.backward()
            if i % args.n_critic_d == 0:
                optimizer_Dv.step()
            # if i % args.n_critic_p == 0:
            #     optimizer_Dp.step()

        # torch.cuda.empty_cache()
        # gc.collect()

        # -----------------
        #  Train Generator
        # -----------------
        for p in list(discriminator_voxel.parameters()):  # + list(discriminator_program.parameters()):
            p.requires_grad = False
        optimizer_G.zero_grad()

        if i % args.n_critic_g == 0:
            if if_multi_gpu:
                validity_G_voxel = data_parallel(discriminator_voxel, batch, (out, max_out_program_index, mask["hard"]), follow_batch, device_ids)
                # validity_G_program = data_parallel(discriminator_program, batch, (out, max_out_program_index, mask["hard"]), follow_batch, device_ids)
            else:
                validity_G_voxel = discriminator_voxel(batch, out, max_out_program_index, mask["hard"])
                # validity_G_program = discriminator_program(batch, out, max_out_program_index, mask["hard"])

            # # 0313 adding lp/tr/far after x epochs
            # if epoch < 9:
            #     # g_loss_func.lp_weight = 0
            #     g_loss_func.tr_weight = 0
            #     # g_loss_func.far_weight = 0
            # else:
            #     # g_loss_func.lp_weight = args.lp_weight + (epoch - 15) * 1
            #     # g_loss_func.lp_weight = (epoch - 15 + 1) * 2
            #     g_loss_func.tr_weight = 5
            #     # g_loss_func.far_weight = 1

            g_loss, gan_loss, lp_loss, tr_loss, far_loss = g_loss_func(validity_G_voxel, None, batch, att["hard"], mask["hard"], pooled_program_feature_from_voxel, max_out_program_index, area_index_in_voxel_feature=6)
            g_loss.backward()
            optimizer_G.step()

    # torch.cuda.empty_cache()
    # gc.collect()

    if epoch % args.plot_period == 0:
        avg_fake_validity_b, avg_fake_validity_s = torch.mean(fake_validity_voxel[0]), torch.mean(fake_validity_voxel[1])
        avg_real_validity_b, avg_real_validity_s = torch.mean(real_validity_voxel[0]), torch.mean(real_validity_voxel[1])
        # avg_fake_validity_p, avg_real_validity_p = torch.mean(fake_validity_program), torch.mean(real_validity_program)
        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f, gp: %.4f] [G loss: %.4f (%.4f, %.4f, %.4f, %.4f)] [Validity Real: (%.4f, %.4f), Fake (%.4f, %.4f)]" %
              (epoch, args.n_epochs, i + 1, len(train_data_loader), d_loss.item(), gp.item(), g_loss.item(), gan_loss.item(), lp_loss.item(), tr_loss.item(), far_loss.item(),
               avg_real_validity_b.item(), avg_real_validity_s.item(),  avg_fake_validity_b.item(), avg_fake_validity_s.item()))
        writer_train.add_scalar('d_loss', d_loss, epoch)
        writer_train.add_scalar('_gp', gp.item(), epoch)
        writer_train.add_scalar('g_loss', g_loss, epoch)
        writer_train.add_scalar('gan_loss', gan_loss, epoch)
        writer_train.add_scalar('link_prediction_loss', lp_loss, epoch)
        writer_train.add_scalar('target_ratio_loss', tr_loss, epoch)
        writer_train.add_scalar('D(fake)', avg_fake_validity_b, epoch)
        writer_train.add_scalar('D(real)', avg_real_validity_b, epoch)
        writer_train.add_scalar('D(fake)_story', avg_fake_validity_s, epoch)
        writer_train.add_scalar('D(real)_story', avg_real_validity_s, epoch)
        writer_train.close()  # prevent memory leaking
        # writer_train.add_scalar('D(fake)_program', avg_fake_validity_p, epoch)
        # writer_train.add_scalar('D(real)_program', avg_real_validity_p, epoch)
        # torch.cuda.empty_cache()



    if epoch % args.eval_period == 0:
        os.mkdir(os.path.join(viz_dir, str(epoch)))
        os.mkdir(os.path.join(var_viz_dir1, str(epoch)))
        os.mkdir(os.path.join(var_viz_dir2, str(epoch)))
        evaluate_new(test_data_loader, generator, args.raw_dir, os.path.join(viz_dir, str(epoch)), follow_batch, device_ids, number_of_batches=1)
        generate_multiple_outputs_from_batch_new(variation_test_batch1, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir1, str(epoch)), follow_batch, device_ids)
        generate_multiple_outputs_from_batch_new(variation_test_batch2, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir2, str(epoch)), follow_batch, device_ids)
        save_model = os.path.join(model_dir, '{}.ckpt'.format(epoch))
        torch.save(generator.state_dict(), save_model)

