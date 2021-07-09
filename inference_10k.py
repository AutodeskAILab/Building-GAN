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
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

from shutil import copyfile

from datetime import datetime
import numpy as np
import shutil, psutil, sys
import time
import gc

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from torch_geometric.data import DataLoader, Batch

from Model.models import Generator, Discriminator_Voxel  #, Discriminator_Program #latest model with LP, room dis
# from Model.models_ablation import Generator # with bf/pe/norm switch, for ablation
# from Model.models_0216backup import Generator # for bf/pe (best model)
# from Model.models_0216_b2 import Generator # pure base

from util_eval import *
from util_graph import *



device_ids = list(range(torch.cuda.device_count())) if cuda else []
assert(args.batch_size % len(device_ids) == 0)
print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if cuda else "Using CPU")
if cuda:
    print([torch.cuda.get_device_name(device_id) for device_id in device_ids])
print(args)
device = torch.device('cuda:'+str(device_ids[0]) if cuda else 'cpu')

# # for variations
# inference_dir = "inference_variation"
# args.train_data_dir = 'Data/6types-processed-variation-v1-z10_data'
# args.raw_dir = 'Data/6types-raw-variation-v1-z10_data'

inference_dir = "inference_test"
# LG
# inference_dir = "LG"

# trained_id = "Mar07_20-55-03_lp_w10+=1_s100_after15"
# epoch_id = "51"
# trained_id = "Mar07_18-05-16_lp_w+=2_s100_after15"
# epoch_id = "57"
# trained_id = "Mar03_16-29-28_lp_w1_s100_after15"
# epoch_id = '72'
# trained_id = "Mar03_16-26-21_lp_w1_s100_after20"
# epoch_id = '54'
# trained_id = "Feb13_15-06-43_l12aevenb8_pe_bf"
# epoch_id = '60' # tried 60 70 85. 60 best FID, 70 best visual
# trained_id = "Feb11_16-10-28_la12att024810f_b8"
# epoch_id = '50'

# attn frequency
# trained_id = "Feb20_23-04-29_l12a036912b8"
# trained_id = "Feb17_23-03-21_l12a04812b8"
# trained_id = "Feb20_09-47-10_l12a0612b8" # epoch 50 exploded, epoch 35 fine
# trained_id = "Feb17_23-02-10_l12alastb8" # epoch 45 exploded, epoch 40 exploded, epoch 5 is fine

# voxel layers
# trained_id = "Feb17_22-56-33_l4aevenb8"
# trained_id = "Feb20_09-49-57_l6aevenb8"
# trained_id = "Feb17_22-57-55_l8aeveryb8"
# trained_id = "Feb20_09-52-28_l10aevenb8"

# building feature/pos_enc
# trained_id = "Feb13_15-10-28_l12aevenb8_bf" #tried 50 35
# epoch_id = "35"

# no relative position
# trained_id = "Mar10_22-12-24_no_relative" # 15
trained_id = "Mar16_15-57-35_no_relative" # fixed some bugs
epoch_id = "24" # 12 and 24

# building dis/story dis
# trained_id = "Mar11_06-25-53_no_story_dis" # 27
# trained_id = "Mar10_22-24-19_no_building_dis" # 15
# epoch_id = "15"


trained_file = 'runs/{}/checkpoints/{}.ckpt'.format(trained_id, epoch_id)
truncated = False


if not truncated:
    viz_dir = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "output")
    var_viz_dir1 = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "var_output1")
    var_viz_dir2 = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "var_output2")
    trunc_num = 1.0

else:
    viz_dir = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "output-trunc")
    var_viz_dir1 = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "var_output1-trunc")
    var_viz_dir2 = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "var_output2-trunc")
    trunc_num = 0.7

mkdirs = [inference_dir, os.path.join(inference_dir, trained_id), os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time), viz_dir, var_viz_dir1, var_viz_dir2]
for mkdir in mkdirs:
    if not os.path.exists(mkdir):
        os.mkdir(mkdir)


# Load Data
follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']

# LG!
# args.batch_size = 1

normal: 96000-99999
train_size = args.train_size//args.batch_size * args.batch_size

# # variation: train size = 0
# train_size = 0

data_fname_list = sorted(os.listdir(args.train_data_dir))
print("Total %d data: %d train / %d test" % (len(data_fname_list), train_size, len(data_fname_list)-train_size))

# no use for variation test
variation_test_data1 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id1]))
variation_test_data2 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id2]))
variation_test_batch1 = Batch.from_data_list([variation_test_data1 for _ in range(args.variation_num)], follow_batch)
variation_test_batch2 = Batch.from_data_list([variation_test_data2 for _ in range(args.variation_num)], follow_batch)

test_data_list = []
for fname in data_fname_list[train_size:]:
    test_data_list.append(torch.load(os.path.join(args.train_data_dir, fname)))

# !!!!!! batch_size=1 for testing!!!
test_data_loader = DataLoader(test_data_list, follow_batch=follow_batch, batch_size=20, shuffle=False, num_workers=args.n_cpu)
program_input_dim = test_data_list[0].program_class_feature.size(-1) + 1  # data_list[0].story_level_feature.size(-1)
voxel_input_dim = test_data_list[0].voxel_feature.size(-1)
voxel_label_dim = test_data_list[0].voxel_label.size(-1)



# Load Model
generator = Generator(program_input_dim, voxel_input_dim, args.latent_dim, args.noise_dim, args.program_layer, args.voxel_layer, device).to(device)
generator.load_state_dict(torch.load(trained_file))
generator.eval()

# ###### 10000 for variation
# for epoch in range(0, 1):
#
#     n_batches = len(test_data_loader)
#
#     os.mkdir(os.path.join(viz_dir, str(epoch)))
#     os.mkdir(os.path.join(var_viz_dir1, str(epoch)))
#     os.mkdir(os.path.join(var_viz_dir2, str(epoch)))
#     evaluate_new(test_data_loader, generator, args.raw_dir, os.path.join(viz_dir, str(epoch)), follow_batch, device_ids, number_of_batches=n_batches,trunc=trunc_num)
#     # generate_multiple_outputs_from_batch(variation_test_batch1, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir1, str(epoch)), follow_batch, device_ids, trunc=trunc_num)
#     # generate_multiple_outputs_from_batch(variation_test_batch2, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir2, str(epoch)), follow_batch, device_ids, trunc=trunc_num)
#
# print("process completed")


# run 3 epochs: 4000, 4000, 2000 (10000 in total)
for epoch in range(0, 3):

    if epoch == 2:
        n_batches = len(test_data_loader) / 2
    else:
        n_batches = len(test_data_loader)
        # n_batches = 5

    os.mkdir(os.path.join(viz_dir, str(epoch)))
    os.mkdir(os.path.join(var_viz_dir1, str(epoch)))
    os.mkdir(os.path.join(var_viz_dir2, str(epoch)))
    evaluate_new(test_data_loader, generator, args.raw_dir, os.path.join(viz_dir, str(epoch)), follow_batch, device_ids, number_of_batches=n_batches,trunc=trunc_num)
    generate_multiple_outputs_from_batch(variation_test_batch1, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir1, str(epoch)), follow_batch, device_ids, trunc=trunc_num)
    generate_multiple_outputs_from_batch(variation_test_batch2, args.variation_num, generator, args.raw_dir, os.path.join(var_viz_dir2, str(epoch)), follow_batch, device_ids, trunc=trunc_num)


# combine 10K
output_id = "_".join(["10k", trained_id, epoch_id])
if truncated:
    output_id += "_trunc"
final_dir = os.path.join(viz_dir, output_id)
os.mkdir(final_dir)
os.mkdir(os.path.join(final_dir, "global_graph_data"))
os.mkdir(os.path.join(final_dir, "local_graph_data"))
os.mkdir(os.path.join(final_dir, "voxel_data"))

print(final_dir)

for epoch in range(3):
    cur_dir = os.path.join(viz_dir, str(epoch))
    # print(cur_dir)
    for graph_folder in os.listdir(cur_dir):

        source_folder = os.path.join(cur_dir, graph_folder)
        paste_folder = os.path.join(final_dir, graph_folder)

        for f in os.listdir(source_folder):

            # print(f)
            fs = f.split('.')
            fs[0] =fs[0] + '-' + str(epoch)
            fnew = '.'.join(fs)
            # print(fnew)
            copyfile(os.path.join(source_folder, f), os.path.join(paste_folder, fnew))

print("process completed")



