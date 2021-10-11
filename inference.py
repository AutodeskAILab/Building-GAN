from train_args import make_args
args = make_args()

gpu_pci_str = str(args.cuda)  # single-gpu version
import os
gpu_pci_ids = [int(i) for i in gpu_pci_str.split(',')]
cuda = len(gpu_pci_ids) > 0
if_multi_gpu = len(gpu_pci_ids) > 1
if cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_pci_str

from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')

from torch_geometric.data import DataLoader, Batch

from Model.models import Generator
from util_eval import *
from util_graph import *


device_ids = list(range(torch.cuda.device_count())) if cuda else []
assert(args.batch_size % len(device_ids) == 0)
print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']) if cuda else "Using CPU")
if cuda:
    print([torch.cuda.get_device_name(device_id) for device_id in device_ids])
print(args)
device = torch.device('cuda:'+str(device_ids[0]) if cuda else 'cpu')

inference_dir = "inference"
trained_id = "iccv2021"
epoch_id = '70'

trained_file = 'runs/{}/checkpoints/{}.ckpt'.format(trained_id, epoch_id)

viz_dir = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "output")
var_viz_dir1 = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "var_output1")
var_viz_dir2 = os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time, "var_output2")

mkdirs = [inference_dir, os.path.join(inference_dir, trained_id), os.path.join(inference_dir, trained_id, epoch_id + "_" + current_time), viz_dir, var_viz_dir1, var_viz_dir2]
for mkdir in mkdirs:
    if not os.path.exists(mkdir):
        os.mkdir(mkdir)

truncated = False
if not truncated:
    trunc_num = 1.0
else:
    trunc_num = 0.7

# Load Data
follow_batch = ['program_target_ratio', 'program_class_feature', 'voxel_feature']
train_size = args.train_size//args.batch_size * args.batch_size
test_size = args.test_size//args.batch_size * args.batch_size

data_fname_list = sorted(os.listdir(args.train_data_dir))
print("Total %d data: %d train / %d test" % (len(data_fname_list), train_size, test_size))

variation_test_data1 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id1]))
variation_test_data2 = torch.load(os.path.join(args.train_data_dir, data_fname_list[args.variation_eval_id2]))
variation_test_batch1 = Batch.from_data_list([variation_test_data1 for _ in range(args.variation_num)], follow_batch)
variation_test_batch2 = Batch.from_data_list([variation_test_data2 for _ in range(args.variation_num)], follow_batch)

test_data_list = []
for fname in data_fname_list[train_size:train_size + test_size]:
    test_data_list.append(torch.load(os.path.join(args.train_data_dir, fname)))

test_data_loader = DataLoader(test_data_list, follow_batch=follow_batch, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)
program_input_dim = test_data_list[0].program_class_feature.size(-1) + 1  # data_list[0].story_level_feature.size(-1)
voxel_input_dim = test_data_list[0].voxel_feature.size(-1)
voxel_label_dim = test_data_list[0].voxel_label.size(-1)

# Load Model
generator = Generator(program_input_dim, voxel_input_dim, args.latent_dim, args.noise_dim, args.program_layer, args.voxel_layer, device).to(device)
generator.load_state_dict(torch.load(trained_file), strict=False)
generator.eval()

# evaluate
n_batches = 20 # total number of generated samples = n_batches * args.batch_size
evaluate(test_data_loader, generator, args.raw_dir, viz_dir, follow_batch, device_ids, number_of_batches=n_batches,trunc=trunc_num)
generate_multiple_outputs_from_batch(variation_test_batch1, args.variation_num, generator, args.raw_dir, var_viz_dir1, follow_batch, device_ids, trunc=trunc_num)
generate_multiple_outputs_from_batch(variation_test_batch2, args.variation_num, generator, args.raw_dir, var_viz_dir2, follow_batch, device_ids, trunc=trunc_num)
