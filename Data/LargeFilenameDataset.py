import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image


class LargeFilenameDataset(Dataset):
    def __init__(self, data_dir, fname_list):
        self.data_dir = data_dir
        self.fname_list = fname_list

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, i):
        data = torch.load(os.path.join(self.data_dir, self.fname_list[i]))

        # # 0301 for room dis
        # # batch.voxel_label_program_node_index  --> 1 graph has -1, but in last batch
        # gt_att = torch.zeros(data.cross_edge_program_index_select.shape)
        # ce_ptr, gt_ptr = 0, 0
        # while ce_ptr < len(gt_att) and gt_ptr < len(data.voxel_label_program_node_index):
        #     if data.voxel_label_program_node_index[gt_ptr] == 0:
        #         while ce_ptr < len(gt_att) and data.cross_edge_voxel_index_select[ce_ptr] == gt_ptr:
        #             ce_ptr += 1
        #         gt_ptr += 1
        #     else:
        #         cur_voxel_id = data.cross_edge_voxel_index_select[ce_ptr]
        #         while ce_ptr < len(gt_att) and data.cross_edge_voxel_index_select[ce_ptr] == cur_voxel_id:
        #             if data.cross_edge_program_index_select[ce_ptr] == data.voxel_label_program_node_index[gt_ptr]:
        #                 gt_att[ce_ptr] = 1.0
        #             ce_ptr += 1
        #         gt_ptr += 1
        # data["gt_att"] = gt_att

        return data


