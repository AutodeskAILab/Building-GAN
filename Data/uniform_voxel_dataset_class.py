"""
Not used???
"""


import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image


class UniformVoxelDataset(Dataset):
    def __init__(self, data_dir, voxel_dim, num_of_class, voxel_fname_list, augment=True):
        self.data_dir = data_dir
        self.voxel_dim = voxel_dim
        self.num_of_class = num_of_class
        self.voxel_fname_list = voxel_fname_list
        self.augment = augment

    def __len__(self):
        return (len(self.voxel_fname_list))

    def __getitem__(self, i):
        data = torch.load(os.path.join(self.data_dir, self.voxel_fname_list[i]))
        data = data + 1  # -1 is a type
        # data = torch.randint(0, num_of_class, size=(2,2,2))

        # convert to one-hot encoding
        voxel = torch.zeros(self.voxel_dim + (self.num_of_class,))
        data = data.view(data.shape + (1,))
        voxel.scatter_(-1, data.type(torch.LongTensor), 1)
        voxel = voxel.permute(3, 0, 1, 2)  # permute to C x D(Z) x H(Y) x W(X)

        if self.augment:
            rot_num = np.random.randint(0, 4, size=1)[0]
            voxel = torch.rot90(voxel, rot_num, [3, 2])  # rotate x to y
            if bool(random.getrandbits(1)):  # flip x
                voxel = torch.flip(voxel, [3])
            if bool(random.getrandbits(1)):  # flip y
                voxel = torch.flip(voxel, [2])
        return voxel


