import torch
import os
from torch.utils.data import Dataset


class LargeFilenameDataset(Dataset):
    def __init__(self, data_dir, fname_list):
        self.data_dir = data_dir
        self.fname_list = fname_list

    def __len__(self):
        return len(self.fname_list)

    def __getitem__(self, i):
        return torch.load(os.path.join(self.data_dir, self.fname_list[i]))
