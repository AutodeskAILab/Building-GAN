import torch
import numpy as np
from pathlib import Path
import json

voxel_dir = "6types-raw_data/voxel_data"
uni_voxel_label_dir = "6types-raw_data_uniform/uni_voxel_label"
Path(uni_voxel_label_dir).mkdir(parents=True, exist_ok=True)
uni_voxel_id_dir = "6types-raw_data_uniform/uni_voxel_id"
Path(uni_voxel_id_dir).mkdir(parents=True, exist_ok=True)

k = 120000

for i in range(k):

    print(i, "/120000")

    with open(voxel_dir + '/voxel_{}.json'.format(str(i).zfill(5))) as f:
        voxels = json.load(f)["voxel_node"]

    uni_voxels_label = np.ones((50, 40, 40))
    uni_voxels_label = - uni_voxels_label

    uni_voxels_id = np.zeros_like(uni_voxels_label)

    for id, voxel in enumerate(voxels):

        sz = np.array(range(int(voxel["coordinate"][0]), int(voxel["coordinate"][0] + voxel["dimension"][0])))
        sy = np.array(range(int(voxel["coordinate"][1]), int(voxel["coordinate"][1] + voxel["dimension"][1])))
        sx = np.array(range(int(voxel["coordinate"][2]), int(voxel["coordinate"][2] + voxel["dimension"][2])))
        uni_voxels_label[np.ix_(sz, sy, sx)] = voxel["type"]
        uni_voxels_id[np.ix_(sz, sy, sx)] = id

    uni_voxels_label_tensor = torch.from_numpy(uni_voxels_label)
    uni_voxels_label_tensor.type(torch.DoubleTensor)

    uni_voxels_id_tensor = torch.from_numpy(uni_voxels_id)
    uni_voxels_id_tensor.type(torch.long)

    torch.save(uni_voxels_label_tensor, uni_voxel_label_dir + "/uni_voxel_label_{}.pt".format(str(i).zfill(6)))
    torch.save(uni_voxels_id_tensor, uni_voxel_id_dir + "/uni_voxel_id_{}.pt".format(str(i).zfill(6)))

