"""
This code converts the voxel graph into regular voxels.
I don't understand why they are called slices
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

voxel_dir = "6types-raw_data/sum/voxel_data"
uni_voxel_label_dir = "6types-raw_data/sum/voxel_data_slice/uni_voxel_label"
Path(uni_voxel_label_dir).mkdir(parents=True, exist_ok=True)
uni_voxel_id_dir = "6types-raw_data/sum/voxel_data_slice/uni_voxel_id"
Path(uni_voxel_id_dir).mkdir(parents=True, exist_ok=True)

# k = 100000
k = 1

for i in range(k):

    print(i, "/100000")

    with open(voxel_dir + '/voxel_{}.json'.format(str(i).zfill(5))) as f:
        voxels = json.load(f)["voxel_node"]

    uni_voxels_label = -np.ones((50, 40, 40))
    uni_voxels_id = np.zeros_like(uni_voxels_label)

    fills = np.zeros_like(uni_voxels_label)
    colors = np.empty(uni_voxels_label.shape, dtype=object)

    colormap = {0: "red", 2:"blue", 3:"green"}

    for id, voxel in enumerate(voxels):
        sz = np.array(range(int(voxel["coordinate"][0]), int(voxel["coordinate"][0] + voxel["dimension"][0])))
        sy = np.array(range(int(voxel["coordinate"][1]), int(voxel["coordinate"][1] + voxel["dimension"][1])))
        sx = np.array(range(int(voxel["coordinate"][2]), int(voxel["coordinate"][2] + voxel["dimension"][2])))
        uni_voxels_label[np.ix_(sz, sy, sx)] = voxel["type"]
        uni_voxels_id[np.ix_(sz, sy, sx)] = id

        if voxel["type"] >= 0:
            colors[np.ix_(sz, sy, sx)] = colormap[voxel["type"]]
            fills[np.ix_(sz, sy, sx)] = 1

    # # save as npy
    # np.save(uni_voxel_label_dir + "/uni_voxel_label_{}.pt".format(str(i).zfill(5)), uni_voxels_label)
    # np.save(uni_voxel_id_dir + "/uni_voxel_id_{}.pt".format(str(i).zfill(5)), uni_voxels_id)

    uni_voxels_label_tensor = torch.from_numpy(uni_voxels_label).type(torch.DoubleTensor)
    uni_voxels_id_tensor = torch.from_numpy(uni_voxels_id).type(torch.long)
    torch.save(uni_voxels_label_tensor, uni_voxel_label_dir + "/uni_voxel_label_{}.pt".format(str(i).zfill(5)))
    torch.save(uni_voxels_id_tensor, uni_voxel_id_dir + "/uni_voxel_id_{}.pt".format(str(i).zfill(5)))

    # visualize voxels, just for debug
    colors = np.transpose(colors, (2, 1, 0))
    fills = np.transpose(fills, (2, 1, 0))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(fills, facecolors=colors)
    plt.show()

