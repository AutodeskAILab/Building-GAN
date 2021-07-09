import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

voxel_dir = "100KPretty-raw_data/voxel_data"
uni_voxel_label_dir = "100KPretty-raw_data/uni_voxel_label"
Path(uni_voxel_label_dir).mkdir(parents=True, exist_ok=True)
uni_voxel_id_dir = "100KPretty-raw_data/uni_voxel_id"
Path(uni_voxel_id_dir).mkdir(parents=True, exist_ok=True)

k = 100000

for i in range(k):

    print(i, "/100000")

    with open(voxel_dir + '/voxel_{}.json'.format(str(i).zfill(5))) as f:
        voxels = json.load(f)["voxel_node"]

    uni_voxels_label = np.ones((50, 40, 40))
    uni_voxels_label = - uni_voxels_label

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

    uni_voxels_label_tensor = torch.from_numpy(uni_voxels_label)
    uni_voxels_label_tensor.type(torch.DoubleTensor)

    uni_voxels_id_tensor = torch.from_numpy(uni_voxels_id)
    uni_voxels_id_tensor.type(torch.long)

    torch.save(uni_voxels_label_tensor, uni_voxel_label_dir + "/uni_voxel_label_{}.pt".format(str(i).zfill(5)))
    torch.save(uni_voxels_id_tensor, uni_voxel_id_dir + "/uni_voxel_id_{}.pt".format(str(i).zfill(5)))

    # visualize voxels, just for debug
    # colors = np.transpose(colors, (2, 1, 0))
    # fills = np.transpose(fills, (2, 1, 0))
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(fills, facecolors=colors)
    # plt.show()


    # Visualize the tensors

    # def get_plot_data(label):
    #     f = label!=0
    #     c = (1-(label+1)/7).view(label.shape+(1,)).repeat([1,1,1,3])
    #     # turn zyx to xyz
    #     tp = lambda x : torch.transpose(x, 0, 2)
    #     return tp(f), tp(c)
    #
    # fig2 = plt.figure()
    # ax2 = fig2.gca(projection="3d")
    # f, c = get_plot_data(uni_voxels_label_tensor)
    # ax2.voxels(f, facecolors=c)
    #
    # fig3 = plt.figure()
    # ax3 = fig3.gca(projection="3d")
    # # uni_voxels_label2 = torch.rot90(uni_voxels_label_tensor, 1, [2, 1])  # rotate x-> y 90x1 degrees
    # # uni_voxels_label2 = torch.flip(uni_voxels_label_tensor, (2,))  # flip x
    # uni_voxels_label2 = torch.flip(uni_voxels_label_tensor, (1,))  # flip y
    # f2, c2 = get_plot_data(uni_voxels_label2)
    # ax3.voxels(f2, facecolors=c2)
    #
    # plt.show()
    # print("xxx")
