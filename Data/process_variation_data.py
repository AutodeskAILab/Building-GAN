"""
This code generates variation samples by fixing the graph data and varying the random variable z.
"""

from Data.GraphConstructor import GraphConstructor
import os
import json
import torch
from shutil import copyfile


num_VG = 1
num_z = 10

num_of_data = 1000
data_dir = "./6types-raw_data/sum"
output_dir = "./6types-processed-variation-v1-z10_data"
os.mkdir(output_dir)

raw_dir = "./6types-raw-variation-v1-z10_data"
raw_voxel = raw_dir + "/voxel_data"
raw_global = raw_dir + "/global_graph_data"
raw_local = raw_dir + "/local_graph_data"
os.mkdir(raw_dir)
os.mkdir(raw_voxel)
os.mkdir(raw_local)
os.mkdir(raw_global)

voxel_dir = "./variation_test"

cnt = 0
for data_id in range(num_of_data):
    print("generate " + str(num_VG*num_z) + "data from Data" + str(data_id))
    data_id_str = str(96000 + data_id).zfill(GraphConstructor.data_id_length)
    with open(os.path.join(data_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str + ".json")) as f:
        global_graph = json.load(f)  # Keys: FAR, site area, Global node (Room Type, target_ratio, connections)
    with open(os.path.join(data_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json")) as f:
        local_graph = json.load(f)
    program_graph = GraphConstructor.construct_program_graph(local_graph, global_graph)

    num_of_story = max(program_graph["story_level_feature"]) + 1  # starting from 1 ~ 11
    voxel_graph_dir = os.path.join(voxel_dir, str(num_of_story), "voxel_data")
    for i in range(num_VG):
        with open(os.path.join(voxel_graph_dir, GraphConstructor.voxel_graph_prefix + str(i).zfill(GraphConstructor.data_id_length) + ".json")) as f:
            raw_voxel_graph = json.load(f)
        voxel_graph = GraphConstructor.construct_voxel_graph(raw_voxel_graph)

        for _ in range(num_z):
            g = GraphConstructor.tensorfy(program_graph, voxel_graph, str(cnt).zfill(GraphConstructor.data_id_length))
            output_fname = "data" + str(cnt).zfill(GraphConstructor.data_id_length) + ".pt"
            torch.save(g, os.path.join(output_dir, output_fname))

            # copy json files and rename them based on current cnt
            global_copy_f = os.path.join(data_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str + ".json")
            local_copy_f = os.path.join(data_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json")
            voxel_copy_f = os.path.join(voxel_graph_dir, GraphConstructor.voxel_graph_prefix + str(i).zfill(GraphConstructor.data_id_length) + ".json")

            global_paste_f = os.path.join(raw_global, "graph_global_"+str(cnt).zfill(GraphConstructor.data_id_length)+".json")
            local_paste_f = os.path.join(raw_local, "graph_local_"+str(cnt).zfill(GraphConstructor.data_id_length)+".json")
            voxel_paste_f = os.path.join(raw_voxel, "voxel_" + str(cnt).zfill(GraphConstructor.data_id_length) + ".json")

            copyfile(global_copy_f, global_paste_f)
            copyfile(local_copy_f, local_paste_f)
            copyfile(voxel_copy_f, voxel_paste_f)

            cnt += 1
