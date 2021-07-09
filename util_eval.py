import torch
import numpy as np
import os
import json
from Data.GraphConstructor import GraphConstructor
from torch_geometric.data import Batch
from shutil import copyfile
from util_graph import get_program_ratio, data_parallel, rebatch_input_for_multi_gpu

def save_output_new(batch_size, batch, class_weights, program_weights, FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir, new_data_id_strs=None):
    if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
        os.mkdir(os.path.join(output_dir, "global_graph_data"))
        os.mkdir(os.path.join(output_dir, "local_graph_data"))
        os.mkdir(os.path.join(output_dir, "voxel_data"))
    # indices = torch.arange(max_out_program_index.shape[0]).view(max_out_program_index.shape).to(max_out_program_index.device)
    # tmp = -tg.nn.max_pool_x(cluster=batch.voxel_feature_batch, x=-indices, batch=torch.cuda.LongTensor([0] * indices.shape[0]))[0]
    data = rebatch_input_for_multi_gpu(batch, list(range(batch_size)), follow_batch, out, class_weights, program_weights, FAR, max_out_program_index)
    num_of_program_node_accum = 0
    batch_all_g = []
    for i, (g, o, cw, pw, far, pid) in enumerate(data):
        data_id_str = g["data_id_str"][0]
        new_data_id_str = g["data_id_str"][0] if new_data_id_strs is None else str(new_data_id_strs[i]).zfill(GraphConstructor.data_id_length)
        o = o.cpu().data.numpy().tolist()
        cw, pw, far = cw.cpu().data.numpy(), pw.cpu().data.numpy(), far.item()
        # Modify Global data
        with open(os.path.join(raw_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str + ".json")) as f:
            global_graph = json.load(f)
        global_graph["new_far"] = far
        for n in global_graph["global_node"]:
            n["new_proportion"] = float(cw[n['type']])
        with open(os.path.join(output_dir, "global_graph_data", GraphConstructor.global_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(global_graph, f)
        # Modify Local data
        d = {}
        with open(os.path.join(raw_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json")) as f:
            local_graph = json.load(f)
        for i, (n, c) in enumerate(zip(local_graph["node"], pw)):
            n["region_far"] = float(c)
            d[i] = [n["type"], n["type_id"]]
        with open(os.path.join(output_dir, "local_graph_data", GraphConstructor.local_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(local_graph, f)

        # Modify Voxel data
        with open(
                os.path.join(raw_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + data_id_str + ".json")) as f:
            voxel_graph = json.load(f)
        for n, label, _pid in zip(voxel_graph["voxel_node"], o, pid):
            query = d[_pid.item() - num_of_program_node_accum]
            n["type"] = query[0] if 1.0 in label else -1  # # label.index(1.0)
            n["type_id"] = query[1] if 1.0 in label else 0
        num_of_program_node_accum += pw.shape[0]
        with open(
                os.path.join(output_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + new_data_id_str + ".json"),
                'w') as f:
            json.dump(voxel_graph, f)

        all_graphs = [global_graph, local_graph, voxel_graph]
        batch_all_g.append(all_graphs)
        # return all_graphs
    return batch_all_g

def save_output(batch_size, batch, class_weights, program_weights, FAR, out, follow_batch, raw_dir, output_dir, new_data_id_strs=None):
    if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
        os.mkdir(os.path.join(output_dir, "global_graph_data"))
        os.mkdir(os.path.join(output_dir, "local_graph_data"))
        os.mkdir(os.path.join(output_dir, "voxel_data"))

    data = rebatch_input_for_multi_gpu(batch, list(range(batch_size)), follow_batch, out, class_weights, program_weights, FAR)

    for i, (g, o, cw, pw, far) in enumerate(data):
        data_id_str = g["data_id_str"][0]
        new_data_id_str = g["data_id_str"][0] if new_data_id_strs is None else str(new_data_id_strs[i]).zfill(GraphConstructor.data_id_length)
        o = o.cpu().data.numpy().tolist()
        cw, pw, far = cw.cpu().data.numpy(), pw.cpu().data.numpy(), far.item()

        # Modify Global data
        with open(os.path.join(raw_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str + ".json")) as f:
            global_graph = json.load(f)
        global_graph["far"] = far
        for n in global_graph["global_node"]:
            n["new_proportion"] = float(cw[n['type']])
        with open(os.path.join(output_dir, "global_graph_data", GraphConstructor.global_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(global_graph, f)

        # Modify Local data
        with open(os.path.join(raw_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json")) as f:
            local_graph = json.load(f)
        for n, c in zip(local_graph["node"], pw):
            n["region_far"] = float(c)
        with open(os.path.join(output_dir, "local_graph_data", GraphConstructor.local_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(local_graph, f)

        # Modify Voxel data
        with open(os.path.join(raw_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + data_id_str + ".json")) as f:
            voxel_graph = json.load(f)
        for n, label in zip(voxel_graph["voxel_node"], o):
            n["type"] = label.index(1.0) if 1.0 in label else -1
        with open(os.path.join(output_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(voxel_graph, f)


def evaluate(data_loader, generator, raw_dir, output_dir, follow_batch, device_ids, number_of_batches=0, trunc=1.0):
    number_of_batches = min(number_of_batches, len(data_loader))
    device = device_ids[0]
    with torch.no_grad():
        for i, g in enumerate(data_loader):
            if i >= number_of_batches:
                break
            z_shape = [g.program_class_feature.shape[0], generator.noise_dim]
            # !!!! only for testing
            # torch.manual_seed(0)
            # !!!!
            z = torch.rand(tuple(z_shape)).to(device)
            if trunc < 1.0:
                z.clamp_(min=-trunc, max=trunc)

            if len(device_ids) > 1:
                out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = data_parallel(generator, g, tuple([z]), follow_batch, device_ids)
            else:
                g.to(device)
                out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = generator(g, z)

            normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(g, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
            save_output(data_loader.batch_size, g, normalized_program_class_weight, normalized_program_weight, FAR, out, follow_batch, raw_dir, output_dir)

def evaluate_new(data_loader, generator, raw_dir, output_dir, follow_batch, device_ids, number_of_batches=0, trunc=1.0):
    number_of_batches = min(number_of_batches, len(data_loader))
    device = device_ids[0]
    with torch.no_grad():
        total_inter = 0
        total_program_edge = 0
        for i, g in enumerate(data_loader):
            if i >= number_of_batches:
                break
            z_shape = [g.program_class_feature.shape[0], generator.noise_dim]
            # !!!!!! for testing only
            # torch.manual_seed(5)
            # !!!!!!
            z = torch.rand(tuple(z_shape)).to(device)
            if trunc < 1.0:
                z.clamp_(min=-trunc, max=trunc)

            if len(device_ids) > 1:
                out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = data_parallel(generator, g, tuple([z]), follow_batch, device_ids)
            else:
                g.to(device)
                out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = generator(g, z)

            inter_edges, missing_edges, gen_edges = check_connectivity(g, max_out_program_index, mask['hard'])

            total_inter += inter_edges.shape[1]
            total_program_edge += g.program_edge.shape[1]

            normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(g, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
            # print("b size-----: ", data_loader.batch_size)
            all_g = save_output_new(data_loader.batch_size, g, normalized_program_class_weight, normalized_program_weight, FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir)

        acc = total_inter/total_program_edge
        print('acc=', acc)
        return all_g

def check_connectivity(g, max_out_program_index, mask):

    data_program_edges = g.program_edge
    out_mask_hard = mask.reshape([-1])
    voxel_edge_out_mask = out_mask_hard[g.voxel_edge]
    sums = torch.sum(voxel_edge_out_mask, dim=0)
    masked_edges = g.voxel_edge[:, sums==2]

    if masked_edges.shape[1] != 0:

        predicted_program_edges = torch.unique(max_out_program_index[masked_edges], dim=1)
        mixed_edges = torch.cat((data_program_edges, predicted_program_edges), dim=1)
        unique_mix_edges, mix_counts = mixed_edges.unique(return_counts=True, dim=1)

        inter_edges = unique_mix_edges[:, mix_counts > 1]
        mixed_gt_edges = torch.cat((data_program_edges, inter_edges), dim=1)
        unique_gt_edges, mix_gt_counts = mixed_gt_edges.unique(return_counts=True, dim=1)
        missing_edges = unique_gt_edges[:, mix_gt_counts == 1]

        mixed_gen_edges = torch.cat((predicted_program_edges, inter_edges), dim=1)
        unique_gen_edges, mix_gen_counts = mixed_gen_edges.unique(return_counts=True, dim=1)
        gen_edges = unique_gen_edges[:, mix_gen_counts == 1]

    else:

        inter_edges = masked_edges
        missing_edges = data_program_edges
        gen_edges = masked_edges
    # print(inter_edges.transpose(0,1), inter_edges.shape)
    # print(missing_edges.transpose(0,1), missing_edges.shape)
    # print(gen_edges.transpose(0,1), gen_edges.shape)
    # print('batch acc = ', inter_edges.shape[1]/g.program_edge.shape[1])

    return inter_edges, missing_edges, gen_edges

def generate_multiple_outputs_from_batch(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids, trunc=1.0):
    device = device_ids[0]
    with torch.no_grad():
        z_shape = [batch.program_class_feature.shape[0], generator.noise_dim]
        z = torch.rand(tuple(z_shape)).to(device)
        if trunc < 1.0:
            z.clamp_(min=-trunc, max=trunc)

        if len(device_ids) > 1:
            out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = data_parallel(generator, batch, tuple([z]), follow_batch, device_ids)
        else:
            batch.to(device)
            out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = generator(batch, z)

        normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(batch, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
        save_output(variation_num, batch, normalized_program_class_weight, normalized_program_weight, FAR, out, follow_batch,
                    raw_dir, output_dir, new_data_id_strs=list(range(variation_num)))

def generate_multiple_outputs_from_batch_new(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids, trunc=1.0):
    device = device_ids[0]
    with torch.no_grad():
        z_shape = [batch.program_class_feature.shape[0], generator.noise_dim]
        z = torch.rand(tuple(z_shape)).to(device)
        if trunc < 1.0:
            z.clamp_(min=-trunc, max=trunc)

        if len(device_ids) > 1:
            out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = data_parallel(generator, batch, tuple([z]), follow_batch, device_ids)
        else:
            batch.to(device)
            out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel = generator(batch, z)

        normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(batch, att["hard"], mask["hard"], area_index_in_voxel_feature=6)

        save_output_new(variation_num, batch, normalized_program_class_weight, normalized_program_weight, FAR, max_out_program_index, out, follow_batch,
                    raw_dir, output_dir, new_data_id_strs=list(range(variation_num)))

def generate_multiple_outputs_from_data(data, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids):
    batch = Batch.from_data_list([data for _ in range(variation_num)], follow_batch)
    generate_multiple_outputs_from_batch(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids)


# def save_output(batch, class_weights, program_weights, out, raw_dir, output_dir):
#     out_np = out.cpu().data.numpy().tolist()
#     class_weights = class_weights.cpu().data.numpy()
#     program_weights = program_weights.cpu().data.numpy()
#
#     if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
#         os.mkdir(os.path.join(output_dir, "global_graph_data"))
#         os.mkdir(os.path.join(output_dir, "local_graph_data"))
#         os.mkdir(os.path.join(output_dir, "voxel_data"))
#
#     for i, data_id_str in enumerate(batch.data_id_str):
#         # # Copy Global Local Graph
#         # global_graph_filename = GraphConstructor.global_graph_prefix + data_id_str + ".json"
#         # local_graph_filename = GraphConstructor.local_graph_prefix + data_id_str + ".json"
#         # copyfile(os.path.join(raw_dir, "global_graph_data", global_graph_filename), os.path.join(output_dir, "global_graph_data", global_graph_filename))
#         # copyfile(os.path.join(raw_dir, "local_graph_data", local_graph_filename), os.path.join(output_dir, "local_graph_data", local_graph_filename))
#
#         # Modify Global data
#         with open(os.path.join(raw_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str + ".json")) as f:
#             global_graph = json.load(f)
#         for n in global_graph["global_node"]:
#             n["proportion"] = float(class_weights[n['type']])
#         with open(os.path.join(output_dir, "global_graph_data", GraphConstructor.global_graph_prefix + data_id_str + ".json"), 'w') as f:
#             json.dump(global_graph, f)
#
#         # Modify Local data
#         with open(os.path.join(raw_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json")) as f:
#             local_graph = json.load(f)
#         for n, c in zip(local_graph["node"], program_weights):
#             n["region_far"] = float(c)
#         with open(os.path.join(output_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json"), 'w') as f:
#             json.dump(local_graph, f)
#
#         # Modify Voxel data
#         with open(os.path.join(raw_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + data_id_str + ".json")) as f:
#             voxel_graph = json.load(f)
#         for n, label in zip(voxel_graph["voxel_node"], out_np):
#             n["type"] = label.index(1.0) if 1.0 in label else -1
#         with open(os.path.join(output_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + data_id_str + ".json"), 'w') as f:
#             json.dump(voxel_graph, f)
#         print("X")
#
#
# def evaluate(data_loader, generator, raw_dir, output_dir, number_of_data_to_viz=0):
#     assert data_loader.batch_size == 1, "Evaluate batch size is not 1"
#     number_of_data_to_viz = min(number_of_data_to_viz, len(data_loader))
#
#     with torch.no_grad():
#         for i, g in enumerate(data_loader):
#             if i >= number_of_data_to_viz:
#                 break
#             z_shape = [g.program_class_feature.shape[0], generator.noise_dim]
#             device = g.voxel_feature.get_device() if g.voxel_feature.is_cuda else "cpu"
#             z = torch.rand(tuple(z_shape)).to(device)
#             out, soft_out, mask, att, max_out_program_index = generator(g, z)
#
#             normalized_program_class_weight, normalized_program_weight = get_program_ratio(g, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
#             save_output(g, normalized_program_class_weight, normalized_program_weight, out, raw_dir, output_dir)
