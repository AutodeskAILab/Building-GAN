import torch
import os
import json
from Data.GraphConstructor import GraphConstructor
from torch_geometric.data import Batch
from util_graph import get_program_ratio, data_parallel, rebatch_for_multi_gpu


def save_output(batch_size, batch, class_weights, program_weights, FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir, new_data_id_strs=None):
    """
    Save the evaluation results
    """
    if not os.path.exists(os.path.join(output_dir, "global_graph_data")):
        os.mkdir(os.path.join(output_dir, "global_graph_data"))
        os.mkdir(os.path.join(output_dir, "local_graph_data"))
        os.mkdir(os.path.join(output_dir, "voxel_data"))

    num_of_program_node_accum = 0
    batch_all_g = []
    data = rebatch_for_multi_gpu(batch, list(range(batch_size)), follow_batch, out, class_weights, program_weights, FAR, max_out_program_index)

    """
    --- data ---
    g: graph
    o: voxel label (n[type])
    cw: (n[new_proportion] in global graph) -- program class ratio/weight 
    pw: (n[region_far] in local graph)
    far: (g[far] in global graph)
    pid: the selected program node id for each voxel node
    """

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
        d = {}  # program id to its type and type id
        with open(os.path.join(raw_dir, "local_graph_data", GraphConstructor.local_graph_prefix + data_id_str + ".json")) as f:
            local_graph = json.load(f)
        for i, (n, c) in enumerate(zip(local_graph["node"], pw)):
            n["region_far"] = float(c)
            d[i] = [n["type"], n["type_id"]]
        with open(os.path.join(output_dir, "local_graph_data", GraphConstructor.local_graph_prefix + new_data_id_str + ".json"), 'w') as f:
            json.dump(local_graph, f)

        # Modify Voxel data
        with open(os.path.join(raw_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + data_id_str + ".json")) as f:
            voxel_graph = json.load(f)
        for n, label, _pid in zip(voxel_graph["voxel_node"], o, pid):
            query = d[_pid.item() - num_of_program_node_accum]
            n["type"] = query[0] if 1.0 in label else -1  # # label.index(1.0)
            n["type_id"] = query[1] if 1.0 in label else 0
        num_of_program_node_accum += pw.shape[0]
        with open(os.path.join(output_dir, "voxel_data", GraphConstructor.voxel_graph_prefix + new_data_id_str + ".json"),'w') as f:
            json.dump(voxel_graph, f)

        all_graphs = [global_graph, local_graph, voxel_graph]
        batch_all_g.append(all_graphs)
    return batch_all_g


def evaluate(data_loader, generator, raw_dir, output_dir, follow_batch, device_ids, number_of_batches=0, trunc=1.0):
    number_of_batches = min(number_of_batches, len(data_loader))
    device = device_ids[0]
    with torch.no_grad():
        total_inter, total_program_edge = 0, 0
        for i, g in enumerate(data_loader):
            if i >= number_of_batches:
                break
            program_z_shape = [g.program_class_feature.shape[0], generator.noise_dim]
            program_z = torch.rand(tuple(program_z_shape)).to(device)
            voxel_z_shape = [g.voxel_feature.shape[0], generator.noise_dim]
            voxel_z = torch.rand(tuple(voxel_z_shape)).to(device)
            if trunc < 1.0:
                program_z.clamp_(min=-trunc, max=trunc)
                voxel_z.clamp_(min=-trunc, max=trunc)

            g.to(device)
            out, soft_out, mask, att, max_out_program_index = generator(g, program_z, voxel_z)
            inter_edges, missing_edges, gen_edges = check_connectivity(g, max_out_program_index, mask['hard'])
            total_inter += inter_edges.shape[1]
            total_program_edge += g.program_edge.shape[1]
            normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(g, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
            all_g = save_output(data_loader.batch_size, g, normalized_program_class_weight, normalized_program_weight,FAR, max_out_program_index, out, follow_batch, raw_dir, output_dir)

        acc = total_inter/total_program_edge
        print('acc=', acc)
        return all_g


def check_connectivity(g, max_out_program_index, mask):
    """
    Extract connectivity from the generated design
        inter_edges:    program edge observed in the generated output
        missing_edges:  program edges only in the input program graph
        gen_edges:      program edges only in the generated output
    """
    # Look at the g.voxel_edge and see if the two voxel nodes are masked
    voxel_edge_out_mask = mask.reshape([-1])[g.voxel_edge]  # Ev x 2
    sums = torch.sum(voxel_edge_out_mask, dim=0)  # Ev x 1
    masked_edges = g.voxel_edge[:, sums == 2]   # Ev x 2 sums ==2 means voxel edges observed in the generated output

    if masked_edges.shape[1] != 0:
        # Now put program index onto the voxel edge and delete duplicates
        predicted_program_edges = torch.unique(max_out_program_index[masked_edges], dim=1)
        # union of program edges and program edges from the generated output
        mixed_edges = torch.cat((g.program_edge, predicted_program_edges), dim=1)
        unique_mix_edges, mix_counts = mixed_edges.unique(return_counts=True, dim=1)
        inter_edges = unique_mix_edges[:, mix_counts > 1]

        # program edges only in the input program graph
        mixed_gt_edges = torch.cat((g.program_edge, inter_edges), dim=1)
        unique_gt_edges, mix_gt_counts = mixed_gt_edges.unique(return_counts=True, dim=1)
        missing_edges = unique_gt_edges[:, mix_gt_counts == 1]

        # program edges only in the generated output
        mixed_gen_edges = torch.cat((predicted_program_edges, inter_edges), dim=1)
        unique_gen_edges, mix_gen_counts = mixed_gen_edges.unique(return_counts=True, dim=1)
        gen_edges = unique_gen_edges[:, mix_gen_counts == 1]
    else:  # there is no voxel edge
        inter_edges = masked_edges
        missing_edges = g.program_edge
        gen_edges = masked_edges

    return inter_edges, missing_edges, gen_edges


def generate_multiple_outputs_from_batch(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids, trunc=1.0):
    device = device_ids[0]
    batch.to(device)
    with torch.no_grad():

        program_z_shape = [batch.program_class_feature.shape[0], generator.noise_dim]
        program_z = torch.rand(tuple(program_z_shape)).to(device)
        voxel_z_shape = [batch.voxel_feature.shape[0], generator.noise_dim]
        voxel_z = torch.rand(tuple(voxel_z_shape)).to(device)
        if trunc < 1.0:
            program_z.clamp_(min=-trunc, max=trunc)
            voxel_z.clamp_(min=-trunc, max=trunc)

        batch.to(device)
        out, soft_out, mask, att, max_out_program_index = generator(batch, program_z, voxel_z)

        normalized_program_class_weight, normalized_program_weight, FAR = get_program_ratio(batch, att["hard"], mask["hard"], area_index_in_voxel_feature=6)
        save_output(variation_num, batch, normalized_program_class_weight, normalized_program_weight, FAR, max_out_program_index, out, follow_batch,
                    raw_dir, output_dir, new_data_id_strs=list(range(variation_num)))


def generate_multiple_outputs_from_data(data, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids):
    batch = Batch.from_data_list([data for _ in range(variation_num)], follow_batch)
    generate_multiple_outputs_from_batch(batch, variation_num, generator, raw_dir, output_dir, follow_batch, device_ids)

