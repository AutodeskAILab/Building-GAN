from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter, scatter_max
import torch
import copy
from Data.VolumeDesignGraph import VolumeDesignGraph


def detach_batch(batch):
    detached_batch = Batch()
    detached_batch.__data_class__ = VolumeDesignGraph
    detached_batch.__slices__ = copy.deepcopy(batch.__slices__)

    for key in batch.keys:
        if torch.is_tensor(batch[key]):
            detached_batch[key] = batch[key].detach()
        else:
            detached_batch[key] = copy.deepcopy(batch[key])
    return detached_batch


def get_program_ratio(graph, att, mask, area_index_in_voxel_feature):
    device = att.get_device() if att.is_cuda else "cpu"
    masked_voxel_weight = mask * graph.voxel_feature[:, area_index_in_voxel_feature].view(-1, 1)  # Nv x 1  if masked, area = 0
    painted_voxel_weight = (att * torch.index_select(masked_voxel_weight, 0, graph.cross_edge_voxel_index_select))  # E x 1   after att *, area = area if painted the color
    program_weight = scatter(src=painted_voxel_weight, index=graph.cross_edge_program_index_select, dim=0, dim_size=graph.program_class_feature.shape[0], reduce="sum")
    program_class_weight = scatter(src=program_weight, index=graph.program_class_cluster, dim=0, dim_size=graph.program_target_ratio.shape[0], reduce='sum')
    batch_sum = scatter_add(program_class_weight, graph.program_target_ratio_batch.to(device), dim=0, dim_size=graph.FAR.shape[0])[graph.program_target_ratio_batch]
    normalized_program_class_weight = program_class_weight / (batch_sum + 1e-16)
    # story_sum = scatter_add(src=program_weight, index=graph.program_floor_cluster, dim=0)[graph.program_floor_cluster]
    # normalized_program_weight = program_weight / (story_sum+1e-16)
    FAR = scatter(src=program_class_weight, index=graph.program_target_ratio_batch, dim=0, dim_size=graph.FAR.shape[0], reduce="sum")
    return normalized_program_class_weight, program_weight, FAR


def find_max_out_program_index(logit, cross_edge_voxel_index, cross_edge_program_index, num_of_voxels):
    """To output max_out_program_index with dimension Nv x 1, we don't do mask"""
    _, out_cross_edge_index = scatter_max(logit, index=cross_edge_voxel_index, dim=0, dim_size=num_of_voxels)  # This might included masked voxels
    max_out_program_index = cross_edge_program_index[out_cross_edge_index]
    return max_out_program_index


def find_max_out_program_index_from_label(label, cross_edge_voxel_index_select, cross_edge_program_index_select, program_class_feature):
    device = label.get_device() if label.is_cuda else "cpu"

    """To output max_out_program_index with dimension Nv x 1, we don't do mask"""
    voxel_node_id = torch.arange(0, label.shape[0], dtype=label.dtype, device=device)  # Nv x1
    voxel_pair = torch.cat((voxel_node_id.view(-1, 1), label), dim=-1)  # (voxel_node_id, program_class_feature) Nv x 5
    cross_edge_program_class_feature = torch.index_select(program_class_feature, 0, cross_edge_program_index_select)  # E x 1
    cross_edge_pair = torch.cat((cross_edge_voxel_index_select.type(cross_edge_program_class_feature.dtype).view(-1, 1),
                                 cross_edge_program_class_feature), dim=--1)  # (voxel_node_id, program_class_feature) Ex2

    # find the index in cross_edge_pair that is equal to voxel_pair. cross_edge_program_index_select will be the program_index
    voxel_pair_index, cross_edge_index = torch.where((cross_edge_pair.t() == voxel_pair.unsqueeze(-1)).all(dim=1))  # if not in voxel_pair_index, label = [0, 0, 0...]
    max_out_program_index = torch.zeros((label.size(0)), dtype=cross_edge_voxel_index_select.dtype, device=device)
    max_out_program_index[voxel_pair_index] = cross_edge_program_index_select.index_select(0, cross_edge_index)
    return max_out_program_index


def unbatch_data_to_data_list(batch):
    """
    Modified by torch_geometric.data.batch.to_data_list()  since the keys are not in order, so when calling __inc__ and number of nodes are queried,
    error occurs because the node features might not be populated yet.
    The "xxx_batch" will be dropped in the output as in the data_list. You recreate them when making batches
    """
    if batch.__slices__ is None:
        raise RuntimeError('Cannot reconstruct data list from batch because the batch object was not created using Batch.from_data_list()')
    keys = [key for key in batch.keys if key[-5:] != 'batch']
    cumsum = {key: 0 for key in keys}
    data_list = []
    for i in range(len(batch.__slices__[keys[0]]) - 1):
        data = batch.__data_class__()
        for key in keys:
            if data[key] is not None:
                continue
            if torch.is_tensor(batch[key]):
                data[key] = batch[key].narrow(data.__cat_dim__(key, batch[key]), batch.__slices__[key][i], batch.__slices__[key][i + 1] - batch.__slices__[key][i])
                # if batch[key].dtype != torch.bool: data[key] = data[key] - cumsum[key]
            else:
                data[key] = batch[key][batch.__slices__[key][i]:batch.__slices__[key][i + 1]]
            # cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
        for key in keys:
            if torch.is_tensor(batch[key]) and batch[key].dtype != torch.bool:
                data[key] = data[key] - cumsum[key]
            cumsum[key] = cumsum[key] + data.__inc__(key, data[key])

        data["data_id_str"] = data["data_id_str"][0]  # This is added, otherwise this will be list not str
        data_list.append(data)
    return data_list


def rebatch_data_for_multi_gpu(batch, device_ids, follow_batch):
    data_list = unbatch_data_to_data_list(batch)
    mini_batch_size = len(data_list)//len(device_ids)
    mini_batch_list, mini_batch_slices = [], [0]
    for i in range(len(device_ids)):
        mini_batch_list.append(Batch.from_data_list(data_list[i * mini_batch_size: (i+1) * mini_batch_size], follow_batch=follow_batch))
        mini_batch_slices.append((i+1) * mini_batch_size)
    return mini_batch_list, mini_batch_slices


def rebatch_input_for_multi_gpu(batch, device_ids, follow_batch, *args):
    """
    split_by_program_node:  z noise
    split_by_voxel_node:    out, mask_hard
                            max_out_program_index
    """
    mini_batch_list, mini_batch_slices = rebatch_data_for_multi_gpu(batch, device_ids, follow_batch)

    rebatch_type_key = []
    for arg in args:
        if arg.shape[0] == batch["FAR"].shape[0]:
            rebatch_type_key.append("FAR")
        elif arg.shape[0] == batch["program_class_feature"].shape[0]:
            rebatch_type_key.append("program_class_feature")
        elif arg.shape[0] == batch["voxel_feature"].shape[0]:
            rebatch_type_key.append("voxel_feature")
        elif arg.shape[0] == batch["program_target_ratio"].shape[0]:
            rebatch_type_key.append("program_target_ratio")
        else:
            raise ValueError("unknown input")

    ret, data = [], batch.__data_class__()
    for i, (device_id, mini_batch) in enumerate(zip(device_ids, mini_batch_list)):
        placeholder = [mini_batch]
        start, end = mini_batch_slices[i], mini_batch_slices[i+1]
        for arg, key in zip(args, rebatch_type_key):
            mini_arg = arg.narrow(data.__cat_dim__(key, None), batch.__slices__[key][start], batch.__slices__[key][end] - batch.__slices__[key][start])
            placeholder.append(mini_arg)
        try:  # the device_id is valid to place on GPUs
            placeholder = [ele.to(device_id) for ele in placeholder]
        except:  # the device_id is invalid. Use case might be just unbatching batches and variable pairs
            pass
        ret.append(tuple(placeholder))
    return ret


def data_parallel(module, batch, _input, follow_batch, device_ids):
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(module, device_ids)
    inputs = rebatch_input_for_multi_gpu(batch, device_ids, follow_batch, *_input)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    return torch.nn.parallel.gather(outputs, output_device)
