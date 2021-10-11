from Data.VolumeDesignGraph import VolumeDesignGraph
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter, scatter_max
import torch
import copy


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
    """
    For each program type, we sum all the areas from the corresponding voxel nodes and obtain program weight.
    We can then normalize it and compute FAR.
    """
    device = att.get_device() if att.is_cuda else "cpu"
    # Nv x 1  if voxel node is masked, area = 0; otherwise, area.
    masked_voxel_weight = mask * graph.voxel_feature[:, area_index_in_voxel_feature].view(-1, 1)
    # E x 1   put area values on the cross edge
    painted_voxel_weight = (att * torch.index_select(masked_voxel_weight, 0, graph.cross_edge_voxel_index_select))
    # Np x 1  sum of voxel node areas on the program node
    program_weight = scatter(src=painted_voxel_weight, index=graph.cross_edge_program_index_select, dim=0, dim_size=graph.program_class_feature.shape[0], reduce="sum")
    # Sums the areas of program nodes for each type
    program_class_weight = scatter(src=program_weight, index=graph.program_class_cluster, dim=0, dim_size=graph.program_target_ratio.shape[0], reduce='sum')
    # Sums the total area for each graph
    batch_sum = scatter_add(program_class_weight, graph.program_target_ratio_batch.to(device), dim=0, dim_size=graph.FAR.shape[0])[graph.program_target_ratio_batch]
    # Normalize the program ratio in each graph
    normalized_program_class_weight = program_class_weight / (batch_sum + 1e-16)
    # Compute FAR
    FAR = scatter(src=program_class_weight, index=graph.program_target_ratio_batch, dim=0, dim_size=graph.FAR.shape[0], reduce="sum")
    return normalized_program_class_weight, program_weight, FAR


def find_max_out_program_index(logit, cross_edge_voxel_index, cross_edge_program_index, num_of_voxels):
    """ max_out_program_index (Nv x 1) is the program node index that each voxel node has the max attention. We also compute voxel nodes that are masked (mask["hard"])"""
    _, out_cross_edge_index = scatter_max(logit, index=cross_edge_voxel_index, dim=0, dim_size=num_of_voxels)
    max_out_program_index = cross_edge_program_index[out_cross_edge_index]
    return max_out_program_index


def unbatch_data_to_data_list(batch):
    """
    Modified by torch_geometric.data.batch.to_data_list() since the keys are not in order, so when calling __inc__ and number of nodes are queried,
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


def rebatch_graph_for_multi_gpu(batch, device_ids, follow_batch):
    """
    Given a batch of data, split to multiple mini-batches.
    """
    data_list = unbatch_data_to_data_list(batch)
    mini_batch_size = len(data_list)//len(device_ids)
    mini_batch_list, mini_batch_slices = [], [0]
    for i in range(len(device_ids)):
        mini_batch_list.append(Batch.from_data_list(data_list[i * mini_batch_size: (i+1) * mini_batch_size], follow_batch=follow_batch))
        mini_batch_slices.append((i+1) * mini_batch_size)
    return mini_batch_list, mini_batch_slices


def rebatch_for_multi_gpu(batch, device_ids, follow_batch, *args):
    """
    This function rebatches batched graphs(batch) and other information(*args) based on the given device/new_batch ids.
    Return dimension [M batches x N features (list of batch and args)]

    split_by_graph: FAR, class_weights
    split_by_program_node:  z noise, program_weights
    split_by_voxel_node:    out, mask_hard, max_out_program_index

    example args: batch = batch, args = out, class_weights, program_weights, FAR, max_out_program_index
    return
        g: graph
        ------------------
        o: voxel label (n[type])
        cw: (n[new_proportion] in global graph) -- program class ratio/weight
        pw: (n[region_far] in local graph)
        far: (g[far] in global graph)
        pid: the selected program node id for each voxel node
    """

    # First rebatch the batched graphs
    mini_batch_list, mini_batch_slices = rebatch_graph_for_multi_gpu(batch, device_ids, follow_batch)

    # Then identify batching method for the args
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

    # Save to the output data structure
    ret, data = [], batch.__data_class__()
    for i, (device_id, mini_batch) in enumerate(zip(device_ids, mini_batch_list)):
        placeholder = [mini_batch]
        start, end = mini_batch_slices[i], mini_batch_slices[i+1]
        for arg, key in zip(args, rebatch_type_key):
            mini_arg = arg.narrow(data.__cat_dim__(key, None), batch.__slices__[key][start], batch.__slices__[key][end] - batch.__slices__[key][start])
            placeholder.append(mini_arg)
        try:
            placeholder = [ele.to(device_id) for ele in placeholder]
        except:
            pass
        ret.append(tuple(placeholder))
    return ret


def data_parallel(module, batch, _input, follow_batch, device_ids):
    """
    Reference code for multi-gpu setups. Not used in the this code repo
    """
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(module, device_ids)
    inputs = rebatch_for_multi_gpu(batch, device_ids, follow_batch, *_input)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    return torch.nn.parallel.gather(outputs, output_device)
