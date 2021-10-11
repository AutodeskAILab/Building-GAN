import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, inits
import torch_geometric as tg
from Data.GraphConstructor import GraphConstructor
from torch_scatter import scatter, scatter_max
from util import gumbel_softmax, softmax_to_hard
from util_graph import find_max_out_program_index
import math


# ----  POSITION ENCODING -------------------------------
def position_encoder(d_model, max_len=20):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def get_voxel_floor_level(vfc, vbi):
    # floor ids are serialized in batch, need to subtract the accumulated num of node in each graph
    pool = -tg.nn.max_pool_x(cluster=vbi, x=-vfc, batch=torch.cuda.LongTensor([0] * vfc.shape[0]))[0]
    return vfc-pool.index_select(0, vbi)
# ---------------------------------------------------------


def MLP(dims, act=""):
    assert len(dims) >= 2
    nn_list = []
    for i in range(len(dims)-1):
        nn_list.append(nn.Linear(dims[i], dims[i+1], bias=True))
        if "leaky" in act:
            nn_list.append(nn.LeakyReLU(negative_slope=0.01))
        elif "relu" in act:
            nn_list.append(nn.ReLU())
        elif "tanh" in act:
            nn_list.append(nn.Tanh())
        elif "sigmoid" in act:
            nn_list.append(nn.Sigmoid())
    return nn.Sequential(*nn_list)


def extract_pos(feature):
    """ Split position features and the rest"""
    pos = feature[:, 3:6]  # 3, 4, 5 are position coordinates
    non_pos = torch.cat((feature[:, 0:3], feature[:, 6:]), dim=-1)
    return pos, non_pos


class ProgramBlock(MessagePassing):
    def __init__(self, hidden_dim):
        super(ProgramBlock, self).__init__()
        self.aggr = 'mean'
        self.messageMLP = MLP([2 * hidden_dim, hidden_dim])
        self.updateMLP = MLP([3 * hidden_dim, hidden_dim], act="leaky")

    def forward(self, feature, edge_index, class_index, class_feature, batch_index):
        """
        feature=x, edge_index=graph.program_edge, class_index=graph.program_class_cluster,
        class_feature=graph.program_target_ratio, batch_index=graph.program_class_feature_batch
        """
        kwargs = {"feature": feature, "class_index": class_index, "class_feature": class_feature, "batch_index": batch_index}
        return self.propagate(edge_index, size=None, **kwargs)

    def message(self, feature_j, feature_i=None):
        return self.messageMLP(torch.cat((feature_i, feature_j), dim=-1))

    def update(self, aggr_out, feature=None, class_index=None, class_feature=None, batch_index=None):
        agg_class_feature = tg.nn.avg_pool_x(cluster=class_index, x=feature, batch=batch_index, size=GraphConstructor.number_of_class)[0] * class_feature.view(-1, 1)  # r_{Cl(i)}c_{i} in equation 4
        pool_class_feature_on_node = agg_class_feature.index_select(0, class_index)
        return self.updateMLP(torch.cat((feature, aggr_out, pool_class_feature_on_node), dim=-1))


class ProgramGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, layer_num):  # hidden_dim = 128
        super(ProgramGNN, self).__init__()
        self.layer_num = layer_num
        self.enc = MLP([input_dim + noise_dim, hidden_dim])
        self.block = ProgramBlock(hidden_dim)

    def forward(self, cf, slf, e, cc, tr, cfb, z):
        """
        cf=graph.program_class_feature, slf=graph.story_level_feature, e=graph.program_edge, cc=graph.program_class_cluster,
        tr=graph.program_target_ratio, cfb=graph.program_class_feature_batch, z=noise
        """
        x = torch.cat([cf, slf.view(-1, 1), z], dim=-1)
        x = self.enc(x)
        for i in range(self.layer_num):
            x = x + self.block(feature=x, edge_index=e, class_index=cc, class_feature=tr, batch_index=cfb)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.dec = MLP([hidden_dim, hidden_dim // 2, 2])
        self.NN_v = nn.Linear(hidden_dim, hidden_dim)
        self.NN_p = nn.Linear(hidden_dim, hidden_dim)
        self.theta = nn.Parameter(torch.Tensor(1, hidden_dim))
        inits.glorot(self.theta)

    def forward(self, program_graph_feature, voxel_feature, cross_edge_program_index, cross_edge_voxel_index):
        """
        program_graph_feature=program_node_feature, program_class_feature=graph.program_class_feature, voxel_feature=voxel_node_feature,
        cross_edge_program_index=graph.cross_edge_program_index_select, cross_edge_voxel_index=graph.cross_edge_voxel_index_select
        """
        soft_mask = torch.nn.functional.gumbel_softmax(self.dec(voxel_feature))
        hard_mask = softmax_to_hard(soft_mask, dim=-1)
        out_mask = {"hard": hard_mask[:, 0].view(-1, 1), "soft": soft_mask[:, 0].view(-1, 1)}

        program_feature_j = program_graph_feature.index_select(0, cross_edge_program_index)
        voxel_feature_i = voxel_feature.index_select(0, cross_edge_voxel_index)
        # compute attention; equation 10
        att = (self.theta.view(1, -1) * torch.tanh(self.NN_v(voxel_feature_i) + self.NN_p(program_feature_j))).sum(-1)
        # equation 11
        soft_att, hard_att = gumbel_softmax(att, cross_edge_voxel_index)
        out_att = {"hard": hard_att.view(-1, 1), "soft": soft_att.view(-1, 1)}

        # equation 12
        weighted_program_feature = out_att["soft"] * program_graph_feature[cross_edge_program_index]
        new_voxel_feature = voxel_feature + out_mask["soft"] * scatter(weighted_program_feature, cross_edge_voxel_index, dim=0, dim_size=voxel_feature.shape[0], reduce="sum")

        return out_mask, out_att, new_voxel_feature

    @staticmethod
    def construct_output(program_class_feature, num_voxel, att, mask, cross_edge_program_index, cross_edge_voxel_index):
        # program node index that each voxel node has the max attention
        max_out_program_index = find_max_out_program_index(att["hard"] * mask["hard"].index_select(0, cross_edge_voxel_index), cross_edge_voxel_index, cross_edge_program_index, num_voxel)

        # If a voxel node is not masked, paste the selected program class feature (i.e. which class) to the voxel node as labels.
        hard_out_label = att["hard"] * program_class_feature[cross_edge_program_index]
        hard_out_label = scatter(hard_out_label, cross_edge_voxel_index, dim=0, dim_size=num_voxel, reduce="sum")
        masked_hard_label = mask["hard"] * hard_out_label

        soft_label = att["soft"] * program_class_feature[cross_edge_program_index]
        soft_label = scatter(soft_label, cross_edge_voxel_index, dim=0, dim_size=num_voxel, reduce="sum")
        masked_soft_label = mask["hard"] * soft_label

        return masked_hard_label, masked_soft_label, max_out_program_index.view(-1)


class VoxelBlock(MessagePassing):
    def __init__(self, hidden_dim, if_agg_story_in_update=True):
        super(VoxelBlock, self).__init__()
        self.aggr = 'mean'
        self.if_agg_story_in_update = if_agg_story_in_update  # True in VoxelGNN_G, False in VoxelGNN_D
        self.messageMLP = MLP([2 * hidden_dim + 3, hidden_dim])
        self.updateMLP = MLP([4 * hidden_dim, hidden_dim], act="leaky") if if_agg_story_in_update else MLP([2 * hidden_dim, hidden_dim])

    def forward(self, voxel_feature, edge_index, batch_index, position, floor_index):
        """
        voxel_feature=v, edge_index=graph.voxel_edge, position=pos, floor_index=graph.voxel_floor_cluster, batch_index=graph.voxel_feature_batch
        cross_edge_voxel_index=graph.cross_edge_program_index_select, cross_edge_program_index=graph.cross_edge_voxel_index_select,
        """
        kwargs = {"voxel_feature": voxel_feature, "batch_index": batch_index, "position": position, "floor_index": floor_index}
        return self.propagate(edge_index, size=None, **kwargs)

    def message(self, voxel_feature_j, voxel_feature_i=None, position_i=None, position_j=None):
        return self.messageMLP(torch.cat((voxel_feature_i, voxel_feature_j, position_i-position_j), dim=-1))

    def update(self, aggr_out, voxel_feature=None, floor_index=None, batch_index=None):
        if self.if_agg_story_in_update:
            # concatenating story feature and building feature to the input of update MLP
            agg_story_feature, agg_story_feature_batch = tg.nn.avg_pool_x(cluster=floor_index, x=voxel_feature, batch=batch_index)  # aggregate features on the same story
            agg_building_feature = tg.nn.avg_pool_x(cluster=agg_story_feature_batch, x=agg_story_feature, batch=torch.cuda.LongTensor([0] * agg_story_feature.shape[0]))[0]  # aggregate all features in the graph
            agg_story_feature = agg_story_feature.index_select(0, floor_index)  # populate the feature to each voxel node
            agg_building_feature = agg_building_feature.index_select(0, batch_index)  # populate the feature to each voxel node
            return self.updateMLP(torch.cat((voxel_feature, aggr_out, agg_story_feature, agg_building_feature-agg_story_feature), dim=-1))
        else:
            return self.updateMLP(torch.cat((voxel_feature, aggr_out), dim=-1))


class VoxelGNN_G(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, layer_num):
        super(VoxelGNN_G, self).__init__()
        self.layer_num = layer_num
        self.noise_dim = noise_dim
        self.register_buffer('pe', position_encoder(hidden_dim, 20))
        self.enc = MLP([input_dim - 3 + noise_dim, hidden_dim])
        self.block = VoxelBlock(hidden_dim, if_agg_story_in_update=True)
        self.attention = Attention(hidden_dim)

    def forward(self, x, v, z, e, cep, cev, vbi, vfc, vfb):
        """
        x=program_node_feature, v=graph.voxel_feature, e=graph.voxel_edge, cep=graph.cross_edge_program_index_select,
        cev=graph.cross_edge_voxel_index_select, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch
        """
        pos, v = extract_pos(v)
        v = self.enc(torch.cat((v, z), dim=-1)) if self.noise_dim != 0 else self.enc(v)
        v = v + self.pe[get_voxel_floor_level(vfc, vbi), :]

        _, _, v = self.attention(program_graph_feature=x, voxel_feature=v, cross_edge_program_index=cep, cross_edge_voxel_index=cev)

        for i in range(self.layer_num):
            # voxel GNN propagation
            v = v + self.block(voxel_feature=v, edge_index=e, position=pos, floor_index=vfc, batch_index=vfb)
            # pointer-based cross-modal module
            if i % 2 == 0 and i != 0:
                out_mask, out_att, v = self.attention(program_graph_feature=x, voxel_feature=v, cross_edge_program_index=cep, cross_edge_voxel_index=cev)

        return out_mask, out_att, v


class VoxelGNN_D(nn.Module):
    def __init__(self, input_feature_dim, input_label_dim, hidden_dim, layer_num):
        super(VoxelGNN_D, self).__init__()
        self.layer_num = layer_num
        self.register_buffer('pe', position_encoder(hidden_dim, 20))
        self.feature_enc = MLP([input_feature_dim-3, hidden_dim])
        self.label_enc = MLP([input_label_dim, hidden_dim])
        self.block = VoxelBlock(2*hidden_dim, if_agg_story_in_update=False)

    def forward(self, v, l, e, vbi, vfc, vfb):
        """
        v=graph.voxel_feature, l=out_label or graph.voxel_label,
        e=graph.voxel_edge, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch
        """
        pos, v = extract_pos(v)
        v = self.feature_enc(v)
        v = v + self.pe[get_voxel_floor_level(vfc, vbi), :]

        l = self.label_enc(l)
        v = torch.cat([v, l], dim=-1)

        for i in range(self.layer_num):
            v = v + self.block(voxel_feature=v, edge_index=e, position=pos, floor_index=vfc, batch_index=vfb)
        return v


class Generator(nn.Module):
    def __init__(self, program_input_dim, voxel_input_dim, hidden_dim, noise_dim, program_layer, voxel_layer, device):  # hidden = 128
        super(Generator, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.programGNN = ProgramGNN(program_input_dim, hidden_dim, noise_dim, program_layer)
        self.voxelGNN = VoxelGNN_G(voxel_input_dim, hidden_dim, 0, voxel_layer)

    def forward(self, graph, program_z, voxel_z):
        program_node_feature = self.programGNN(cf=graph.program_class_feature, slf=graph.story_level_feature, e=graph.program_edge, cc=graph.program_class_cluster,
                                               tr=graph.program_target_ratio, cfb=graph.program_class_feature_batch, z=program_z)
        mask, att, voxel_node_feature = self.voxelGNN(x=program_node_feature, v=graph.voxel_feature, z=voxel_z, e=graph.voxel_edge, cep=graph.cross_edge_program_index_select,
                                                      cev=graph.cross_edge_voxel_index_select, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)

        out, soft_out, max_out_program_index = Attention.construct_output(program_class_feature=graph.program_class_feature, num_voxel=graph.voxel_feature.shape[0], att=att, mask=mask,
                                                                          cross_edge_program_index=graph.cross_edge_program_index_select, cross_edge_voxel_index=graph.cross_edge_voxel_index_select)

        return out, soft_out, mask, att, max_out_program_index


class DiscriminatorVoxel(nn.Module):
    def __init__(self, voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer, act=""):
        super(DiscriminatorVoxel, self).__init__()
        self.voxelGNN_D = VoxelGNN_D(voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer)
        self.building_dec = MLP([2 * hidden_dim, hidden_dim, 1], act)
        self.story_dec = MLP([2 * hidden_dim, hidden_dim, 1], act)

    def forward(self, graph, out_label):
        voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        story_feature = tg.nn.global_max_pool(voxel_node_feature, graph.voxel_floor_cluster)
        graph_feature = tg.nn.global_max_pool(voxel_node_feature, graph.voxel_feature_batch.to(out_label.get_device()))
        return self.building_dec(graph_feature), self.story_dec(story_feature)
