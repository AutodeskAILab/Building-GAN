import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, inits
import torch_geometric as tg
from torch_geometric.utils import softmax, degree
from Data.GraphConstructor import GraphConstructor
from torch_scatter import scatter, scatter_max
from util import gumbel_softmax, softmax_to_hard
from util_graph import find_max_out_program_index
import time

if_norm = False

# 0213 positional encoding
import math
def position_encoder(d_model, max_len=20):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # pe = pe.unsqueeze(0).transpose(0, 1)
    return pe
def get_voxel_floor_level(vfc, vbi):
    # floor ids are serialized in batch, need to subtract the accumulated num of node in each graph
    pool = -tg.nn.max_pool_x(cluster=vbi, x=-vfc, batch=torch.cuda.LongTensor([0] * vfc.shape[0]))[0]  # accumulated num of node = min floor i
    return vfc-pool.index_select(0, vbi)

class LayerNorm(torch.nn.Module):
    """ This is not yet released, but on the GitHub of pytorch geometric """
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.Tensor([in_channels]))
            self.bias = nn.Parameter(torch.Tensor([in_channels]))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        inits.ones(self.weight)
        inits.zeros(self.bias)

    def forward(self, x, batch=None):
        """"""
        if batch is None:
            x = x - x.mean()
            out = x / (x.std(unbiased=False) + self.eps)

        else:
            if batch.get_device() != x.get_device():
                batch = batch.to(x.get_device())
            batch_size = int(batch.max()) + 1
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)
            mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce='add').sum(dim=-1, keepdim=True) / norm
            x = x - mean[batch]
            var = scatter(x * x, batch, dim=0, dim_size=batch_size, reduce='add').sum(dim=-1, keepdim=True)
            var = var / norm
            out = x / (var.sqrt()[batch] + self.eps)
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels})'


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
    pos = feature[:, 3:6]  # 3, 4, 5 are coordinates
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

        agg_class_feature = tg.nn.avg_pool_x(cluster=class_index, x=feature, batch=batch_index, size=GraphConstructor.number_of_class)[0]  # class embedding Nxh
        agg_class_feature = agg_class_feature * class_feature.view(-1, 1)
        pool_class_feature_on_node = agg_class_feature.index_select(0, class_index)
        return self.updateMLP(torch.cat((feature, aggr_out, pool_class_feature_on_node), dim=-1))


class ProgramGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, layer_num):  # hidden_dim = 128
        super(ProgramGNN, self).__init__()
        self.layer_num = layer_num
        self.enc = MLP([input_dim + noise_dim, hidden_dim])
        self.encNorm = LayerNorm(hidden_dim)
        self.block = ProgramBlock(hidden_dim)
        self.blockNorm = nn.Sequential(*[LayerNorm(hidden_dim) for _ in range(layer_num)])

    def forward(self, cf, slf, e, cc, tr, cfb, z):
        """
        cf=graph.program_class_feature, slf=graph.story_level_feature, e=graph.program_edge, cc=graph.program_class_cluster,
        tr=graph.program_target_ratio, cfb=graph.program_class_feature_batch, z=noise
        """
        x = torch.cat([cf, slf.view(-1, 1), z], dim=-1)  # NxF  TODO: Split height and room type?
        x = self.enc(x)
        if if_norm:
            x = self.encNorm(x, cfb)
        for i in range(self.layer_num):
            # start_time = time.time()
            x = x + self.block(feature=x, edge_index=e, class_index=cc, class_feature=tr, batch_index=cfb)
            if if_norm:
                x = self.blockNorm[i](x, cfb)
            # print(" ?-programGNN-prop "+str(i)+" : " + str(time.time() - start_time))
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
        att = (self.theta.view(1, -1) * torch.tanh(self.NN_v(voxel_feature_i) + self.NN_p(program_feature_j))).sum(-1)  # attention on each cross_edge
        soft_att, hard_att = gumbel_softmax(att, cross_edge_voxel_index)
        out_att = {"hard": hard_att.view(-1, 1), "soft": soft_att.view(-1, 1)}

        weighted_program_feature = out_att["soft"] * program_graph_feature[cross_edge_program_index]
        new_voxel_feature = voxel_feature + out_mask["soft"] * scatter(weighted_program_feature, cross_edge_voxel_index, dim=0, dim_size=voxel_feature.shape[0], reduce="sum")

        return out_mask, out_att, new_voxel_feature

    @staticmethod
    def construct_output(program_class_feature, num_voxel, att, mask, cross_edge_program_index, cross_edge_voxel_index):
        max_out_program_index = find_max_out_program_index(att["hard"] * mask["hard"].index_select(0, cross_edge_voxel_index), cross_edge_voxel_index, cross_edge_program_index, num_voxel)

        hard_out_label = att["hard"] * program_class_feature[cross_edge_program_index]  # E x 1  one-hot, diff
        hard_out_label = scatter(hard_out_label, cross_edge_voxel_index, dim=0, dim_size=num_voxel, reduce="sum")  # Nv x 1 one-hot, diff
        masked_hard_label = mask["hard"] * hard_out_label

        soft_label = att["soft"] * program_class_feature[cross_edge_program_index]
        soft_label = scatter(soft_label, cross_edge_voxel_index, dim=0, dim_size=num_voxel, reduce="sum")  # Nv x 1 one-hot, diff
        masked_soft_label = mask["hard"] * soft_label

        return masked_hard_label, masked_soft_label, max_out_program_index.view(-1)


class VoxelBlock(MessagePassing):
    def __init__(self, hidden_dim, if_agg_story_in_update=False, if_edge_index2=False):
        super(VoxelBlock, self).__init__()
        self.aggr = 'mean'
        self.messageMLP = MLP([2 * hidden_dim + 3, hidden_dim])
        self.messageMLP2 = MLP([2 * hidden_dim + 3, hidden_dim])

        # # 0213 ori
        # self.updateMLP = MLP([3 * hidden_dim, hidden_dim], act="leaky") if if_agg_story_in_update or if_edge_index2 else MLP([2 * hidden_dim, hidden_dim])

        # 0213 building feature
        self.updateMLP = MLP([4 * hidden_dim, hidden_dim], act="leaky") if if_agg_story_in_update or if_edge_index2 else MLP([2 * hidden_dim, hidden_dim])

        self.if_agg_story_in_update = if_agg_story_in_update  # True in G, False in D
        self.if_edge_index2 = if_edge_index2                      # False in G, True in D ... this is the dynamic edge

    def forward(self, voxel_feature, edge_index, program_feature, batch_index, position, floor_index, voxel_edge_mask, edge_index2):
        """
        voxel_feature=v, program_feature=x, edge_index=graph.voxel_edge,
        position=pos, floor_index=graph.voxel_floor_cluster, batch_index=graph.voxel_feature_batch
        cross_edge_voxel_index=graph.cross_edge_program_index_select, cross_edge_program_index=graph.cross_edge_voxel_index_select,
        """
        assert(self.if_edge_index2 == (edge_index2 is not None))
        kwargs = {"voxel_feature": voxel_feature, "program_feature": program_feature, "batch_index": batch_index,
                  "position": position, "floor_index": floor_index, "voxel_edge_mask": voxel_edge_mask, "edge_index2": edge_index2}
        return self.propagate(edge_index, size=None, **kwargs)

    def message(self, voxel_feature_j, voxel_feature_i=None, position_i=None, position_j=None):
        return self.messageMLP(torch.cat((voxel_feature_i, voxel_feature_j, position_i-position_j), dim=-1))

    def update(self, aggr_out, voxel_feature=None, program_feature=None, floor_index=None, batch_index=None, position=None, edge_index2=None, ptr=None):
        if self.if_agg_story_in_update:

            # # 0213 original
            # agg_story_feature = tg.nn.avg_pool_x(cluster=floor_index, x=voxel_feature, batch=batch_index)[0]  # class embedding Nxh
            # agg_story_feature = agg_story_feature.index_select(0, floor_index)
            # return self.updateMLP(torch.cat((voxel_feature, aggr_out, agg_story_feature), dim=-1))

            # 0213 building feature
            agg_story_feature, agg_story_feature_batch = tg.nn.avg_pool_x(cluster=floor_index, x=voxel_feature,
                                                                          batch=batch_index)  # class embedding Nxh
            agg_building_feature = tg.nn.avg_pool_x(cluster=agg_story_feature_batch, x=agg_story_feature,
                                                    batch=torch.cuda.LongTensor([0] * agg_story_feature.shape[0]))[0]
            agg_story_feature = agg_story_feature.index_select(0, floor_index)
            agg_building_feature = agg_building_feature.index_select(0, batch_index)
            return self.updateMLP(torch.cat((voxel_feature, aggr_out, agg_story_feature, agg_building_feature-agg_story_feature), dim=-1))


        elif self.if_edge_index2:
            # construct second edge type
            voxel_feature2_i, position2_i = voxel_feature.index_select(0, edge_index2[1]), position.index_select(0, edge_index2[1])
            voxel_feature2_j, position2_j = voxel_feature.index_select(0, edge_index2[0]), position.index_select(0, edge_index2[0])
            msg2 = self.messageMLP2(torch.cat((voxel_feature2_i, voxel_feature2_j, position2_i - position2_j), dim=-1))
            aggr_out2 = self.aggregate(msg2, edge_index2[1], ptr, dim_size=voxel_feature.shape[0])
            return self.updateMLP(torch.cat((voxel_feature, aggr_out, aggr_out2), dim=-1))
        else:
            return self.updateMLP(torch.cat((voxel_feature, aggr_out), dim=-1))
        # if self.if_cross_edge:
        #     # # compute attention
        #     program_feature_j = program_feature.index_select(0, cross_edge_program_index)
        #     voxel_feature_i = voxel_feature.index_select(0, cross_edge_voxel_index)
        #     alpha = (self.att_p * program_feature_j + self.att_v * voxel_feature_i).sum(-1)
        #     alpha = F.leaky_relu(alpha, self.negative_slope)
        #     alpha = softmax(alpha, cross_edge_voxel_index)
        #     att = scatter(src=alpha.view(-1, 1) * program_feature_j, index=cross_edge_voxel_index, dim=0, dim_size=voxel_feature.shape[0], reduce=self.aggr)
        #
        #     # Max doesn't make sense cuz every voxel receives the same fixed info from program graph
        #     # att = scatter(src=program_feature_j, index=cross_edge_voxel_index, dim=0, dim_size=voxel_feature.shape[0], reduce="max")  # simpler than attention
        #     return self.updateMLP(torch.cat((voxel_feature, aggr_out, agg_story_feature, att), dim=-1))
        # else:
        #     return self.updateMLP(torch.cat((voxel_feature, aggr_out, agg_story_feature), dim=-1))

    def aggregate(self, inputs, index, ptr=None, dim_size=None, voxel_edge_mask=None):
        if voxel_edge_mask is not None:
            inputs = inputs * voxel_edge_mask.view(-1, 1)
        return super(VoxelBlock, self).aggregate(inputs, index, ptr, dim_size)


class VoxelGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, noise_dim, layer_num, if_given=False):
        super(VoxelGNN, self).__init__()

        # 0213 positional enconding
        pe = position_encoder(hidden_dim, 20)
        self.register_buffer('pe', pe)

        self.layer_num = layer_num
        self.noise_dim = noise_dim
        self.if_given = if_given
        self.enc = MLP([input_dim - 3 + noise_dim, hidden_dim])
        self.encNorm = LayerNorm(hidden_dim)

        self.attention = Attention(hidden_dim)
        self.block = VoxelBlock(hidden_dim, if_agg_story_in_update=True, if_edge_index2=False)
        self.blockNorm = nn.Sequential(*[LayerNorm(hidden_dim) for _ in range(layer_num)])

    def forward(self, x, v, z, e, cep, cev, vbi, vfc, vfb, g=None):
        """
        x=program_node_feature, v=graph.voxel_feature, e=graph.voxel_edge, cep=graph.cross_edge_program_index_select,
        cev=graph.cross_edge_voxel_index_select, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch
        """

        # 0213 original
        assert bool(g) == self.if_given
        pos, v = extract_pos(v)
        v = self.enc(torch.cat((v, z), dim=-1)) if self.noise_dim != 0 else self.enc(v)
        if if_norm:
            v = self.encNorm(v, vbi)

        # 0213 positional encoding
        v = v + self.pe[get_voxel_floor_level(vfc, vbi), :]  # pos[0] is z coordinate

        # return_mask = None
        # return_att = None

        _, _, v = self.attention(program_graph_feature=x, voxel_feature=v, cross_edge_program_index=cep, cross_edge_voxel_index=cev)

        for i in range(self.layer_num):
            # start_time = time.time()
            v = v + self.block(voxel_feature=v, edge_index=e, program_feature=x, position=pos, floor_index=vfc, batch_index=vfb, voxel_edge_mask=None, edge_index2=None)

            if i % 2 == 0 and i != 0:
                out_mask, out_att, v = self.attention(program_graph_feature=x, voxel_feature=v, cross_edge_program_index=cep, cross_edge_voxel_index=cev)

                # if i == 8:
                #     return_mask = out_mask
                #     return_att = out_att
            # _, _, v = self.attention(program_graph_feature=x, voxel_feature=v, cross_edge_program_index=cep, cross_edge_voxel_index=cev)

            if if_norm:
                v = self.blockNorm[i](v, vbi)
            # print(" ?-voxelGNN-prop " + str(i) + " : " + str(time.time() - start_time))

        # out_mask, out_att, v = self.attention(program_graph_feature=x, voxel_feature=v, cross_edge_program_index=cep, cross_edge_voxel_index=cev)

        # return return_mask, return_att, v
        return out_mask, out_att, v


class VoxelGNN_D(nn.Module):
    def __init__(self, input_feature_dim, input_label_dim, hidden_dim, layer_num, if_edge_index2, if_extract=True, if_given=False):
        super(VoxelGNN_D, self).__init__()

        # 0213 positional enconding
        pe = position_encoder(hidden_dim, 20)
        self.register_buffer('pe', pe)

        self.layer_num = layer_num
        self.if_extract = if_extract
        self.if_given = if_given
        self.feature_enc = MLP([input_feature_dim-3, hidden_dim]) if if_extract else MLP([input_feature_dim, hidden_dim])
        self.label_enc = MLP([input_label_dim, hidden_dim])
        self.encNorm = LayerNorm(2*hidden_dim)
        self.block = VoxelBlock(2*hidden_dim, if_agg_story_in_update=False, if_edge_index2=if_edge_index2)
        self.blockNorm = nn.Sequential(*[LayerNorm(2*hidden_dim) for _ in range(layer_num)])

    def forward(self, v, l, e, e_mask, vbi, vfc, vfb, e2=None, g=None):
        """
        v=graph.voxel_feature, l=out_label or graph.voxel_label,
        e=graph.voxel_edge, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch
        """

        # 0213 original
        assert bool(g) == self.if_given
        if self.if_extract:
            pos, v = extract_pos(v)
        else:
            pos, _ = extract_pos(v)
        v = self.feature_enc(v)

        # 0213 positional encoding
        v = v + self.pe[get_voxel_floor_level(vfc, vbi), :]  # pos[0] is z coordinate

        # 0213 original
        l = self.label_enc(l)
        v = torch.cat([v, l], dim=-1)
        if if_norm:
            v = self.encNorm(v, vbi)

        for i in range(self.layer_num):
            # start_time = time.time()
            v = v + self.block(voxel_feature=v, edge_index=e, program_feature=None, position=pos, floor_index=vfc, batch_index=vfb, voxel_edge_mask=e_mask, edge_index2=e2)
            if if_norm:
                v = self.blockNorm[i](v, vbi)
            # print(" ?-voxelGNN-D-prop " + str(i) + " : " + str(time.time() - start_time))
        return v


class Generator(nn.Module):
    def __init__(self, program_input_dim, voxel_input_dim, hidden_dim, noise_dim, program_layer, voxel_layer, device):  # hidden = 128
        super(Generator, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.programGNN = ProgramGNN(program_input_dim, hidden_dim, noise_dim, program_layer)
        self.voxelGNN = VoxelGNN(voxel_input_dim, hidden_dim, 0, voxel_layer)
        # self.pointer_dec = PointerDecoder(hidden_dim, self.device)

    def forward(self, graph, z):
        # start_time = time.time()
        # print(" G-programGNN: " + str(time.time() - start_time))
        program_node_feature = self.programGNN(cf=graph.program_class_feature, slf=graph.story_level_feature, e=graph.program_edge, cc=graph.program_class_cluster,
                                               tr=graph.program_target_ratio, cfb=graph.program_class_feature_batch, z=z)

        voxel_noise = torch.rand(tuple([graph.voxel_feature.shape[0], self.noise_dim])).to(program_node_feature.get_device())
        mask, att, voxel_node_feature = self.voxelGNN(x=program_node_feature, v=graph.voxel_feature, z=voxel_noise, e=graph.voxel_edge, cep=graph.cross_edge_program_index_select,
                                                      cev=graph.cross_edge_voxel_index_select, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        out, soft_out, max_out_program_index = Attention.construct_output(program_class_feature=graph.program_class_feature, num_voxel=graph.voxel_feature.shape[0], att=att, mask=mask,
                                                                          cross_edge_program_index=graph.cross_edge_program_index_select, cross_edge_voxel_index=graph.cross_edge_voxel_index_select)

        # Aggregate voxel features on program graph; No Transform; For unsupervised learning
        design_mask_id = torch.where(mask["hard"] == 1.0)[0].to(z.get_device())  # since it's from real data, we mask where is exactly 0.0
        masked_out_program_index, masked_voxel_node_feature = max_out_program_index[design_mask_id], voxel_node_feature[design_mask_id]
        pooled_program_feature_from_voxel = scatter(src=masked_voxel_node_feature, index=masked_out_program_index.flatten(), dim=0, dim_size=graph.program_class_feature.shape[0], reduce="sum")

        return out, soft_out, mask, att, max_out_program_index, pooled_program_feature_from_voxel


class Discriminator_Voxel(nn.Module):
    def __init__(self, voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer, act=""):
        super(Discriminator_Voxel, self).__init__()
        self.voxelGNN_D = VoxelGNN_D(voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer, if_edge_index2=False)
        # self.voxelGNN_D2 = VoxelGNN_D(voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer, if_edge_index2=False)

        # act = ""          WGAN GP & LSGAN & hinge
        # act = "sigmoid"   NSGAN
        self.building_dec = MLP([2 * hidden_dim, hidden_dim, 1], act)
        self.story_dec = MLP([2 * hidden_dim, hidden_dim, 1], act)

    def forward(self, graph, out_label, out_program_index, out_mask):
        # ############ Dynamic Edge ########################
        # out_label_i, out_label_j = out_program_index.index_select(0, graph.voxel_edge[0]), out_program_index.index_select(0, graph.voxel_edge[1])
        # # random_int = torch.cuda.FloatTensor(graph.voxel_edge.shape[1]).uniform_()
        # # threshold = 1.0
        # same_program_edge = graph.voxel_edge.index_select(1, torch.where(
        #     torch.eq(out_label_i, out_label_j) & (graph.voxel_edge_mask != 0)  # & (random_int < threshold)
        # )[0]).data
        # # assert(same_program_edge.shape[1] == 0)
        # diff_program_edge = graph.voxel_edge.index_select(1, torch.where(
        #     (out_label_i != out_label_j) | (graph.voxel_edge_mask == 0)   # | (random_int >= threshold)
        # )[0]).data
        #
        # # same_program_edge = graph.voxel_edge.index_select(1, torch.where(random_int>0.5)[0])
        # # diff_program_edge = graph.voxel_edge.index_select(1, torch.where(random_int<0.5)[0])
        #
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=same_program_edge, e2=diff_program_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=graph.voxel_edge_mask, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)

        voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        story_feature = tg.nn.global_max_pool(voxel_node_feature, graph.voxel_floor_cluster)
        graph_feature = tg.nn.global_max_pool(voxel_node_feature, graph.voxel_feature_batch.to(out_label.get_device()))

        # Separate also works, but not better. Use same story edge for story discriminator works better
        # voxel_node_feature2 = self.voxelGNN_D2(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=graph.voxel_edge_mask, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # story_feature = tg.nn.global_max_pool(voxel_node_feature2, graph.voxel_floor_cluster)
        return self.building_dec(graph_feature), self.story_dec(story_feature)

        # %%%%% Experiments when trying to add program discriminator %%%%%
        # out_label_i, out_label_j = out_program_index.index_select(0, graph.voxel_edge[0]), out_program_index.index_select(0, graph.voxel_edge[1])
        # same_program_edge = graph.voxel_edge.index_select(1, torch.where(out_label_i == out_label_j)[0])
        # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=same_program_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=same_program_edge, e_mask=torch.rand(same_program_edge.shape[1], device=out_label.get_device()), vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=(out_label_i == out_label_j).to(torch.float), vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        #
        # # Reconstruct full edges from out_labels --> This works
        # # test_edge = torch.cat((same_program_edge, graph.voxel_edge.index_select(1, torch.where(out_label_i != out_label_j)[0])), dim=-1)
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=test_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        #
        # # These work
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge[:, :50], e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=torch.rand(out_label_i.shape, device=out_label.get_device()), vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # # random_mask = torch.randperm(graph.voxel_edge.shape[1])[:2000].to(out_label.get_device())
        # # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge.index_select(1, random_mask), e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)

        # %%%%% Same story "voxel -> story-> building" outputs corn design%%%%%
        # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e_mask=graph.voxel_edge_mask, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
        # story_feature = tg.nn.global_max_pool(voxel_node_feature, graph.voxel_floor_cluster)
        # story_feature_batch = scatter(src=graph.voxel_feature_batch.to(out_label.get_device()), index=graph.voxel_floor_cluster, reduce='mean')
        # graph_feature = tg.nn.global_max_pool(story_feature, story_feature_batch)


# class Discriminator_Program(nn.Module):
#     def __init__(self, voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer):  # hidden = 128
#         super(Discriminator_Program, self).__init__()
#         self.voxelGNN_D = VoxelGNN_D(voxel_input_feature_dim, voxel_input_label_dim, hidden_dim, voxel_layer, if_edge_index2=True)
#         self.voxel_enc = MLP([voxel_input_feature_dim + voxel_input_label_dim, 2 * hidden_dim])
#         self.program_dec = MLP([2 * hidden_dim, hidden_dim, 1])
#
#     def forward(self, graph, out_label, out_program_index, out_mask):
#         out_label_i, out_label_j = out_program_index.index_select(0, graph.voxel_edge[0]), out_program_index.index_select(0, graph.voxel_edge[1])
#         same_program_edge = graph.voxel_edge.index_select(1, torch.where(torch.eq(out_label_i, out_label_j))[0])
#         diff_program_edge = graph.voxel_edge.index_select(1, torch.where(out_label_i != out_label_j)[0])
#
#         # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=same_program_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
#         # voxel_node_feature = self.voxel_enc(torch.cat([graph.voxel_feature, out_label], dim=-1))
#         # voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=graph.voxel_edge, e2=same_program_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
#         voxel_node_feature = self.voxelGNN_D(v=graph.voxel_feature, l=out_label, e=diff_program_edge, e2=same_program_edge, e_mask=None, vbi=graph.voxel_feature_batch, vfc=graph.voxel_floor_cluster, vfb=graph.voxel_feature_batch)
#
#         design_mask_id = torch.where(out_mask == 1.0)[0].to(out_label.get_device())  # since it's from real data, we mask where is exactly 0.0
#         masked_out_program_index, masked_voxel_node_feature = out_program_index.index_select(0, design_mask_id), voxel_node_feature.index_select(0, design_mask_id)
#         pooled_program_feature_from_voxel = scatter(src=masked_voxel_node_feature, index=masked_out_program_index.flatten(), dim=0, dim_size=graph.program_class_feature.shape[0], reduce="mean")  # use mean so that the embedding is not biased by the amount of connected voxel nodes
#         return self.program_dec(pooled_program_feature_from_voxel)
#
#         # return self.program_dec(pooled_program_feature_from_voxel), self.program_dec(pooled_program_feature_from_voxel), self.program_dec(pooled_program_feature_from_voxel)
#
#         # design_mask_id = torch.where(out_mask == 1.0)[0].to(out_label.get_device())  # since it's from real data, we mask where is exactly 0.0
#         # masked_out_program_index, masked_voxel_node_feature = out_program_index[design_mask_id], voxel_node_feature[design_mask_id]
#         # pooled_program_feature_from_voxel = scatter(src=masked_voxel_node_feature, index=masked_out_program_index, dim=0, dim_size=graph.program_class_feature.shape[0], reduce="mean")
#
#         # final_program_feature = self.programGNN_D(cf=graph.program_class_feature, slf=graph.story_level_feature, e=graph.program_edge, cc=graph.program_class_cluster,
#         #                                           tr=graph.program_target_ratio, cfb=graph.program_class_feature_batch, vf=pooled_program_feature_from_voxel)
#         # # Graph Pooling
#         # graph_feature = tg.nn.global_max_pool(final_program_feature, graph.program_class_feature_batch.to(out_label.get_device()))
#         # # graph_feature = tg.nn.global_mean_pool(final_program_feature, graph.program_class_feature_batch.to(out_label.get_device()))
#         # validity = self.global_dec(graph_feature)
#         #
#         # return validity


# %%%% Old code %%%%
# class ProgramGNN_D(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_num):  # hidden_dim = 128
#         super(ProgramGNN_D, self).__init__()
#         self.layer_num = layer_num
#         self.enc = MLP([input_dim, hidden_dim])
#         self.encNorm = LayerNorm(3 * hidden_dim)
#         self.block = ProgramBlock(3 * hidden_dim)
#         self.blockNorm = nn.Sequential(*[LayerNorm(3 * hidden_dim) for _ in range(layer_num)])
#
#     def forward(self, cf, slf, e, cc, tr, cfb, vf):
#         x = torch.cat([cf, slf.view(-1, 1)], dim=-1)  # NxF
#         x = self.enc(x)
#         x = torch.cat([x, vf], dim=-1)
#         if if_norm:
#             x = self.encNorm(x, cfb)
#         for i in range(self.layer_num):
#             x = x + self.block(feature=x, edge_index=e, class_index=cc, class_feature=tr, batch_index=cfb)
#             if if_norm:
#                 x = self.blockNorm[i](x, cfb)
#         return x


# class PointerDecoder(nn.Module):
#     def __init__(self, hidden_dim, device):
#         super(PointerDecoder, self).__init__()
#         # boolean decoder
#         self.dec = MLP([hidden_dim, hidden_dim//2, 2], "relu")
#         # attention parameters
#         # self.NN_v = nn.Linear(hidden_dim, hidden_dim)
#         # self.NN_p = nn.Linear(hidden_dim, hidden_dim)
#         # self.theta = nn.Parameter(torch.Tensor(1, hidden_dim))
#         # inits.glorot(self.theta)
#         # self.device = device
#
#     def forward(self, program_graph_feature, program_class_feature, voxel_feature, cross_edge_program_index, cross_edge_voxel_index):
#         """
#         program_graph_feature=program_node_feature, program_class_feature=graph.program_class_feature, voxel_feature=voxel_node_feature,
#         cross_edge_program_index=graph.cross_edge_program_index_select, cross_edge_voxel_index=graph.cross_edge_voxel_index_select
#         """
#         soft_mask = torch.nn.functional.gumbel_softmax(self.dec(voxel_feature))
#         hard_mask = softmax_to_hard(soft_mask, dim=-1)
#         mask = {"hard": hard_mask[:, 0].view(-1, 1), "soft": soft_mask[:, 0].view(-1, 1)}
#
#         program_feature_j = program_graph_feature.index_select(0, cross_edge_program_index)
#         voxel_feature_i = voxel_feature.index_select(0, cross_edge_voxel_index)
#         att = (self.theta.view(1, -1) * torch.tanh(self.NN_v(voxel_feature_i) + self.NN_p(program_feature_j))).sum(-1)
#
#         soft_att, hard_att = gumbel_softmax(att, cross_edge_voxel_index)
#         att = {"hard": hard_att.view(-1, 1), "soft": soft_att.view(-1, 1)}
#         max_out_program_index = find_max_out_program_index(att["hard"] * mask["hard"].index_select(0, cross_edge_voxel_index),
#                                                            cross_edge_voxel_index, cross_edge_program_index, voxel_feature.shape[0])
#
#         hard_out_label = att["hard"] * program_class_feature[cross_edge_program_index]                                           # E x 1  one-hot, diff
#         hard_out_label = scatter(hard_out_label, cross_edge_voxel_index, dim=0, dim_size=voxel_feature.shape[0], reduce="sum")   # Nv x 1 one-hot, diff
#         masked_hard_label = mask["hard"] * hard_out_label
#
#         soft_label = att["soft"] * program_class_feature[cross_edge_program_index]                                      # E x 1  one-hot, diff
#         soft_label = scatter(soft_label, cross_edge_voxel_index, dim=0, dim_size=voxel_feature.shape[0], reduce="sum")  # Nv x 1 one-hot, diff
#         masked_soft_label = mask["hard"] * soft_label
#
#         """
#         masked_out_label: masked, one-hot, diff, label (Nv x C) for Discriminator and visualization
#         mask            : hard/soft  # 0 = no color
#         att             : hard/soft  (E x 1), should be used with cross_edge
#         max_out_program_index : this is masked
#         """
#
#         return masked_hard_label, masked_soft_label, mask, att, max_out_program_index


