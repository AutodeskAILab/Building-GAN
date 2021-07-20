import torch
from torch_scatter import scatter_max, scatter_add
from torch_geometric.utils.num_nodes import maybe_num_nodes


def softmax_to_hard(y_soft, dim):
    """the function uses the hard version trick as in gumbel softmax"""
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


def gumbel_softmax(src, index, tau=-1, num_nodes=None):
    """modified from torch_geometric.utils import softmax"""
    num_nodes = maybe_num_nodes(index, num_nodes)

    gumbels = -torch.empty_like(src, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (src + gumbels) / tau  # ~Gumbel(logits,tau)

    out = gumbels - scatter_max(gumbels, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    argmax = scatter_max(out, index, dim=0, dim_size=num_nodes)[1]
    out_hard = torch.zeros_like(out, memory_format=torch.legacy_contiguous_format).scatter_(0, argmax, 1.0)
    out_hard = out_hard - out.detach() + out

    return out, out_hard


