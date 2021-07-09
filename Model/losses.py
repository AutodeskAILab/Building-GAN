import torch
import torch.nn as nn
import torch_geometric as tg
# from torch_scatter import scatter_max, scatter
from Model.models import MLP
from Model.models import Attention
from util import *
from util_graph import get_program_ratio, find_max_out_program_index_from_label, data_parallel
from util_eval import check_connectivity

class VolumetricDesignLoss_D(nn.Module):
    def __init__(self, gan_loss, gp_lambda=10):
        super(VolumetricDesignLoss_D, self).__init__()
        self.gan_loss = gan_loss
        self.gp_lambda = gp_lambda

    def forward(self, real_validity_voxel, real_validity_program, fake_validity_voxel, fake_validity_program, gp=None):
        # device = real_validity_voxel[0].get_device()
        # valid0 = torch.FloatTensor(real_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        # valid1 = torch.FloatTensor(real_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)
        # fake0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(0.0).to(device)
        # fake1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(0.0).to(device)

        if self.gan_loss == "WGANGP":  # WGANGP

            # Original (withouth room dis)
            Dv_loss = -torch.mean(real_validity_voxel[0]) - torch.mean(real_validity_voxel[1]) + torch.mean(fake_validity_voxel[0]) + torch.mean(fake_validity_voxel[1])

            # #0301 with room discriminator
            # Dv_loss = -torch.mean(real_validity_voxel[0]) - torch.mean(real_validity_voxel[1]) - 0.5 * torch.mean(
            #     real_validity_voxel[2]) \
            #           + torch.mean(fake_validity_voxel[0]) + torch.mean(fake_validity_voxel[1]) + 0.5 * torch.mean(
            #     fake_validity_voxel[2])

            return Dv_loss + (self.gp_lambda * gp if gp is not None else 0)
            # Dp_loss = -torch.mean(real_validity_program) + torch.mean(fake_validity_program)
            # return Dv_loss + Dp_loss + (self.gp_lambda * gp if gp is not None else 0)
        # elif self.gan_loss == "NSGAN":  # NS GAN  log(D(x))+log(1-D(G(z)))
        #     loss = nn.BCELoss()
        #     return loss(real_validity_voxel[0], valid0) + loss(real_validity_voxel[1], valid1) + loss(fake_validity_voxel[0], fake0) + loss(fake_validity_voxel[1], fake1)
        # elif self.gan_loss == "LSGAN":  # LS GAN  (D(x)-1)^2 + (D(G(z)))^2
        #     loss = nn.MSELoss()
        #     return 0.5 * (loss(real_validity_voxel[0], valid0) + loss(real_validity_voxel[1], valid1) + loss(fake_validity_voxel[0], fake0) + loss(fake_validity_voxel[1], fake1))
        # elif self.gan_loss == "hinge":  # SA GAN
        #     loss = nn.ReLU()
        #     return loss(1.0 - real_validity_voxel[0]).mean() + loss(1.0 - real_validity_voxel[1]).mean() + loss(fake_validity_voxel[0] + 1.0).mean() + loss(fake_validity_voxel[1] + 1.0).mean()


class VolumetricDesignLoss_G(nn.Module):
    def __init__(self, lp_weight, tr_weight, far_weight, embedding_dim, sample_size, similarity_fun, gan_loss, lp_loss, hinge_margin):  # hidden_dim = 128
        super(VolumetricDesignLoss_G, self).__init__()
        self.gan_loss = gan_loss
        self.tr_weight = tr_weight
        self.lp_weight = lp_weight
        self.far_weight = far_weight
        self.link_predictor = link_predictor(embedding_dim, sample_size, similarity_fun, lp_loss, hinge_margin)

    def forward(self, fake_validity_voxel, fake_validity_program, graph, att, mask, pooled_program_feature_from_voxel, max_out_program_index, area_index_in_voxel_feature):
        device = att.get_device() if att.is_cuda else "cpu"
        # target0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        # target1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)

        # adversarial loss
        if self.gan_loss == "WGANGP":        # WGANGP
            adversarial_loss_voxel = -torch.mean(fake_validity_voxel[0])-torch.mean(fake_validity_voxel[1])
            # adversarial_loss_program = -torch.mean(fake_validity_program)
            adversarial_loss = adversarial_loss_voxel  # + adversarial_loss_program
        # elif self.gan_loss == "NSGAN":        # NS GAN  log(D(G(z)))
        #     loss = nn.BCELoss()
        #     adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        # elif self.gan_loss == "LSGAN":        # LS GAN  (D(G(z))-1)^2
        #     loss = nn.MSELoss()
        #     adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        # elif self.gan_loss == "hinge":        # SA GAN
        #     adversarial_loss = -torch.mean(fake_validity_voxel[0])-torch.mean(fake_validity_voxel[1])

        # link prediction loss
        inter_edges, missing_edges, gen_edges = check_connectivity(graph, max_out_program_index, mask)


        # negative_edges = graph.neg_same_story_program_edge
        # link_prediction_loss = torch.Tensor([0]).to(device) if self.lp_weight ==0 else self.lp_weight * self.link_predictor.get_loss(pooled_program_feature_from_voxel, graph.program_edge, negative_edges)
        if gen_edges.shape[1] != 0:
            # 0215 old
            # link_prediction_loss = torch.Tensor([0]).to(device) if self.lp_weight ==0 else self.lp_weight * self.link_predictor.get_loss(pooled_program_feature_from_voxel, inter_edges, missing_edges, gen_edges)
            link_prediction_loss = torch.Tensor([0]).to(
                device) if self.lp_weight == 0 else self.lp_weight * self.link_predictor.get_loss(
                pooled_program_feature_from_voxel, inter_edges, missing_edges, gen_edges)

        else:
            link_prediction_loss = torch.Tensor([0]).to(device)

        # target ratio --> attention-weighted_area
        normalized_program_class_weight, _, FAR = get_program_ratio(graph, att, mask, area_index_in_voxel_feature)
        target_ratio_loss = torch.Tensor([0]).to(device) if self.tr_weight == 0 else self.tr_weight * nn.functional.smooth_l1_loss(normalized_program_class_weight.flatten(), graph.program_target_ratio)
        far_loss = torch.Tensor([0]).to(device) if self.far_weight == 0 else self.far_weight * nn.functional.smooth_l1_loss(FAR.view(FAR.shape[0]), graph.FAR)
        total_loss = adversarial_loss + link_prediction_loss + target_ratio_loss + far_loss

        return total_loss, adversarial_loss, link_prediction_loss, target_ratio_loss, far_loss

#not used
class VolumetricDesignLoss_G_no_attn(nn.Module):
    def __init__(self, lp_weight, tr_weight, embedding_dim, sample_size, similarity_fun, gan_loss, lp_loss, hinge_margin):  # hidden_dim = 128
        super(VolumetricDesignLoss_G_no_attn, self).__init__()
        self.gan_loss = gan_loss
        self.tr_weight = tr_weight
        self.lp_weight = lp_weight
        self.link_predictor = link_predictor(embedding_dim, sample_size, similarity_fun, lp_loss, hinge_margin)

    def forward(self, fake_validity_voxel, fake_validity_program, graph, att, mask, pooled_program_feature_from_voxel, max_out_program_index, area_index_in_voxel_feature):
        # device = att.get_device() if att.is_cuda else "cpu"
        device = fake_validity_voxel[0].get_device() if fake_validity_voxel[0].is_cuda else "cpu"
        # target0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        # target1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)

        # adversarial loss
        if self.gan_loss == "WGANGP":        # WGANGP
            adversarial_loss_voxel = -torch.mean(fake_validity_voxel[0])-torch.mean(fake_validity_voxel[1])
            # adversarial_loss_program = -torch.mean(fake_validity_program)
            adversarial_loss = adversarial_loss_voxel  # + adversarial_loss_program
        # elif self.gan_loss == "NSGAN":        # NS GAN  log(D(G(z)))
        #     loss = nn.BCELoss()
        #     adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        # elif self.gan_loss == "LSGAN":        # LS GAN  (D(G(z))-1)^2
        #     loss = nn.MSELoss()
        #     adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        # elif self.gan_loss == "hinge":        # SA GAN
        #     adversarial_loss = -torch.mean(fake_validity_voxel[0])-torch.mean(fake_validity_voxel[1])

        # # link prediction loss
        # inter_edges, missing_edges, gen_edges = check_connectivity(graph, max_out_program_index, mask)
        #
        #
        # # negative_edges = graph.neg_same_story_program_edge
        # # link_prediction_loss = torch.Tensor([0]).to(device) if self.lp_weight ==0 else self.lp_weight * self.link_predictor.get_loss(pooled_program_feature_from_voxel, graph.program_edge, negative_edges)
        # if gen_edges.shape[1] != 0:
        #     # 0215 old
        #     # link_prediction_loss = torch.Tensor([0]).to(device) if self.lp_weight ==0 else self.lp_weight * self.link_predictor.get_loss(pooled_program_feature_from_voxel, inter_edges, missing_edges, gen_edges)
        #     link_prediction_loss = torch.Tensor([0]).to(
        #         device) if self.lp_weight == 0 else self.lp_weight * self.link_predictor.get_loss(
        #         pooled_program_feature_from_voxel, inter_edges, missing_edges, gen_edges)
        #
        # else:
        link_prediction_loss = torch.Tensor([0]).to(device)

        # target ratio --> attention-weighted_area
        normalized_program_class_weight, _, FAR = get_program_ratio(graph, att, mask, area_index_in_voxel_feature)
        target_ratio_loss = self.tr_weight * nn.functional.smooth_l1_loss(normalized_program_class_weight.flatten(), graph.program_target_ratio)

        total_loss = adversarial_loss + link_prediction_loss + target_ratio_loss

        return total_loss, adversarial_loss, link_prediction_loss, target_ratio_loss

def compute_gradient_penalty(Dv, Dp, batch, label, out, follow_batch, device_ids):
    device = out.get_device()
    u = torch.FloatTensor(label.shape[0], 1).uniform_(0, 1).to(device)
    mixed_sample = torch.autograd.Variable(label * u + out * (1 - u), requires_grad=True).to(device)  # Nv x C shouldn't use softmax after this
    mask = (mixed_sample.max(dim=-1)[0] != 0).type(torch.float32).view(-1, 1)
    sample = softmax_to_hard(mixed_sample, -1) * mask

    max_out_program_index = None

    if len(device_ids) > 1:
        dv_loss = data_parallel(Dv, batch, (sample, max_out_program_index, mask), follow_batch, device_ids)
        # dp_loss = data_parallel(Dv, batch, (sample, max_out_program_index, mask), follow_batch, device_ids)
    else:
        dv_loss = Dv(batch, sample, max_out_program_index, mask)
        # dp_loss = Dp(batch, sample, max_out_program_index, mask)

    grad_b = torch.autograd.grad(outputs=dv_loss[0], inputs=sample, grad_outputs=torch.ones(dv_loss[0].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_s = torch.autograd.grad(outputs=dv_loss[1], inputs=sample, grad_outputs=torch.ones(dv_loss[1].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    Dv_gp = ((grad_b.norm(2, 1) - 1) ** 2).mean() + ((grad_s.norm(2, 1) - 1) ** 2).mean()

    return Dv_gp, ((grad_b.norm(2, 1) - 1) ** 2).mean(), ((grad_s.norm(2, 1) - 1) ** 2).mean(), None

    # grad_p = torch.autograd.grad(outputs=dp_loss, inputs=sample, grad_outputs=torch.ones(dp_loss.shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    # Dp_gp = ((grad_p.norm(2, 1) - 1) ** 2).mean()
    #
    # gradient_penalty = Dv_gp + Dp_gp
    # return gradient_penalty, ((grad_b.norm(2, 1) - 1) ** 2).mean(), ((grad_s.norm(2, 1) - 1) ** 2).mean(), ((grad_p.norm(2, 1) - 1) ** 2).mean()

def compute_gradient_penalty_no_attn(Dv, Dp, batch, label, out, follow_batch, device_ids):
    device = out.get_device()
    u = torch.FloatTensor(label.shape[0], 1).uniform_(0, 1).to(device)
    mixed_sample = torch.autograd.Variable(label * u + out * (1 - u), requires_grad=True).to(device)  # Nv x C shouldn't use softmax after this
    mask = (mixed_sample.max(dim=-1)[0] != 0).type(torch.float32).view(-1, 1)
    sample = softmax_to_hard(mixed_sample, -1) * mask

    max_out_program_index = None

    if len(device_ids) > 1:
        dv_loss = data_parallel(Dv, batch, (sample, max_out_program_index, mask), follow_batch, device_ids)
        # dp_loss = data_parallel(Dv, batch, (sample, max_out_program_index, mask), follow_batch, device_ids)
    else:
        dv_loss = Dv(batch, sample, None, None)
        # dp_loss = Dp(batch, sample, max_out_program_index, mask)

    grad_b = torch.autograd.grad(outputs=dv_loss[0], inputs=sample, grad_outputs=torch.ones(dv_loss[0].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_s = torch.autograd.grad(outputs=dv_loss[1], inputs=sample, grad_outputs=torch.ones(dv_loss[1].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    Dv_gp = ((grad_b.norm(2, 1) - 1) ** 2).mean() + ((grad_s.norm(2, 1) - 1) ** 2).mean()

    return Dv_gp, ((grad_b.norm(2, 1) - 1) ** 2).mean(), ((grad_s.norm(2, 1) - 1) ** 2).mean(), None

def compute_gradient_penalty_room_dis(Dv, Dp, batch, att, mask, follow_batch, device_ids):
    device = att.get_device()
    # # batch.voxel_label_program_node_index  --> 1 graph has -1, but in last batch
    # gt_voxel_program_index = batch.voxel_label_program_node_index * batch.voxel_bool
    # gt_att = torch.zeros(batch.cross_edge_program_index_select.shape).cuda()
    # ce_ptr, gt_ptr = 0, 0
    # while ce_ptr<len(gt_att) and gt_ptr<len(gt_voxel_program_index):
    #     if gt_voxel_program_index[gt_ptr]==0:
    #         while ce_ptr<len(gt_att) and batch.cross_edge_voxel_index_select[ce_ptr]==gt_ptr:
    #             ce_ptr+=1
    #         gt_ptr+=1
    #     else:
    #         cur_voxel_id = batch.cross_edge_voxel_index_select[ce_ptr]
    #         while ce_ptr<len(gt_att) and batch.cross_edge_voxel_index_select[ce_ptr] == cur_voxel_id:
    #             if batch.cross_edge_program_index_select[ce_ptr]== gt_voxel_program_index[gt_ptr]:
    #                 gt_att[ce_ptr]=1.0
    #             ce_ptr+=1
    #         gt_ptr+=1
    gt_att = batch.gt_att
    u_mask = torch.FloatTensor(mask.shape[0], 1).uniform_(0, 1).cuda()
    soft_mask = torch.nn.functional.gumbel_softmax(torch.autograd.Variable(mask * u_mask + batch.voxel_bool.view(-1, 1) * (1 - u_mask), requires_grad=True).cuda())
    hard_mask = softmax_to_hard(soft_mask, dim=-1)
    new_mask = {"hard": hard_mask[:, 0].view(-1, 1), "soft": soft_mask[:, 0].view(-1, 1)}
    u = torch.FloatTensor(gt_att.shape[0], 1).uniform_(0, 1).cuda()
    soft_att, hard_att = gumbel_softmax(torch.autograd.Variable(att * u + gt_att.view(-1, 1) * (1 - u), requires_grad=True).cuda(), batch.cross_edge_voxel_index_select)
    new_att = {"hard": hard_att.view(-1, 1), "soft": soft_att.view(-1, 1)}
    out, _, max_out_program_index = Attention.construct_output(program_class_feature=batch.program_class_feature,
                                                               num_voxel=batch.voxel_feature.shape[0], att=new_att,
                                                               mask=new_mask,
                                                               cross_edge_program_index=batch.cross_edge_program_index_select,
                                                               cross_edge_voxel_index=batch.cross_edge_voxel_index_select)
    if len(device_ids) > 1:
        dv_loss = data_parallel(Dv, batch, (out, max_out_program_index, hard_mask), follow_batch, device_ids)
        # dp_loss = data_parallel(Dv, batch, (sample, max_out_program_index, mask), follow_batch, device_ids)
    else:
        dv_loss = Dv(batch, out, max_out_program_index, hard_mask)
        # dp_loss = Dp(batch, sample, max_out_program_index, mask)
    grad_b = torch.autograd.grad(outputs=dv_loss[0], inputs=out, grad_outputs=torch.ones(dv_loss[0].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_s = torch.autograd.grad(outputs=dv_loss[1], inputs=out, grad_outputs=torch.ones(dv_loss[1].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_r = torch.autograd.grad(outputs=dv_loss[2], inputs=out, grad_outputs=torch.ones(dv_loss[2].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]

    Dv_gp = ((grad_b.norm(2, 1) - 1) ** 2).mean() + ((grad_s.norm(2, 1) - 1) ** 2).mean() + 0.5 * ((grad_r.norm(2, 1) - 1) ** 2).mean()
    return Dv_gp, ((grad_b.norm(2, 1) - 1) ** 2).mean(), ((grad_s.norm(2, 1) - 1) ** 2).mean(), ((grad_r.norm(2, 1) - 1) ** 2).mean(), None


class link_predictor(nn.Module):
    """Reference : https://github.com/facebookresearch/PyTorch-BigGraph/tree/4ca238f7d27a6e720b30587f326dd5a40315cc88"""
    def __init__(self, embedding_dim, sample_size, similarity_fun, loss_fun, hinge_margin, if_neg=True):
        """
        if_negative_sample: If training on functional graph, we only care about positive samples, and use sigmoid loss
        """
        super(link_predictor, self).__init__()
        self.program_emb_enc = MLP([embedding_dim, embedding_dim])
        self.sample_size = sample_size
        self.hinge_margin = hinge_margin
        self.mlp = MLP([2 * embedding_dim, 1])
        self.similarity_fun = similarity_fun
        self.loss_fun = loss_fun
        self.if_neg = if_neg


    def get_loss(self, embeddings, inter_edges, missing_edges, gen_edges):
        embeddings = self.program_emb_enc(embeddings)

        if inter_edges.shape[1] == 0: # sample only from missing if no intersection
            pos_edges = self.sample(missing_edges, self.sample_size)
        else:
            pos_edges = torch.cat((self.sample(inter_edges, self.sample_size // 2), self.sample(missing_edges, self.sample_size // 2)), dim=1)

        pos_scores = self.get_score(embeddings, pos_edges, self.similarity_fun)

        if not self.if_neg:
            if self.loss_fun == "BCE":
                return nn.functional.binary_cross_entropy_with_logits(
                    pos_scores, pos_scores.new_ones(()).expand(pos_scores.shape[0]), reduction="mean")  # should use sum?
            elif self.loss_fun == "l2":  # this should work worse than binary cross entropy
                return torch.norm(pos_scores-1, 2, dim=-1)
        else:

            neg_edges = self.sample(gen_edges, self.sample_size)
            neg_scores = self.get_score(embeddings, neg_edges, self.similarity_fun)

            if self.loss_fun == "hinge":
                # max(0, margin - distance) where distance = f+ - f_
                # margin_ranking_loss = max(0, margin -y * (x1-x2)), y=-1, x1= f_, x2=f+
                loss = nn.functional.margin_ranking_loss(neg_scores, pos_scores, target=pos_scores.new_full((1, 1), -1, dtype=torch.float),
                                                         margin=self.hinge_margin, reduction="mean")  # should use sum?
            elif self.loss_fun == "BCE":
                pos_loss = nn.functional.binary_cross_entropy_with_logits(
                    pos_scores, pos_scores.new_ones(()).expand(pos_scores.shape[0]), reduction="mean")  # should use sum?
                neg_loss = nn.functional.binary_cross_entropy_with_logits(
                    neg_scores, neg_scores.new_zeros(()).expand(neg_scores.shape[0]), reduction="mean", )  # should use sum?
                loss = pos_loss + 1.0 * neg_loss  # GraphSAGE use 1.0, PyTorchBigGraph use 1/N
            elif self.loss_fun == "skipgram":  # exp^(l) = sum{exp^(pos)} / sum{exp^{neg}}
                loss = torch.sum(pos_scores - torch.log(torch.sum(torch.exp(neg_scores))))
            # elif self.loss_fun == "l2":  # this should work worse than binary cross entropy
            #     loss = torch.norm(pos_scores-1, 2, dim=-1) + torch.norm(neg_scores, 2, dim=-1)
            else:
                raise KeyError("Unknown loss type")
            return loss

    @staticmethod
    def sample(edges, num_of_sample):
        sample_index = torch.randint(low=0, high=edges.size(1), size=(num_of_sample, ), device=edges.device)
        return edges.index_select(1, sample_index)  # 2 x E

    def get_score(self, embeddings, edges, similarity_fun):
        src = embeddings.index_select(0, edges[0])  # N x d
        dst = embeddings.index_select(0, edges[1])  # N x d

        if similarity_fun == "dot":  # the embeddings should be normalized (GraphSAGE normalize the embedding)
            score = (src * dst).sum(-1)
        elif similarity_fun == "cos":  # Graph Factorization
            src, dst = nn.functional.normalize(src, p=2, dim=1), nn.functional.normalize(dst, p=2, dim=1)
            score = (src * dst).sum(-1)
        elif similarity_fun == "l2":  # Laplacian Eigenmaps
            score = -torch.norm(src-dst, 2, dim=-1)
        elif similarity_fun == "mlp":  # no one use this for embedding learning
            score = self.mlp(torch.cat((src, dst), dim=-1))
        else:
            raise KeyError("unknown similarity function")

        return score








