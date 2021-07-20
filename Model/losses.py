import torch
import torch.nn as nn
import torch_geometric as tg
from util import *
from util_graph import get_program_ratio


class VolumetricDesignLoss_D(nn.Module):
    def __init__(self, gan_loss, gp_lambda=10):
        super(VolumetricDesignLoss_D, self).__init__()
        self.gan_loss = gan_loss
        self.gp_lambda = gp_lambda

    def forward(self, real_validity_voxel, fake_validity_voxel, gp=None):
        if self.gan_loss == "WGANGP":
            Dv_loss = -torch.mean(real_validity_voxel[0]) - torch.mean(real_validity_voxel[1]) + torch.mean(fake_validity_voxel[0]) + torch.mean(fake_validity_voxel[1])
            return Dv_loss + (self.gp_lambda * gp if gp is not None else 0)

        device = real_validity_voxel[0].get_device()
        valid0 = torch.FloatTensor(real_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        valid1 = torch.FloatTensor(real_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)
        fake0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(0.0).to(device)
        fake1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(0.0).to(device)

        if self.gan_loss == "NSGAN":  # NS GAN  log(D(x))+log(1-D(G(z)))
            loss = nn.BCELoss()
            return loss(real_validity_voxel[0], valid0) + loss(real_validity_voxel[1], valid1) + loss(fake_validity_voxel[0], fake0) + loss(fake_validity_voxel[1], fake1)
        elif self.gan_loss == "LSGAN":  # LS GAN  (D(x)-1)^2 + (D(G(z)))^2
            loss = nn.MSELoss()
            return 0.5 * (loss(real_validity_voxel[0], valid0) + loss(real_validity_voxel[1], valid1) + loss(fake_validity_voxel[0], fake0) + loss(fake_validity_voxel[1], fake1))
        elif self.gan_loss == "hinge":  # SA GAN
            loss = nn.ReLU()
            return loss(1.0 - real_validity_voxel[0]).mean() + loss(1.0 - real_validity_voxel[1]).mean() + loss(fake_validity_voxel[0] + 1.0).mean() + loss(fake_validity_voxel[1] + 1.0).mean()
        else:
            raise TypeError("self.gan_loss is not valid")


class VolumetricDesignLoss_G(nn.Module):
    def __init__(self, lp_weight, tr_weight, far_weight, embedding_dim, sample_size, similarity_fun, gan_loss, lp_loss, hinge_margin):  # hidden_dim = 128
        super(VolumetricDesignLoss_G, self).__init__()
        self.gan_loss = gan_loss
        self.tr_weight = tr_weight
        self.lp_weight = lp_weight
        self.far_weight = far_weight

    def forward(self, fake_validity_voxel, graph, att, mask, area_index_in_voxel_feature):
        device = att.get_device() if att.is_cuda else "cpu"
        target0 = torch.FloatTensor(fake_validity_voxel[0].shape[0], 1).fill_(1.0).to(device)
        target1 = torch.FloatTensor(fake_validity_voxel[1].shape[0], 1).fill_(1.0).to(device)

        # adversarial loss
        if self.gan_loss == "WGANGP":
            adversarial_loss_voxel = -torch.mean(fake_validity_voxel[0])-torch.mean(fake_validity_voxel[1])
            adversarial_loss = adversarial_loss_voxel  # + adversarial_loss_program
        elif self.gan_loss == "NSGAN":
            loss = nn.BCELoss()
            adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        elif self.gan_loss == "LSGAN":
            loss = nn.MSELoss()
            adversarial_loss = loss(fake_validity_voxel[0], target0) + loss(fake_validity_voxel[1], target1)
        elif self.gan_loss == "hinge":
            adversarial_loss = -torch.mean(fake_validity_voxel[0])-torch.mean(fake_validity_voxel[1])

        # auxiliary loss
        normalized_program_class_weight, _, FAR = get_program_ratio(graph, att, mask, area_index_in_voxel_feature)
        target_ratio_loss = torch.Tensor([0]).to(device) if self.tr_weight == 0 else self.tr_weight * nn.functional.smooth_l1_loss(normalized_program_class_weight.flatten(), graph.program_target_ratio)
        far_loss = torch.Tensor([0]).to(device) if self.far_weight == 0 else self.far_weight * nn.functional.smooth_l1_loss(FAR.view(FAR.shape[0]), graph.FAR)
        total_loss = adversarial_loss + target_ratio_loss + far_loss

        return total_loss, adversarial_loss, target_ratio_loss, far_loss


def compute_gradient_penalty(Dv, batch, label, out):
    # Interpolated sample
    device = out.get_device()
    u = torch.FloatTensor(label.shape[0], 1).uniform_(0, 1).to(device)  # weight between model and gt label
    mixed_sample = torch.autograd.Variable(label * u + out * (1 - u), requires_grad=True).to(device)  # Nv x C
    mask = (mixed_sample.max(dim=-1)[0] != 0).type(torch.float32).view(-1, 1)
    sample = softmax_to_hard(mixed_sample, -1) * mask

    # compute gradient penalty
    dv_loss = Dv(batch, sample)
    grad_b = torch.autograd.grad(outputs=dv_loss[0], inputs=sample, grad_outputs=torch.ones(dv_loss[0].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    grad_s = torch.autograd.grad(outputs=dv_loss[1], inputs=sample, grad_outputs=torch.ones(dv_loss[1].shape).to(device), retain_graph=True, create_graph=True, only_inputs=True)[0]
    dv_gp_b = ((grad_b.norm(2, 1) - 1) ** 2).mean()
    dv_gp_s = ((grad_s.norm(2, 1) - 1) ** 2).mean()
    dv_gp = dv_gp_b + dv_gp_s

    return dv_gp, dv_gp_b, dv_gp_s


