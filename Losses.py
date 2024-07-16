## packages
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

# source: OpenOOD
class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04):  # 0.04 is used in the paper and benchmark
        super(LogitNormLoss, self).__init__()
        self.tau = tau

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.tau
        return F.cross_entropy(logit_norm, target)  # Should be in the NN losses


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        return F.cross_entropy(x, target)

class CE_loss(nn.Module):
    def __init__(self):
        super(CE_loss, self).__init__()

    def foward(soft_output_pred, soft_output):
            CE_loss = - torch.sum(soft_output.squeeze() * torch.log(soft_output_pred))
            return CE_loss

class UCE_loss(nn.Module):
    def __init__(self, output_dim):
        super(UCE_loss, self).__init__()
        self.output_dim = output_dim
        #TODO add self.regr and find out what this is
    def forward(self, alpha, soft_output): #Need to check what soft output, UPDATE it is Y_train_hot = torch.zeros(Y_train.shape[0], train_loader.dataset.output_dim) - Y_train_hot.scatter_(1, Y_train, 1) NEEDS TO BE A ONE HOT ENCODING
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.output_dim)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)
        return UCE_loss
