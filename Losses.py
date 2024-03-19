## packages
import torch
from torch import nn
import torch.nn.functional as F


# source: OpenOOD
class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04): # 0.04 is used in the paper and benchmark
        super(LogitNormLoss, self).__init__()
        self.tau = tau

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.tau
        return F.cross_entropy(logit_norm, target)  # Should be in the NN losses
