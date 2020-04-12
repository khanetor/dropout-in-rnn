import torch
from torch import nn, Tensor

class GaussianLoss(nn.Module):
    def forward(self, mean: Tensor, log_var: Tensor, target: Tensor) -> Tensor:
        return 0.5 * torch.mean(torch.exp(-log_var) * (target.view(-1,1)-mean)**2 + log_var)
