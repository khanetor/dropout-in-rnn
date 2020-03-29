import torch
from torch import nn, Tensor


class SBCEWithLogitsLoss(nn.Module):
    """Stochastic BCE with Logits Loss"""
    
    def __init__(self, sample_size: int=20):
        super(SBCEWithLogitsLoss, self).__init__()
        self.T = sample_size

    def forward(self, means: Tensor, variances: Tensor, labels: Tensor):
        if len(labels.shape) != 1:
            raise Exception("Unsupported label tensor size. Please use one-dimension tensor.")
        if not torch.all((labels == 1) + (labels == 0)):
            raise Exception("")
        if not (means.shape == variances.shape == labels.shape):
            raise Exception("Input shapes mismatch")

        eps = torch.randn(self.T, len(means))
        X = means + variances * eps
        loss = torch.exp((torch.log(torch.exp(X) + 1) - labels * X)).mean(axis=0)
        loss = torch.log(loss).mean()
        return loss
