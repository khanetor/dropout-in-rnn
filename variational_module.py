import torch
from torch import nn


class MCDualOutputModule(nn.Module):
    """Stochastic module that samples from given model
    Assuming that model returns a tuple of 2 tensors"""
    def __init__(self, model: nn.Module, sample_size:int=10):
        super(MCDualOutputModule, self).__init__()
        self.model = model
        self.sample_size = sample_size
    
    def forward(self, *args):
        M, V = [], []

        for i in range(self.sample_size):
            m, v = self.model(*args)
            M.append(m)
            V.append(v)

        return torch.stack(M), torch.stack(V)
