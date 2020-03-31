import torch
from torch import nn

class StochasticSingleOutputModule(nn.Module):
    def __init__(self, sample_size:int=10):
        super(StochasticSingleOutputModule, self).__init__()
        self.samples = sample_size
    
    def __call__(self, *args):
        if self.training:
            return self.forward(*args)

        M = []

        for i in range(self.samples):
            m = self.forward(*args)
            M.append(m)

        return torch.stack(M)


class StochasticDualOutputModule(StochasticSingleOutputModule):
    
    def __call__(self, *args):
        if self.training:
            return self.forward(*args)

        M, V = [], []

        for i in range(self.samples):
            m, v = self.forward(*args)
            M.append(m)
            V.append(v)

        return torch.stack(M), torch.stack(V)
