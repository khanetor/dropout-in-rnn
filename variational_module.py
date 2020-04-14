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


class MCModuleMixin:
    """Monte Carlo mixin
    This mixin provide a method `sample` to sample from defined model
    Use this Mixin by inheriting this class
    Assuming that model returns a tuple of 2 tensors"""

    def get_output_shape(self, *args):
        "Override this to get output dimensions."
        raise NotImplementedError("Need to define output shape")
    
    def sample(self, T:int, *args):
        # Construct empty outputs
        shape_m, shape_v = self.get_output_shape(*args)
        M, V = torch.empty(T, *shape_m), torch.empty(T, *shape_v)
        
        for t in range(T):
            M[t], V[t] = self(*args)
        
        return M, V