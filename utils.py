"""Dropout RNN training utilities"""

import torch
from torch import nn


class _DropoutLoss(nn.Module):
    def __init__(self, length_scale, precision, dropout):
        super(_DropoutLoss, self).__init__()
        self.length_scale = length_scale
        self.precision = precision
        self.dropout = dropout
        self.loss = None
    
    def forward(self, outputs, labels, named_params, N):
        if self.loss is None:
            raise "Loss function not initialize"

        named_params = [p for p in named_params]
        M = 0.5 * self.length_scale**2 * self.dropout / self.precision / N * \
            torch.norm(
                torch.cat([p.view(-1) for name, p in named_params if name.endswith("weight")]),
                2
            )
        m = 0.5 * self.length_scale**2 / self.precision / N * \
            torch.norm(
                torch.cat([p.view(-1) for name, p in named_params if name.endswith("bias")]),
                2
            )
        return self.loss(outputs, labels) + M + m

class DropoutMSELoss(_DropoutLoss):
    def __init__(self, length_scale, precision, dropout):
        super(DropoutMSELoss, self).__init__(length_scale, precision, dropout)
        self.loss = nn.MSELoss()
    

class DropoutCELoss(_DropoutLoss):
    def __init__(self, length_scale, precision, dropout):
        super(DropoutCELoss, self).__init__(length_scale, precision, dropout)
        self.loss = nn.CrossEntropyLoss()


class DropoutBCELoss(_DropoutLoss):
    def __init__(self, length_scale, precision, dropout):
        super(DropoutBCELoss, self).__init__(length_scale, precision, dropout)
        self.loss = nn.BCELoss()
