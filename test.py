import torch
from torch import nn

from rnn import StochasticLSTM

l = StochasticLSTM(input_size=3, hidden_size=4)

t = torch.randn((5, 10, 3))

output, (h, c) = l(t)
