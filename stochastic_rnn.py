"""Dropout variant of RNN layers
Binary dropout is applied even in inference
"""
from typing import Optional, Tuple
import torch
from torch import nn, Tensor


class StochasticLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        """
        Args:
        - dropout_rate: should be between 0 and 1
        """
        super(StochasticLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if not 0 <= dropout_rate <= 1:
            raise Exception("Dropout rate should be between 0 and 1")
        self.dropout = dropout_rate
        if input_size == 1:
            self.bernoulli_x = torch.distributions.Bernoulli(1.0)
        else:
            self.bernoulli_x = torch.distributions.Bernoulli(1 - self.dropout)
        self.bernoulli_h = torch.distributions.Bernoulli(1 - self.dropout)

        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Wg = nn.Linear(self.input_size, self.hidden_size)
        
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        input shape (sequence, batch, input dimension)
        output shape (sequence, batch, output dimension)
        return output, (hidden_state, cell_state)
        """

        T, B = input.shape[0:2]

        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, dtype=input.dtype)
            c_t = torch.zeros(B, self.hidden_size, dtype=input.dtype)
        else:
            h_t, c_t = hx

        hn = torch.empty(T, B, self.hidden_size, dtype=input.dtype)

        # Dropout masks for 4 gates, scale input by 1 / (1 - p)
        GATES = 4
        zx = self.bernoulli_x.sample((GATES, B, self.input_size)) / (1 - self.dropout)
        zh = self.bernoulli_h.sample((GATES, B, self.hidden_size)) / (1 - self.dropout)

        for t in range(T):
            x_i, x_f, x_o, x_g = (input[t] * zx[m] for m in range(GATES))
            h_i, h_f, h_o, h_g = (h_t * zh[m] for m in range(GATES))

            i = torch.sigmoid(self.Ui(h_i) + self.Wi(x_i))
            f = torch.sigmoid(self.Uf(h_f) + self.Wf(x_f))
            o = torch.sigmoid(self.Uo(h_o) + self.Wo(x_o))
            g = torch.tanh(self.Ug(h_g) + self.Wg(x_g))

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            hn[t] = h_t
        
        return hn, (h_t, c_t)


class StochasticLSTM(nn.Module):
    """LSTM stacked layers with dropout and MCMC"""

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, num_layers: int=1):
        super(StochasticLSTM, self).__init__()
        self.num_layers = num_layers
        self.first_layer = StochasticLSTMCell(input_size, hidden_size, dropout_rate)
        self.hidden_layers = nn.ModuleList([StochasticLSTMCell(hidden_size, hidden_size, dropout_rate) for i in range(num_layers-1)])
    
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        #h_n, c_n = [], []
        B = input.shape[1]
        h_n = torch.empty(self.num_layers, B, self.first_layer.hidden_size)
        c_n = torch.empty(self.num_layers, B, self.first_layer.hidden_size)
        
        outputs, (h, c) = self.first_layer(input, hx)
        h_n[0] = h
        c_n[0] = c

        for i, layer in enumerate(self.hidden_layers):
            outputs, (h, c) = layer(outputs, (h, c))
            h_n[i+1] = h
            c_n[i+1] = c

        return outputs, (h_n, c_n)
        