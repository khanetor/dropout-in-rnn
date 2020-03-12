import torch
from torch import nn


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
            self.bernoulli_x = torch.distributions.Bernoulli(
                torch.full((self.input_size,), 1.0)
            )
        else:
            self.bernoulli_x = torch.distributions.Bernoulli(
                torch.full((self.input_size,), 1 - self.dropout)
            )
        self.bernoulli_h = torch.distributions.Bernoulli(
            torch.full((self.hidden_size,), 1 - self.dropout)
        )
        

        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Wg = nn.Linear(self.input_size, self.hidden_size)
        
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hx=None):
        """
        input shape (sequence, batch, input dimension)
        output shape (sequence, batch, output dimension)
        return output, (hidden_state, cell_state)
        """

        T, B, _ = input.shape

        if hx is None:
            hx = [torch.zeros((B, self.hidden_size), dtype=input.dtype)]
        else:
            hx = [_h for _h in hx]

        c = [torch.zeros((B, self.hidden_size), dtype=input.dtype)]
        o = []

        # Dropout
        zx = self.bernoulli_x.sample()
        zh = self.bernoulli_h.sample()

        for t in range(T):
            x = input[t] * zx
            h = hx[t] * zh

            i = torch.sigmoid(self.Ui(h) + self.Wi(x))
            f = torch.sigmoid(self.Uf(h) + self.Wf(x))

            o.append(torch.sigmoid(self.Uo(h) + self.Wo(x)))
            g = torch.tanh(self.Ug(h) + self.Wg(x))

            c.append(f * c[t] + i * g)
            hx.append(o[t] * torch.tanh(c[t + 1]))

        o = torch.stack(o)
        c = torch.stack(c[1:])
        hx = torch.stack(hx[1:])
        
        return o, (hx, c)


class StochasticLSTM(nn.Module):
    """Pass input through LSTM several time and find average"""

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super(StochasticLSTM, self).__init__()
        self.layer = StochasticLSTMCell(input_size, hidden_size, dropout_rate)
    
    def forward(self, input, hx=None):
        if self.training:
            loops = 1
        else:
            loops = 10

        oo, hh, cc = [], [], []

        for loop in range(loops):
            o, (hid, c) = self.layer(input, hx)
            oo.append(o)
            hh.append(hid)
            cc.append(c)
        oo = torch.mean(torch.stack(oo), axis=0)
        hh = torch.mean(torch.stack(hh), axis=0)
        cc = torch.mean(torch.stack(cc), axis=0)
        return oo, (hh, cc)
