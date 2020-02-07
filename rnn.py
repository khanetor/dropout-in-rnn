import torch
from torch import nn


class StochasticLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super(StochasticLSTM, self).__init__()
        
        self.iter = 10
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.dropout = dropout
        self.bernoulli_x = torch.distributions.Bernoulli(
            torch.full((self.input_size,), self.dropout)
        )
        self.bernoulli_h = torch.distributions.Bernoulli(
            torch.full((hidden_size,), self.dropout)
        )

        self.Wi = torch.randn((self.input_size, self.hidden_size), dtype=torch.double)
        self.Ui = torch.randn((self.hidden_size, self.hidden_size), dtype=torch.double)

        self.Wf = torch.randn((self.input_size, self.hidden_size), dtype=torch.double)
        self.Uf = torch.randn((self.hidden_size, self.hidden_size), dtype=torch.double)

        self.Wo = torch.randn((self.input_size, self.hidden_size), dtype=torch.double)
        self.Uo = torch.randn((self.hidden_size, self.hidden_size), dtype=torch.double)

        self.Wg = torch.randn((self.input_size, self.hidden_size), dtype=torch.double)
        self.Ug = torch.randn((self.hidden_size, self.hidden_size), dtype=torch.double)

    def forward(self, input, hx=None):
        """
        input shape (sequence, batch, input dimension)
        output shape (sequence, batch, output dimension)
        return output, (hidden_state, cell_state)
        """

        T, B, _ = input.shape

        if hx is None:
            hx = torch.zeros((self.iter, T + 1, B, self.hidden_size), dtype=torch.double)
        else:
            hx = hx.unsqueeze(0).repeat(self.iter, T + 1, B, self.hidden_size)

        c = torch.zeros((self.iter, T + 1, B, self.hidden_size), dtype=torch.double)
        o = torch.zeros((self.iter, T, B, self.hidden_size), dtype=torch.double)

        for it in range(self.iter):
            # Dropout
            zx = self.bernoulli_x.sample()
            zh = self.bernoulli_h.sample()

            for t in range(T):
                x = input[t] * zx
                h = hx[it, t] * zh

                i = torch.sigmoid(torch.matmul(h, self.Ui) + torch.matmul(x, self.Wi))
                f = torch.sigmoid(torch.matmul(h, self.Uf) + torch.matmul(x, self.Wf))

                o[it, t] = torch.sigmoid(
                    torch.matmul(h, self.Uo) + torch.matmul(x, self.Wo)
                )
                g = torch.tanh(torch.matmul(h, self.Ug) + torch.matmul(x, self.Wg))

                c[it, t + 1] = f * c[it, t] + i * g
                hx[it, t + 1] = o[it, t] * torch.tanh(c[it, t + 1])

        o = torch.mean(o, axis=0)
        c = torch.mean(c[:, 1:], axis=0)
        hx = torch.mean(hx[:, 1:], axis=0)

        return o, (hx, c)
