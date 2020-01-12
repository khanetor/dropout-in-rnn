import torch
from torch import nn


class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(MyLSTM, self).__init__()

        dropout = 0.5
        self.bernoulli_x = torch.distributions.Bernoulli(
            torch.tensor([dropout, dropout, dropout])
        )
        self.bernoulli_h = torch.distributions.Bernoulli(
            torch.tensor([dropout, dropout, dropout, dropout])
        )

        self.iter = 10

        self.Wi = torch.randn((input_size, hidden_size))
        self.Ui = torch.randn((hidden_size, hidden_size))

        self.Wf = torch.randn((input_size, hidden_size))
        self.Uf = torch.randn((hidden_size, hidden_size))

        self.Wo = torch.randn((input_size, hidden_size))
        self.Uo = torch.randn((hidden_size, hidden_size))

        self.Wg = torch.randn((input_size, hidden_size))
        self.Ug = torch.randn((hidden_size, hidden_size))

    def forward(self, input, hx=None):
        """
        input shape (sequence, batch, input dimension)
        output shape (sequence, batch, output dimension)
        """

        T, B, _ = input.shape

        if hx is None:
            hx = torch.zeros((self.iter, T + 1, 10, 4))
        else:
            hx = hx.unsqueeze(0).repeat(self.iter, T + 1, 10, 4)

        c = torch.zeros((self.iter, T + 1, 10, 4))
        o = torch.zeros((self.iter, T, 10, 4))

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
