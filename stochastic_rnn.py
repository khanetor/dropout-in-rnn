"""Dropout variant of RNN layers
Binary dropout is applied in training and in inference
User can specify dropout rate, or
dropout rate can be learned during training
"""
from typing import Optional, Tuple
import torch
from torch import nn, Tensor


class StochasticLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: Optional[float]=None):
        """
        Args:
        - dropout_rate: should be between 0 and 1
        """
        super(StochasticLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if dropout_rate is None:
            self.p_logit = nn.Parameter(torch.empty(1).normal_())
        elif not 0 < dropout_rate < 1:
            raise Exception("Dropout rate should be between in (0, 1)")
        else:
            self.p_logit = dropout_rate

        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Wg = nn.Linear(self.input_size, self.hidden_size)
        
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.init_weights()

    def init_weights(self):
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()
        
        self.Wi.weight.data.uniform_(-k,k)
        self.Wi.bias.data.uniform_(-k,k)
        
        self.Wf.weight.data.uniform_(-k,k)
        self.Wf.bias.data.uniform_(-k,k)
        
        self.Wo.weight.data.uniform_(-k,k)
        self.Wo.bias.data.uniform_(-k,k)
        
        self.Wg.weight.data.uniform_(-k,k)
        self.Wg.bias.data.uniform_(-k,k)
        
        self.Ui.weight.data.uniform_(-k,k)
        self.Ui.bias.data.uniform_(-k,k)
        
        self.Uf.weight.data.uniform_(-k,k)
        self.Uf.bias.data.uniform_(-k,k)
        
        self.Uo.weight.data.uniform_(-k,k)
        self.Uo.bias.data.uniform_(-k,k)
        
        self.Ug.weight.data.uniform_(-k,k)
        self.Ug.bias.data.uniform_(-k,k)
        
    # Note: value p_logit at infinity can cause numerical instability
    def _sample_mask(self, B):
        """Dropout masks for 4 gates, scale input by 1 / (1 - p)"""
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)
        GATES = 4
        eps = torch.tensor(1e-7)
        t = 1e-1
        
        ux = torch.rand(GATES, B, self.input_size)
        uh = torch.rand(GATES, B, self.hidden_size)

        if self.input_size == 1:
            zx = (1-torch.sigmoid((torch.log(eps) - torch.log(1+eps)
                                   + torch.log(ux+eps) - torch.log(1-ux+eps))
                                 / t))
        else:
            zx = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)
                                   + torch.log(ux+eps) - torch.log(1-ux+eps))
                                 / t)) / (1-p)
        zh = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)
                               + torch.log(uh+eps) - torch.log(1-uh+eps))
                             / t)) / (1-p)
        return zx, zh

    def regularizer(self):        
        if isinstance(self.p_logit, float):
            p = torch.tensor(self.p_logit)
        else:
            p = torch.sigmoid(self.p_logit)
        
        # Weight
        weight_sum = torch.tensor([
            torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("weight")
        ]).sum() * (1-p)
        
        # Bias
        bias_sum = torch.tensor([
            torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("bias")
        ]).sum()
        
        if isinstance(self.p_logit, float):
            dropout_reg = torch.zeros(1)
        else:
             # Dropout
            dropout_reg = self.input_size * (p * torch.log(p) + (1-p)*torch.log(1-p))
        return weight_sum, bias_sum, dropout_reg
        
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

        # Masks
        zx, zh = self._sample_mask(B)
        
        for t in range(T):
            x_i, x_f, x_o, x_g = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_o, h_g = (h_t * zh_ for zh_ in zh)

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

    def __init__(self, input_size: int, hidden_size: int, dropout_rate:Optional[float]=None, num_layers: int=1):
        super(StochasticLSTM, self).__init__()
        self.num_layers = num_layers
        self.first_layer = StochasticLSTMCell(input_size, hidden_size, dropout_rate)
        self.hidden_layers = nn.ModuleList([StochasticLSTMCell(hidden_size, hidden_size, dropout_rate) for i in range(num_layers-1)])
    
    def regularizer(self):
        total_weight_reg, total_bias_reg, total_dropout_reg = self.first_layer.regularizer()
        for l in self.hidden_layers:
            weight, bias, dropout = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias
            total_dropout_reg += dropout
        return total_weight_reg, total_bias_reg, total_dropout_reg

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
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
        