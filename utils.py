class DropoutMSELoss():
    def __init__(self, length_scale, precision, dropout):
        super(DropoutMSELoss, self).__init__()
        self.loss = nn.MSELoss()
        self.length_scale = length_scale
        self.precision = precision
        self.dropout = dropout
    
    def __call__(self, outputs, labels, named_params, N):
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

    
class DropoutMSELoss():
    def __init__(self, length_scale, precision, dropout):
        super(DropoutMSELoss, self).__init__()
        self.loss = nn.MSELoss()
        self.length_scale = length_scale
        self.precision = precision
        self.dropout = dropout
    
    def __call__(self, outputs, labels, named_params, N):
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