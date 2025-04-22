import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    def forward(self, x):
        y = torch.matmul(x, self.weight.T)
        if self.bias is not None:
            y = y + self.bias
        return y