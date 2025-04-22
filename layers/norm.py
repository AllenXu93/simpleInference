import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class ManualLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = torch.mean((x-mean).square(), dim=-1, keepdim=True)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias
