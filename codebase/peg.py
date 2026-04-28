import torch, torch.nn as nn

class PEG(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, k, padding=k//2, groups=dim)
        self.gamma   = nn.Parameter(torch.zeros(1))

    def forward(self, tokens, H, W):
        b, n, c = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, c, H, W)
        pos = self.dw_conv(x)
        x   = x + self.gamma * pos
        return x.flatten(2).transpose(1, 2)
