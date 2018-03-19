import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def init_weights(modules):
    pass


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3,
                 act=nn.ReLU(inplace=True)):
        super(BasicBlock, self).__init__()

        pad = 1 if ksize == 3 else 0
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, 1, pad),
            act
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 act=nn.ReLU(inplace=True)):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            act,
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            act,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        init_weights(self.modules)

    def forward(self, x):
        residual = x
        out = self.body(x)
        out += residual
        return out


class UpsampleBlock(nn.Module):
    def __init__(self,
                 n_channels,
                 scale,
                 act=nn.ReLU(inplace=True)):
        super(UpsampleBlock, self).__init__()

        modules = []
        # only support x2, x4, x8
        for _ in range(int(math.log(scale, 2))):
            modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1), act]
            modules += [nn.PixelShuffle(2)]
        self.body = nn.Sequential(*modules)

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out
