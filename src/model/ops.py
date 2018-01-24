import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def init_weights(modules):
    pass


class SUnit(nn.Module):
    def __init__(self,
                 in_channels, out_channels):
        super(SUit, self).__init__()

        self.ru = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Sigmoid(inpace=True)
        )

    def forward(self, x):
        return x * self.ru(x)


def select_act(act, in_channels, out_channels):
    if act == "sunit":
        return SUit(in_channels, out_channels)
    elif act == "relu":
        return nn.ReLU()
    else:
        raise NotImplementedError("We only support SUnit and ReLU")
   

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3,
                 act="relu"):
        super(BasicBlock, self).__init__()

        pad = 1 if ksize == 3 else 0
        act = select_act(act, in_channels, out_channels)
        
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
                 act="relu"):
        super(ResidualBlock, self).__init__()

        act = select_act(act, in_channels, out_channels)
        self.body = nn.Sequential(
            act,
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            act,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = out + x
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, 
                 scale, 
                 act="relu"):
        super(UpsampleBlock, self).__init__()

        modules = []
        act = select_act(act, n_channels, n_channels)
        # only support x2, x4, x8
        for _ in range(int(math.log(scale, 2))):
            modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1), act]
            modules += [nn.PixelShuffle(2)]
        self.body = nn.Sequential(*modules)

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out
