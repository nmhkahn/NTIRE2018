import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import model.ops as ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(192, 192)
        self.b2 = ops.ResidualBlock(192, 192)
        self.b3 = ops.ResidualBlock(192, 192)
        self.b4 = ops.ResidualBlock(192, 192)
        self.c1 = ops.BasicBlock(192*2, 192, 1)
        self.c2 = ops.BasicBlock(192*3, 192, 1)
        self.c3 = ops.BasicBlock(192*4, 192, 1)
        self.c4 = ops.BasicBlock(192*5, 192, 1)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        
        b4 = self.b4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.c4(c4)
       
        return o4
        

class CARN(nn.Module):
    def __init__(self, do_up=True):
        super(CARN, self).__init__()
        
        self.b1 = Block(192, 192)
        self.b2 = Block(192, 192)
        self.b3 = Block(192, 192)
        self.b4 = Block(192, 192)
        self.c1 = ops.BasicBlock(192*2, 192, 1)
        self.c2 = ops.BasicBlock(192*3, 192, 1)
        self.c3 = ops.BasicBlock(192*4, 192, 1)
        self.c4 = ops.BasicBlock(192*5, 192, 1)
        
        if do_up:
            self.up = ops.UpsampleBlock(192, scale=4)

        self.do_up = do_up
        
    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        b4 = self.b4(o3)
        c4 = torch.cat([c3, b4], dim=1)
        o4 = self.c4(c4)
        
        out = o4
        if self.do_up:
            out = self.up(out)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.entry = ops.BasicBlock(3, 192, 3, act=nn.ReLU())
        self.progression = nn.ModuleList([
            CARN(False),
            CARN(),
        ])
        
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(192, 3, 3, 1, 1),
            nn.Conv2d(192, 3, 3, 1, 1),
        ])

    def forward(self, x, stage):
        out = self.entry(x)
        for i, (carn, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            out = carn(out)
            if i == stage:
                out = to_rgb(out)
                if stage == 0:
                    out += x
                else:
                    out += F.upsample(x, scale_factor=2*2**stage)
                break
    
        return out
