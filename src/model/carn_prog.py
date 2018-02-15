import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import model.ops as ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(64, 64)
        self.b2 = ops.ResidualBlock(64, 64)
        self.b3 = ops.ResidualBlock(64, 64)
        self.b4 = ops.ResidualBlock(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1)
        self.c2 = ops.BasicBlock(64*3, 64, 1)
        self.c3 = ops.BasicBlock(64*4, 64, 1)
        self.c4 = ops.BasicBlock(64*5, 64, 1)

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
    def __init__(self, do_upsample=True):
        super(CARN, self).__init__()
        
        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.b4 = Block(64, 64)
        self.b5 = Block(64, 64)
        self.b6 = Block(64, 64)
        self.b7 = Block(64, 64)
        self.b8 = Block(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1)
        self.c2 = ops.BasicBlock(64*3, 64, 1)
        self.c3 = ops.BasicBlock(64*4, 64, 1)
        self.c4 = ops.BasicBlock(64*5, 64, 1)
        self.c5 = ops.BasicBlock(64*6, 64, 1)
        self.c6 = ops.BasicBlock(64*7, 64, 1)
        self.c7 = ops.BasicBlock(64*8, 64, 1)
        self.c8 = ops.BasicBlock(64*9, 64, 1)
        
        if do_upsample:
            self.up = ops.UpsampleBlock(64, scale=2)
        self.do_upsample = do_upsample
        
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
        
        b5 = self.b5(o4)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)
        
        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        
        b7 = self.b7(o6)
        c7 = torch.cat([c6, b7], dim=1)
        o7 = self.c7(c7)
        
        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)
        
        out = o8
        if self.do_upsample:
            out = self.up(out)

        return out


class Net(nn.Module):
    def __init__(self, do_up_first=True):
        super(Net, self).__init__()
        self.entry = ops.BasicBlock(3, 64, 3, act=nn.ReLU())
        self.progression = nn.ModuleList([
            CARN(do_up_first),
            CARN(),
            CARN()
        ])
        
        self.to_rgb = nn.ModuleList([
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1),
        ])

        self.do_up_first = do_up_first

    def forward(self, x, stage):
        out = self.entry(x)
        for i, (carn, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            out = carn(out)
            if i == stage:
                out = to_rgb(out)
                if self.do_up_first:
                    out += F.upsample(x, scale_factor=2*2**stage)
                elif stage > 0:
                    out += F.upsample(x, scale_factor=2*2**(stage-1))
                else:
                    out += x
                break
    
        return out
