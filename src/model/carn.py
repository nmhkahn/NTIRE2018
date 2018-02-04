import math
import torch
import torch.nn as nn
import model.ops as ops

class Block(nn.Module):
    def __init__(self, 
                 in_channels, out_channels,
                 act):
        super(Block, self).__init__()

        self.b1 = ops.ResidualBlock(64, 64, act=act)
        self.b2 = ops.ResidualBlock(64, 64, act=act)
        self.b3 = ops.ResidualBlock(64, 64, act=act)
        self.b4 = ops.ResidualBlock(64, 64, act=act)
        self.c1 = ops.BasicBlock(64*2, 64, 1, act=act)
        self.c2 = ops.BasicBlock(64*3, 64, 1, act=act)
        self.c3 = ops.BasicBlock(64*4, 64, 1, act=act)
        self.c4 = ops.BasicBlock(64*5, 64, 1, act=act)

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
        

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        act = nn.ReLU()
        self.b1 = Block(64, 64, act=act)
        self.b2 = Block(64, 64, act=act)
        self.b3 = Block(64, 64, act=act)
        self.b4 = Block(64, 64, act=act)
        self.c1 = ops.BasicBlock(64*2, 64, 1, act=act)
        self.c2 = ops.BasicBlock(64*3, 64, 1, act=act)
        self.c3 = ops.BasicBlock(64*4, 64, 1, act=act)
        self.c4 = ops.BasicBlock(64*5, 64, 1, act=act)
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)
        self.upsample = ops.UpsampleBlock(64, scale=8, act=act)
        self.exit = ops.BasicBlock(64, 3, 3, act=act)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = 9 * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                
    def forward(self, x):
        x = self.entry(x)
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

        out = self.upsample(o4)
        out = self.exit(out)
        return out