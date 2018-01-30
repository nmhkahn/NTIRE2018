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
        

class CARN(nn.Module):
    def __init__(self, **kwargs):
        super(CARN, self).__init__()
        
        act = nn.ReLU()
        self.b1 = Block(64, 64, act=act)
        self.b2 = Block(64, 64, act=act)
        self.b3 = Block(64, 64, act=act)
        self.b4 = Block(64, 64, act=act)
        self.c1 = ops.BasicBlock(64*2, 64, 1, act=act)
        self.c2 = ops.BasicBlock(64*3, 64, 1, act=act)
        self.c3 = ops.BasicBlock(64*4, 64, 1, act=act)
        self.c4 = ops.BasicBlock(64*5, 64, 1, act=act)
        
        self.upsample = ops.UpsampleBlock(64, scale=2, act=act)
        self.exit = ops.BasicBlock(64, 64, 3, act=act)
                
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

        out = self.upsample(o4)
        out = self.exit(out)
        return out


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        # x8 -> x4
        self.n1 = CARN(**kwargs)
        self.u1 = ops.UpsampleBlock(3, scale=2)
        self.r1 = nn.Conv2d(64, 3, 3, 1, 1)
        # x4 -> x2
        self.n2 = CARN(**kwargs)
        self.u2 = ops.UpsampleBlock(3, scale=2)
        self.r2 = nn.Conv2d(64, 3, 3, 1, 1)
        # x2 -> x1
        self.n3 = CARN(**kwargs)
        self.u3 = ops.UpsampleBlock(3, scale=2)
        self.r3 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        entry = self.entry(x)
    
        n1 = self.n1(entry)
        u1 = self.u1(x)
        r1 = self.r1(n1)
        x4 = u1 + r1

        n2 = self.n2(n1)
        u2 = self.u2(x4)
        r2 = self.r2(n2)
        x2 = u2 + r2

        n3 = self.n3(n2)
        u3 = self.u3(x2)
        r3 = self.r3(n3)
        x1 = u3 + r3

        return x1, x2, x4
        
