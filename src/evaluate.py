import os
import json
import time
import importlib
import argparse
import numpy as np
import skimage.measure as measure
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--act", type=str)
    
    parser.add_argument("--dirname", type=str)
    parser.add_argument("--data_from", type=str)
    parser.add_argument("--data_to", type=str)
    
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def evaluate(net, dataset, cfg):
    mean_runtime = 0
    mean_psnr = 0
    scale_diff = cfg.scale_diff
    for step, (lr, hr, name) in enumerate(dataset):
        t1 = time.time()
        h, w = lr.size()[1:]
        h_half, w_half = int(h/2), int(w/2)
        h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

        lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
        lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
        lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
        lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
        lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
        lr_patch = Variable(lr_patch, volatile=True).cuda()
       
        sr = torch.FloatTensor(4, 3, h_chop*scale_diff, w_chop*scale_diff)
        for i, patch in enumerate(lr_patch):
            sr[i] = net(patch.unsqueeze(0))[0].data
            
        h, h_half, h_chop = h*scale_diff, h_half*scale_diff, h_chop*scale_diff
        w, w_half, w_chop = w*scale_diff, w_half*scale_diff, w_chop*scale_diff

        result = torch.FloatTensor(3, h, w).cuda()
        result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
        result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
        result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
        result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
        sr = result
        t2 = time.time()
        
        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        # crop HR to match SR
        hr = hr[:h, :w]
        bnd = cfg.scale_diff + 6
        im1 = hr[bnd:-bnd, bnd:-bnd]
        im2 = sr[bnd:-bnd, bnd:-bnd]

        mean_psnr += psnr(im1, im2) / len(dataset)
        mean_runtime += (t2-t1) / len(dataset)

    return mean_runtime, mean_psnr


def main(cfg):
    cfg.scale_from = [int(s) for s in cfg.data_from if s.isdigit()][-1]
    if "HR" in cfg.data_to:
        cfg.scale_to = 1
    else:
        cfg.scale_to   = [int(s) for s in cfg.data_to if s.isdigit()][-1]
    cfg.scale_diff = int(cfg.scale_from/cfg.scale_to)
    
    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net(scale_from=cfg.scale_from,
                     scale_to=cfg.scale_to,
                     act=cfg.act)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    
    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.cuda()

    dataset = TestDataset(cfg.dirname,
                          cfg.scale_diff,
                          cfg.data_from,
                          cfg.data_to)

    mean_runtime, mean_psnr = evaluate(net, dataset, cfg)
    print("Mean runtime: {:.3f}s mean PSNR: {:.2f}".format(mean_runtime, mean_psnr))
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
