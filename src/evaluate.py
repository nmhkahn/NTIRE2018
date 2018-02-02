import os
import json
import time
import importlib
import argparse
import scipy.misc as misc
import numpy as np
import skimage.measure as measure
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset


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


def evaluate(net, dataset, chunk, stage, cfg):
    net.eval()
    scale_diff = 2*2**stage
    mean_psnr, mean_runtime = 0, 0
    for step, (lr, hr, name) in enumerate(dataset):
        t1 = time.time()
        h, w = lr.size()[1:]
        h_chunk, w_chunk = int(h/chunk), int(w/chunk)
        h_chop, w_chop   = h_chunk + cfg.shave, w_chunk + cfg.shave

        lr_patch = torch.FloatTensor(chunk**2, 3, h_chop, w_chop)
        for i in range(chunk):
            for j in range(chunk):
                h_from, h_to = i*h_chunk, (i+1)*h_chunk+cfg.shave
                w_from, w_to = j*w_chunk, (j+1)*w_chunk+cfg.shave
                if (i+1) == chunk: h_from, h_to = h-h_chop, h
                if (j+1) == chunk: w_from, w_to = w-w_chop, w
                lr_patch[i+j*chunk].copy_(lr[:, h_from:h_to, w_from:w_to])
        lr_patch = Variable(lr_patch, volatile=True).cuda()
       
        sr = torch.FloatTensor(chunk**2, 3, h_chop*scale_diff, w_chop*scale_diff)
        for i, patch in enumerate(lr_patch):
            out = net(patch.unsqueeze(0), stage, 1)
            sr[i] = out.data
            del out

        h, h_chunk, h_chop = h*scale_diff, h_chunk*scale_diff, h_chop*scale_diff
        w, w_chunk, w_chop = w*scale_diff, w_chunk*scale_diff, w_chop*scale_diff

        result = torch.FloatTensor(3, h, w).cuda()
        for i in range(chunk):
            for j in range(chunk):
                h_from, h_to = 0, h_chunk
                w_from, w_to = 0, w_chunk
                hh_from, hh_to = i*h_chunk, (i+1)*h_chunk
                ww_from, ww_to = j*w_chunk, (j+1)*w_chunk

                if (i+1) == chunk: 
                    h_from, h_to = -h_chunk-(h-(i+1)*h_chunk), None
                    hh_from, hh_to = i*h_chunk, None
                if (j+1) == chunk: 
                    w_from, w_to = -w_chunk-(w-(j+1)*w_chunk), None
                    ww_from, ww_to = j*w_chunk, None

                result[:, hh_from:hh_to, ww_from:ww_to].copy_(sr[i+j*chunk, :, h_from:h_to, w_from:w_to])
        sr = result
        t2 = time.time()
        
        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        # match resolution when stage is less then two
        sr = misc.imresize(sr, 8/scale_diff)

        # crop HR to match SR
        hr = hr[:sr.shape[0], :sr.shape[1]]
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

    mean_runtime, mean_psnr = evaluate(net, dataset, 4, 2, cfg)
    print("Mean runtime: {:.3f}s mean PSNR: {:.2f}".format(mean_runtime, mean_psnr))
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
