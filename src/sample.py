import os
import json
import time
import importlib
import argparse
import numpy as np
import scipy.misc as misc
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    
    parser.add_argument("--dirname", type=str)
    parser.add_argument("--data_from", type=str)
    parser.add_argument("--data_to", type=str)
    
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--shave", type=int, default=20)
    
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--stage", type=int)

    return parser.parse_args()


def save_image(arr, filename):
    im = Image.fromarray(arr)
    im.save(filename)


def sample(net, dataset, chunk, stage, cfg):
    from torch.nn import functional as F
    
    scale_diff = int(cfg.scales[0]/cfg.scales[stage+1])

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
            sr[i] = net(patch.unsqueeze(0), stage).data
           
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

                result[:, hh_from:hh_to, ww_from:ww_to].copy_(
                    sr[i+j*chunk, :, h_from:h_to, w_from:w_to])
        t2 = time.time()

        sr = result.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        
        # match resolution when stage is less then two
        if int(cfg.scale_from/scale_diff) > 1:
            sr = misc.imresize(sr, cfg.scale_from/scale_diff)
        
        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        name_from = cfg.data_from.split("_")[-1]
        name_to   = cfg.data_to.split("_")[-1]
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name, 
                              "{}-{}".format(name_from, name_to))
        
        if not os.path.exists(sr_dir):
            os.makedirs(sr_dir)
            
        sr_im_path = os.path.join(sr_dir, "{}".format(name))
        save_image(sr, sr_im_path)
        mean_runtime += (t2-t1) / len(dataset)
        print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
            .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))

    print("Mean runtime: {:.3f}s".format(mean_runtime))


def main(cfg):
    cfg.scale_from = [int(s) for s in cfg.data_from if s.isdigit()][-1]
    if "HR" in cfg.data_to:
        cfg.scale_to = 1
    else:
        cfg.scale_to = [int(s) for s in cfg.data_to if s.isdigit()][-1]
    cfg.scale_diff = int(cfg.scale_from/cfg.scale_to)
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    module = importlib.import_module("model.{}".format(cfg.model))

    do_up_first = True if cfg.scale_from == 8 else False
    net = module.Net(do_up_first)
    net.eval()
    
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
    sample(net, dataset, cfg.chunk, cfg.stage, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
