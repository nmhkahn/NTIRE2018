import os
import json
import time
import importlib
import argparse
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import TestDataset
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--act", type=str)
    
    parser.add_argument("--dirname", type=str)
    parser.add_argument("--data_from", type=str)
    parser.add_argument("--data_to", type=str)
    
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--sample_dir", type=str)
    parser.add_argument("--shave", type=int, default=20)

    return parser.parse_args()


def save_image(tensor, filename):
    tensor = tensor.cpu()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)


def sample(net, dataset, cfg):
    scale_diff = cfg.scale_diff
    mean_runtime = 0
    for step, (lr, name) in enumerate(dataset):
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
    net = module.Net()
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
                          cfg.data_from,
                          cfg.scale_diff)
    sample(net, dataset, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
