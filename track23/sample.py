import os
import json
import time
import copy
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


def generate_single(net, lr, chunk, stage, cfg):
    scale_diff = cfg.scale_diff
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
    
    h, h_chunk, h_chop = h*scale_diff, h_chunk*scale_diff, h_chop*scale_diff
    w, w_chunk, w_chop = w*scale_diff, w_chunk*scale_diff, w_chop*scale_diff

    sr = np.empty((chunk**2, h_chop, w_chop, 3), dtype=np.uint8)
    for i, patch in enumerate(lr_patch):
        out = net(patch.unsqueeze(0), stage).data[0]
        out = out.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        sr[i] = out

    result = np.empty((h, w, 3), dtype=np.uint8)
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

            result[hh_from:hh_to, ww_from:ww_to, :] = \
                copy.deepcopy(sr[i+j*chunk, h_from:h_to, w_from:w_to, :])
    return result


def recover_origin(image, index):
    angle = [0, 3, 2, 1]
    image = np.rot90(image, angle[index%4]).copy()
    if index > 3:
        image = np.fliplr(image).copy()

    return image


def sample(net, dataset, chunk, stage, cfg):
    net.eval()
    scale_diff = cfg.scale_diff
    mean_psnr, mean_runtime = 0, 0
    for step, (lr_ensemble, name) in enumerate(dataset):

        h, w = lr_ensemble[0].size()[1:]
        sr_ensemble = np.zeros((len(lr_ensemble), h*scale_diff, w*scale_diff, 3))

        t1 = time.time()
        for i, lr in enumerate(lr_ensemble):
            tmp_image = generate_single(net, lr, chunk, stage, cfg)
            sr_ensemble[i] = recover_origin(tmp_image, i)

        sr = np.mean(sr_ensemble, axis=0)
        sr = np.clip(sr, 0, 255)
        sr = sr.astype(np.uint8)

        t2 = time.time()

        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        name_from = cfg.data_from.split("_")[-1]
        sr_dir = os.path.join(cfg.sample_dir, model_name)
        
        if not os.path.exists(sr_dir):
            os.makedirs(sr_dir)
            
        sr_im_path = os.path.join(sr_dir, "{}".format(name))
        save_image(sr, sr_im_path)
        mean_runtime += (t2-t1) / len(dataset)
        print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
            .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[0], sr.shape[1], t2-t1))

    print("Mean runtime: {:.3f}s".format(mean_runtime))


def main(cfg):
    cfg.scales = list()
    cfg.scales.append([int(s) for s in cfg.data_from if s.isdigit()][-1])
    if "HR" in cfg.data_to:
        cfg.scales.append(1)
    else:
        cfg.scales.append([int(s) for s in cfg.data_to if s.isdigit()][-1])
    cfg.scale_diff = 4

    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net()
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
                          cfg.data_from, self_ensemble=True)
    sample(net, dataset, cfg.chunk, cfg.stage, cfg)
 

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
