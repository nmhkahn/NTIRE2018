import os
import json
import time
import copy
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

def generate_intra_ensemble(net, patch, stage, cfg):
    x = patch.unsqueeze(0)
    out = x.clone()
    
    # generate first stage using input patch
    in_ = net.forward_from_to(x, out, 0, 1)
    in_ = in_.data[0].cpu().numpy()

    h, w = patch.size(1)*cfg.scale_diff, patch.size(2)*cfg.scale_diff
    patches = np.empty((4, h, w, 3), dtype=np.uint8)
    for i, angle_i in enumerate([0, 2]):
        # intra-ensemble for first -> second stage
        out = np.rot90(in_, angle_i, axes=(1, 2)).copy()
        
        out = Variable(torch.from_numpy(out), volatile=True)
        out = out.unsqueeze(0).cuda()

        out_ = net.forward_from_to(x, out, 1, 2)
        out_ = out_.data[0].cpu().numpy()

        for j, angle_j in enumerate([0, 2]):
            out = np.rot90(out_, angle_j, axes=(1, 2)).copy()
    
            out = Variable(torch.from_numpy(out), volatile=True)
            out = out.unsqueeze(0).cuda()
            
            # transform x
            index = (angle_i+angle_j) % 4
            x_np = x.data[0].cpu().numpy()
            x_transformed = np.rot90(x_np, index, axes=(1, 2)).copy()
            x_transformed = Variable(torch.from_numpy(x_transformed), volatile=True)
            x_transformed = x_transformed.unsqueeze(0).cuda()

            out = net.forward_from_to(x_transformed, out, 2, 3)
            out_np = out.data[0].cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            misc.imsave("result/{}_{}_0.png".format(i, j), out_np)
            # recover to origin
            out_np = recover_origin(out_np, index)
            
            patches[i*2+j] = out_np

    out = np.mean(patches, axis=0)
    return out
    

def generate_single(net, lr, hr, chunk, stage, cfg):
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
        #out = generate_intra_ensemble(net, patch, stage, cfg)
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


def evaluate(net, dataset, chunk, stage, cfg):
    net.eval()
    scale_diff = cfg.scale_diff
    mean_psnr, mean_runtime = 0, 0
    for step, (lr_ensemble, hr, name) in enumerate(dataset):
        hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

        h, w = lr_ensemble[0].size()[1:]
        sr_ensemble = np.zeros((len(lr_ensemble), h*scale_diff, w*scale_diff, 3))

        t1 = time.time()
        for i, lr in enumerate(lr_ensemble):
            tmp_image = generate_single(net, lr, hr, chunk, stage, cfg)
            sr_ensemble[i] = recover_origin(tmp_image, i)

        sr = np.mean(sr_ensemble, axis=0)
        sr = np.clip(sr, 0, 255)

        t2 = time.time()

        # match resolution when stage is less then two
        if int(cfg.scales[0]/scale_diff) > 1:
            sr = misc.imresize(sr, cfg.scales[0]/scale_diff)
        
        # crop HR to match SR
        hr = hr[:sr.shape[0], :sr.shape[1]]
        bnd = int(cfg.scales[0]/cfg.scales[-1]) + 6
        hr = hr[bnd:-bnd, bnd:-bnd]
        sr = sr[bnd:-bnd, bnd:-bnd]

        mean_psnr += psnr(hr, sr) / len(dataset)
        mean_runtime += (t2-t1) / len(dataset)

        print(step, psnr(hr, sr), t2-t1)

    return mean_runtime, mean_psnr


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


def main(cfg):
    cfg.scales = list()
    cfg.scales.append([int(s) for s in cfg.data_from if s.isdigit()][-1])
    if "HR" in cfg.data_to:
        cfg.scales.append(1)
    else:
        cfg.scales.append([int(s) for s in cfg.data_to if s.isdigit()][-1])
    cfg.scale_diff = 8

    module = importlib.import_module("model.{}".format(cfg.model))
    net = module.Net()
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict["state_dict"])
    net.cuda()

    dataset = TestDataset(cfg.dirname,
                          cfg.scale_diff,
                          cfg.data_from,
                          cfg.data_to, self_ensemble=True)

    mean_runtime, mean_psnr = evaluate(net, dataset, cfg.chunk, cfg.stage, cfg)
    print("Mean runtime: {:.3f}s mean PSNR: {:.3f}".format(mean_runtime, mean_psnr))


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
