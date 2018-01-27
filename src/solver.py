import os
import random
import time
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset

class Solver():
    def __init__(self, model, cfg):
        self.step = 0
        self.cfg = cfg

        self.path_from = cfg.path_from
        self.path_to   = cfg.path_to
        self.data_from = cfg.data_from
        self.data_to   = cfg.data_to

        self.scale_from = [int(s) for s in self.data_from if s.isdigit()][0]
        self.scale_to   = [int(s) for s in self.data_to if s.isdigit()][0]
        self.scale_diff = int(self.scale_from/self.scale_to)
        
        self.refiner = model(scale_from=self.scale_from, 
                             scale_to=self.scale_to,
                             act=cfg.act)
        self.loss_fn = nn.L1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(self.path_from, self.path_to,
                                       self.data_from, self.data_to,
                                       self.scale_diff,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        self.refiner = self.refiner.cuda()
        self.loss_fn = self.loss_fn.cuda()
        
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, device_ids=range(cfg.num_gpu))
        
        t1 = time.time()
        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:
                self.refiner.train()
                lr, hr = inputs[0], inputs[1]

                hr = Variable(hr, requires_grad=False).cuda()
                lr = Variable(lr, requires_grad=False).cuda()
                sr = refiner(lr)
                loss = self.loss_fn(sr, hr)
                
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate

                self.step += 1
                if cfg.verbose and self.step % cfg.print_every == 0:
                    t2 = time.time()
                    remain_step = cfg.max_steps - self.step
                    eta = (t2-t1)*remain_step/cfg.print_every/3600
                    print("[{}K/{}K] {:.5f} ETA: {:.1f} hours".
                          format(int(self.step/1000), int(cfg.max_steps/1000), loss.data[0], eta))
                            
                    t1 = time.time()
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break

    def evaluate(self, test_data_dir, num_step=0):
        cfg = self.cfg
        mean_psnr = 0
        self.refiner.eval()
        
        test_data   = TestDataset(test_data_dir, 
                                  scale_from=self.scale_from,
                                  scale_to=self.scale_to)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = Variable(lr_patch, volatile=True).cuda()
            
            # run refine process in here!
            sr = self.refiner(lr_patch).data
            
            h, h_half, h_chop = h*self.scale_diff, h_half*self.scale_diff, h_chop*self.scale_diff
            w, w_half, w_chop = w*self.scale_diff, w_half*self.scale_diff, w_chop*self.scale_diff
            
            # merge splited patch images
            result = torch.FloatTensor(3, h, w).cuda()
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            """
            sr_dir = os.path.join(cfg.sample_dir,
                                  cfg.ckpt_name,
                                  str(num_step),
                                  test_data_dir.split("/")[-1],
                                  "X{}_{}".format(self.scale_from, self.scale_to),
                                  "SR")
            if not os.path.exists(sr_dir):
                os.makedirs(sr_dir)

            sr_im_path = os.path.join(sr_dir, "{}".format(name))
            misc.imsave(sr_im_path, sr)
            """
            
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR  
            bnd = self.scale_diff + 6
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            mean_psnr += psnr(im1, im2) / len(test_data)

        return mean_psnr

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        splited = path.split(".")[0].split("_")[-1]
        try:
            self.step = int(path.split(".")[0].split("_")[-1])
        except ValueError:
            self.step = 0
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr


def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    psnr = measure.compare_psnr(im1, im2, data_range=1)
    return psnr


def ssim(im1, im2):
    ssim = measure.compare_ssim(im1, im2, 
                                K1=0.01, K2=0.03,
                                gaussian_weights=True, 
                                sigma=1.5,
                                use_sample_covariance=False,
                                multichannel=True)

    return ssim
