import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import model.ops as ops
from evaluate import evaluate
from dataset import TrainDataset, TestDataset

class Solver():
    def __init__(self, model, cfg):
        self.cfg = cfg

        self.data_path = cfg.data_path
        self.data_names = cfg.data_names
        self.scales = cfg.scales
        
        self.refiner = model().cuda()
        self.loss_fn = nn.L1Loss().cuda()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(self.data_path,
                                       self.data_names,
                                       self.scales,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True)
        
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():
                num_params += param.nelement()
            print("# of params:", num_params)
            
        self.step = 0
        self.stage = 0
        self.max_stage = len(cfg.scales) - 1

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, device_ids=range(cfg.num_gpu))
        
        for stage in range(self.max_stage):
            self.stage = stage
            self._fit_stage()
            self.step = 0
                
    def _fit_stage(self):
        cfg = self.cfg
        stage = self.stage

        while True:
            for data in self.train_loader:
                self.refiner.train()

                data = [Variable(d, requires_grad=False).cuda() for d in data]
                
                alpha = min(1, 2*self.step/cfg.max_steps[stage])
                output = self.refiner(data[0], stage, alpha)
                loss = self.loss_fn(output, data[stage+1])

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                self.step += 1
                if cfg.verbose and (self.step+1) % cfg.print_every == 0:
                    psnr = self.eval(stage)
                    print("[Stage {}: {}K/{}K] {:.3f}".
                          format(stage, int(self.step)+1, int(cfg.max_steps[stage]), psnr))
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)
            
                if (self.step+1) == cfg.max_steps[stage]: return

    def eval(self, stage):
        cfg = self.cfg
        cfg.scale_diff = int(cfg.scales[0]/cfg.scales[stage])
        dataset = TestDataset(cfg.test_dirname,
                              cfg.scale_diff,
                              cfg.test_data_from,
                              cfg.test_data_to)
        _, mean_psnr = evaluate(self.refiner, dataset, 4, stage, cfg)
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
            ckpt_dir, "{}_stage_{}_{}.pth".format(ckpt_name, self.stage, self.step))
        torch.save(self.refiner.state_dict(), save_path)
