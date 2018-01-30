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
        self.step = 0
        self.cfg = cfg

        self.data_path = cfg.data_path
        self.data_names = cfg.data_names
        self.scales = cfg.scales
        
        self.refiner = model().cuda()
        # self.loss_fn = nn.L1Loss().cuda()
        self.loss_fn = ops.CharbonnierLoss().cuda()

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

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, device_ids=range(cfg.num_gpu))
        
        t1 = time.time()
        learning_rate = cfg.lr
        while True:
            for data in self.train_loader:
                self.refiner.train()

                data = [Variable(d, requires_grad=False).cuda() for d in data]
                outputs = refiner(data[-1])
                
                loss = self.loss_fn(outputs[0], data[0]) + \
                       self.loss_fn(outputs[1], data[1]) + \
                       self.loss_fn(outputs[2], data[2])
                
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate

                self.step += 1
                if cfg.verbose and self.step % cfg.print_every == 0:
                    psnr = self.eval()
                    t2 = time.time()
                    remain_step = cfg.max_steps - self.step
                    eta = (t2-t1)*remain_step/cfg.print_every/3600
                    print("[{}K/{}K] {:.3f} ETA: {:.1f} hours".
                          format(int(self.step/1000), int(cfg.max_steps/1000), psnr, eta))
                            
                    t1 = time.time()
                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break


    def eval(self):
        cfg = self.cfg
        cfg.scale_diff = int(cfg.scales[-1]/cfg.scales[0])
        dataset = TestDataset(cfg.test_dirname,
                              cfg.scale_diff,
                              cfg.test_data_from,
                              cfg.test_data_to)
        _, mean_psnr = evaluate(self.refiner, dataset, cfg)
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
