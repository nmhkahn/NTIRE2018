import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import model.ops as ops
from evaluate import evaluate
from dataset import TrainDataset, TestDataset

class Solver():
    def __init__(self, model, cfg):
        self.cfg = cfg

        self.data_path = cfg.data_path
        self.data_names = cfg.data_names
        self.scales = cfg.scales

        self.refiner = model(True, cfg.scales[0]).cuda()
        self.loss_fn = nn.L1Loss().cuda()
        self.optim = optim.Adam(self.refiner.parameters(), cfg.lr)
        
        self.train_data = TrainDataset(self.data_path,
                                       self.data_names,
                                       self.scales,
                                       size=cfg.patch_size)
        

        self.writer = SummaryWriter()
        num_params = 0
        for param in self.refiner.parameters():
            num_params += param.nelement()
        print("# of params:", num_params)
            
        self.step = 0

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, device_ids=range(cfg.num_gpu))
        
        loader = DataLoader(self.train_data,
                            batch_size=cfg.batch_size[0],
                            num_workers=1,
                            shuffle=True, drop_last=True)
            
        while True:
            for data in loader:
                self.refiner.train()

                data = [Variable(d, requires_grad=False).cuda() for d in data]
                
                output = self.refiner(data[0])
                loss = self.loss_fn(output, data[1])

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(), cfg.clip)
                self.optim.step()

                self.step += 1
                if (self.step+1) % cfg.print_every == 0:
                    psnr = self.eval()
                    global_step = self.step
                    
                    self.writer.add_scalar("loss", loss.data[0], global_step)
                    self.writer.add_scalar("psnr", psnr, global_step)
                    print("[{}K/{}K] {:.3f}".
                          format(int((self.step+1)/1000), int(cfg.max_steps[0]/1000), psnr))

                    self.save(cfg.ckpt_dir, cfg.ckpt_name)
            
                if (self.step+1) >= cfg.max_steps[0]: return

    def eval(self):
        cfg = self.cfg
        cfg.scale_diff = int(cfg.scales[0]/cfg.scales[1])
        dataset = TestDataset(cfg.test_dirname,
                              cfg.scale_diff,
                              cfg.test_data_from,
                              cfg.test_data_to)
        _, mean_psnr = evaluate(self.refiner, dataset, 2, 0, cfg)
        return mean_psnr
    
    def load(self, path):
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            # name = k[7:] # remove "module."
            new_state_dict[name] = v
        self.refiner.load_state_dict(new_state_dict)
        
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step+1))

        state = {
            "state_dict": self.refiner.state_dict(),
            "optimizer": self.optim.state_dict()
        }
        torch.save(state, save_path)
