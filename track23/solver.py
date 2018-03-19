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

        self.scales = cfg.scales
        self.data_path = cfg.data_path
        self.data_names = cfg.data_names

        self.refiner = model().cuda()
        self.loss_fn = nn.L1Loss().cuda()

        init_param = list(self.refiner.entry.parameters()) + \
                     list(self.refiner.progression[0].parameters()) + \
                     list(self.refiner.to_rgb[0].parameters())
        self.optim = optim.Adam(init_param, cfg.lr)

        self.train_data = TrainDataset(self.data_path,
                                       self.data_names,
                                       self.scales,
                                       size=cfg.patch_size)

        self.writer = SummaryWriter()
        num_params = 0
        for param in self.refiner.parameters():
            num_params += param.nelement()
        print("# of params:", num_params)

        self.step = 1
        self.stage = 0
        self.max_stage = len(cfg.scales) - 1

    def fit(self):
        cfg = self.cfg
        refiner = nn.DataParallel(self.refiner, device_ids=range(cfg.num_gpu))

        for stage in range(self.stage, self.max_stage):
            self.stage = stage
            loader = DataLoader(self.train_data,
                                batch_size=cfg.batch_size[stage],
                                num_workers=2,
                                shuffle=True, drop_last=True)
            self._fit_stage(loader, refiner)

            # reset step for next stage
            self.step = 1
            if (stage+1) == self.max_stage: break

            # decay previous parameters and add new parameters to optim
            for param_group in self.optim.param_groups:
                param_group["lr"] = param_group["lr"] * 0.1

            new_params = list(self.refiner.progression[stage+1].parameters()) + \
                         list(self.refiner.to_rgb[stage+1].parameters())
            self.optim.add_param_group({"params": new_params, "lr": cfg.lr})

    def _fit_stage(self, loader, refiner):
        cfg = self.cfg
        stage = self.stage

        while True:
            for data in loader:
                if self.step >= cfg.max_steps[stage]: return
                refiner.train()

                data = [Variable(d, requires_grad=False).cuda() for d in data]
                output = refiner(data[0], stage)
                loss = self.loss_fn(output, data[stage+1])

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(refiner.parameters(), cfg.clip)
                self.optim.step()

                if self.step % cfg.print_every == 0:
                    psnr = self.eval(stage)
                    global_step = self.step + sum(cfg.max_steps[:stage])

                    self.writer.add_scalar("loss", loss.data[0], global_step)
                    self.writer.add_scalar("psnr", psnr, global_step)
                    print("[Stage{}: {}K/{}K] {:.3f}".format(
                        stage, int(self.step/1000), int(cfg.max_steps[stage]/1000), psnr))

                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

                self.decay_lr(self.step)
                self.step += 1

    def eval(self, stage):
        cfg = self.cfg
        cfg.scale_diff = int(cfg.scales[0]/cfg.scales[stage+1])
        dataset = TestDataset(cfg.test_dirname,
                              cfg.scale_diff,
                              cfg.test_data_from,
                              cfg.test_data_to)
        _, mean_psnr = evaluate(self.refiner, dataset, 2, stage, cfg)
        return mean_psnr

    def load(self, path):
        state_dict = torch.load(path)["state_dict"]
        optim_dict = torch.load(path)["optimizer"]

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            # name = k[7:] # remove "module."
            new_state_dict[name] = v
        self.refiner.load_state_dict(new_state_dict)
        self.optim.load_state_dict(optim_dict)

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_stage_{}_{}.pth".format(ckpt_name, self.stage, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_lr(self, step):
        if step % self.cfg.decay == 0:
            decay_rate = 0.1
        else:
            decay_rate = 1

        for i, param_group in enumerate(self.optim.param_groups):
            old_lr = param_group["lr"]
            param_group["lr"] = old_lr * decay_rate

            if param_group["lr"] != old_lr:
                print("param_group {} is decayed from {:.1e} to {:.1e}".format(
                    i, old_lr, param_group["lr"]))
