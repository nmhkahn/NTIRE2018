import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(lr, hr, size, scale):
    h, w = lr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    hsize = size*scale
    hx, hy = x*scale, y*scale

    crop_lr = lr[y:y+size, x:x+size].copy()
    crop_hr = hr[hy:hy+hsize, hx:hx+hsize].copy()

    return crop_lr, crop_hr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, 
                 path_from, path_to, 
                 data_from, data_to, 
                 scale_diff, size):
        super(TrainDataset, self).__init__()

        self.size = size
        self.scale_diff = scale_diff
        f_from = h5py.File(path_from, "r")
        f_to   = h5py.File(path_to, "r")
        
        self.im_from = [v[:] for v in f_from[data_from].values()]
        self.im_to   = [v[:] for v in f_to[data_to].values()]
        
        f_from.close(); f_to.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        im_from, im_to = self.im_from[index], self.im_to[index]
        im_from, im_to = random_crop(im_from, im_to, self.size, self.scale_diff)
        im_from, im_to = random_flip_and_rotate(im_from, im_to)
        
        return self.transform(im_from), self.transform(im_to)

    def __len__(self):
        return len(self.im_to)
        

class TestDataset(data.Dataset):
    def __init__(self, 
                 dirname,
                 scale_diff,
                 data_from, data_to=None):
        super(TestDataset, self).__init__()

        self.name  = dirname.split("/")[-1]
        self.scale_diff = scale_diff

        self.im_from = glob.glob(os.path.join("{}/{}/*.png".format(dirname, data_from)))
        self.im_from.sort()
        
        if data_to:
            self.data_to = True
            self.im_to   = glob.glob(os.path.join("{}/{}/*.png".format(dirname, data_to)))
            self.im_to.sort()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        im_from = Image.open(self.im_from[index])

        im_from = im_from.convert("RGB")
        filename = self.im_from[index].split("/")[-1]

        if self.data_to:
            im_to   = Image.open(self.im_to[index])
            return self.transform(im_from), self.transform(im_to), filename
        return self.transform(im_from), filename

    def __len__(self):
        return len(self.im_from)
