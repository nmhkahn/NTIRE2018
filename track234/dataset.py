import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(images, scales, size):
    """
    Args
        images: Single image with different scale. zero-index element must be LR
                and last-index as HR.
        scales: Scale description of images args.
        size:   Size of cropped LR image.
    """

    h, w = images[0].shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)
    
    cimages = list()
    for i, image in enumerate(images):
        scale_diff = int(scales[0]/scales[i])
        hsize = size*scale_diff
        hx, hy = x*scale_diff, y*scale_diff

        cimages.append(image[hy:hy+hsize, hx:hx+hsize].copy())

    return cimages


def random_flip_and_rotate(images):
    if random.random() < 0.5:
        for i, image in enumerate(images):
            images[i] = np.flipud(image)

    if random.random() < 0.5:
        for i, image in enumerate(images):
            images[i] = np.fliplr(image)

    angle = random.choice([0, 1, 2, 3])
    for i, image in enumerate(images):
        images[i] = np.rot90(image, angle)

    # have to copy before be called by transform function
    new_image = list()
    for image in images:
        new_image.append(image.copy())
    return new_image


class TrainDataset(data.Dataset):
    def __init__(self, 
                 path, 
                 data_names, 
                 scales, size):
        super(TrainDataset, self).__init__()

        self.size = size
        self.scales = scales
        self.data = list()
        
        f = h5py.File(path, "r")
        for name in data_names:
            self.data.append([v[:] for v in f[name].values()])
        f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        images = [d[index] for d in self.data]
        images = random_crop(images, self.scales, self.size)
        images = random_flip_and_rotate(images)
        
        return [self.transform(image) for image in images]

    def __len__(self):
        return len(self.data[0])
        

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
