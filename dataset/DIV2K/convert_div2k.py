import os
import glob
import h5py
import argparse
import scipy.misc as misc
import numpy as np

def main():
    f = h5py.File("div2k_train.h5", "w")
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    for subdir in ["HR", "LR_x2", "LR_x4", "LR_x8"]:
        im_paths = glob.glob(os.path.join("train/DIV2K_train_{}/*.png".format(subdir)))
        im_paths.sort()
        grp = f.create_group(subdir.split("_")[-1])

        for i, path in enumerate(im_paths):
            im = misc.imread(path)
            print(path)
            grp.create_dataset(str(i), data=im)


if __name__ == "__main__":
    main()
