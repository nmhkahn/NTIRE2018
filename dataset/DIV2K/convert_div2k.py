import os
import glob
import h5py
import argparse
import scipy.misc as misc
import numpy as np

def gt2h5():
    f = h5py.File("gt.h5", "w")
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    for subdir in ["DIV2K_train_HR", "DIV2K_train_LR_x2", "DIV2K_train_LR_x4"]:
        im_paths = glob.glob(os.path.join("train/{}/*.png".format(subdir)))
        im_paths.sort()
        grp = f.create_group(subdir.split("_")[-1])

        for i, path in enumerate(im_paths):
            im = misc.imread(path)
            print(path)
            grp.create_dataset(str(i), data=im)


def target2h5(target):
    f = h5py.File("{}.h5".format(target), "w")
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    for subdir in ["DIV2K_train_LR_{}".format(target)]:
        im_paths = glob.glob(os.path.join("train/{}/*.png".format(subdir)))
        im_paths.sort()
        grp = f.create_group(subdir.split("_")[-1])

        for i, path in enumerate(im_paths):
            im = misc.imread(path)
            print(path)
            grp.create_dataset(str(i), data=im)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str)
    args = parser.parse_args()

    if args.target == "gt":
        gt2h5()
    else:
        target2h5(args.target)
    
if __name__ == "__main__":
    main()
