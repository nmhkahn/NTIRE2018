import os
import glob
import h5py
import scipy.misc as misc
import numpy as np

dataset_type = "train"

f = h5py.File("{}.h5".format(dataset_type), "w")
dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["HR", "X2", "X4", "X8"]:
    im_paths = glob.glob(os.path.join("{}/{}/*.png".format(dataset_type, subdir)))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = misc.imread(path)
        print(path)
        grp.create_dataset(str(i), data=im)
