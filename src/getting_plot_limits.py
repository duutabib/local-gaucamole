import pdb
import pickle
from glob import glob

import numpy as np
from scipy.stats import skew

print("franctically getting all file pickle paths...")
data_root = "/Users/duuta/ppp/data/data00/pickled/"
data_files = glob(f"{data_root}*.pickle")
fnames = [fname.split("/")[-1].strip(".pickle") for fname in data_files]


def read_data(data):
    y0 = data["Fsp"]
    ymin = np.min(y0)
    ythres = ymin + 1234e-6
    y = np.clip(y0, ythres, np.inf)
    logy = np.log10(y)

    return y, logy


y_maxlims = []
y_minlims = []
logy_maxlims = []
logy_minlims = []


# init vars
y  = None
ymu = None 
ysk = None
ystd = None
logymu = None
logystd = None
logysk = None


for fname, fpath in zip(fnames, data_files):
    ofile = open(fpath, "rb")
    data = pickle.load(ofile)
    ofile.close()

    print(f"working hard on {fpath}")
    y, logy = read_data(data)

    print("computing limits of y...")
    muy = np.mean(y, axis=1)
    sty = np.std(y)
    histy = np.histogram(muy)[1]
    mu_std0 = muy + sty
    mu_std1 = muy - sty

    print("computing limits of logy...")
    sklogy = skew(logy, axis=1)
    stlogy = np.std(logy, axis=1)
    mulogy = np.mean(logy, axis=1)

    print("getting the min and max...")
    y_minlims.append(fname)
    y_minlims.append(y.shape)
    y_minlims.append(np.min(muy))
    y_minlims.append(np.min(sty))
    y_minlims.append(np.min(histy))
    y_minlims.append(np.min(mu_std0))
    y_minlims.append(np.min(mu_std1))

    y_maxlims.append(fname)
    y_maxlims.append(np.max(muy))
    y_maxlims.append(np.max(sty))
    y_maxlims.append(np.max(histy))
    y_maxlims.append(np.max(mu_std0))
    y_maxlims.append(np.max(mu_std1))

    logy_minlims.append(fname)
    logy_minlims.append(np.min(mulogy))
    logy_minlims.append(np.min(sklogy))
    logy_minlims.append(np.min(stlogy))

    logy_maxlims.append(fname)
    logy_maxlims.append(np.max(mulogy))
    logy_maxlims.append(np.max(sklogy))
    logy_maxlims.append(np.max(stlogy))


# max & mins
print(f"mins for y {y_minlims}")
print(" " * 1000)

print(f"maxs for y {y_maxlims}")
print(" " * 1000)

print(f"mins for logy {logy_minlims}")
print(" " * 1000)

print(f"maxs for logy {logy_maxlims}")
