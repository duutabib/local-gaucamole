import math
import os
import pdb
import pickle
from numba import jit
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import helpers as h
import utils as u

print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
# nroot = "/Users/duuta/ppp/data/stringer/live_data/"
saveROOT = "/Users/duuta/ppp/plots/makeplots/"
# data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
# data_names = [
#   fname.split("/")[-1].strip(".mat").strip("natimg2800_") for fname in data_files
#]
# print("Here the data files ", data_names)
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print("here are all pickle paths", picklepaths)


# need to write a more versitile read function.
def read_files(file0):
    """returns two data arrays (M X N)"""

    print(f"reading data ................")
    ofile0 = open(file0, "rb")

    data0 = pickle.load(ofile0)

    ofile0.close()

    return data0

@jit
def compute_alpha_subpops(arr0, size=100):
    """return a list of alpha values of selection
    maxp: step size as in np.arange()
    data0: arr N*M (N stimuli, M neurons"""

    alphaValues = []  # init list for alpha vals
    minpop = 29
    n = arr0.shape[2]

    pops = np.arange(n, step=size)

    pops0 = pops[minpop:, ].tolist()

    for pop in tqdm(pops0):

        X0 = arr0[:, :, 0:pop]
        ss0 = compute_cvPCA(X0)
        nss0 = ss0.shape[0]
        nX0 = int(math.ceil(nss0* 0.05))
        alpha, _ , _= u.get_powerlaw(ss0, np.arange(11, nX0).astype('int'))
        alpha = np.round(alpha, 2)

        alphaValues.append(alpha)

    return alphaValues, pops0


def clean_unpack(data0):
    """returns clean and zscored signal"""
    resp0, spon0, istim0 = h.unbox(data0)

    resp0 = h.denoise_resp(resp0, spon0)

    resp0 = h.dupSignal(resp0, istim0)

    return resp0


def compute_cvPCA(resp0):
    ss0 = u.shuff_cvPCA(resp0, nshuff=10)
    ss0 = ss0.mean(axis=0)
    ss0 = ss0 / ss0.sum()

    return ss0


if __name__ == "__main__":
    data0 = read_files(picklepaths[0]) 

    d0 = clean_unpack(data0)

    yVals, xVals = compute_alpha_subpops(d0, size=100)

    print(xVals, yVals)

    fig = plt.figure(figsize=(15, 20))
    ax = plt.axes()

    plt.scatter(xVals, yVals, c=yVals, cmap="Spectral")
    plt.axline((0, 0), slope=1, ls="--")
    plt.ylabel("Alpha Values", fontsize=20)
    plt.xlabel("Size of subPopulations of ROIs", fontsize=20)
    plt.title("Alpha values of localy decreasing subpopulations of ROIs", fontsize=20)
    plt.colorbar()
    plt.savefig(f"{saveROOT}alpha_decreasing_pop_rois.png")
    plt.close()
