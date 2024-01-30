import os
import sys
import pdb
import scipy
import math
import time
import pickle
import random
import utils as u
import numpy as np
import helpers as h
from glob import glob
import matplotlib.pyplot as plt


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/live_data/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
picklepaths = list(glob(f"{data_root}data_*.pickle"))[:2]
print('here are all pickle paths', picklepaths)
saveROOT="/Users/duuta/ppp/plots/makeplots/"


print("frantically reading all files ...")

# these are the fractions
fracs = [1.000, 0.500, 0.250, 0.125, 0.0625, 0.0312, 0.0156]
nl = fracs.__len__()

# def arr to store values
arr0 = []
arr1 = []

lsarr = [arr0, arr1]

for fpath, arr in zip(picklepaths, lsarr):
    for frac in fracs[:4]:

        print(f"reading data................")
        ofile = open(fpath, 'rb')
        data = pickle.load(ofile)
        ofile.close()

        print(f'unboxing {fpath} for resp, spon, istim.....')
        resp, spon, istim = h.unbox(data)
        resp = h.denoise_resp(resp, spon)

        print('getting total # of neurons....')
        tnn = resp.shape[1]   # total number of neurons
        frac_neurons = math.ceil(frac*tnn) # frac of neurons
        
        print(f'getting this {frac=} of neurons')
        resp = resp[:, :frac_neurons]
        
        print('duplicating signal.......')
        resp_ = h.dupSignal(resp, istim)
        print('print shape of resp_', resp_.shape)

        print("computing...cross validated PCA...")
        ss0 = u.shuff_cvPCA(resp_, nshuff=10)
        ss0 = ss0.mean(axis=0)

        print('normalizing power laws...')
        ss0 = ss0/ss0.sum()

        arr.append(ss0)

    ar0 = np.array(arr0)
    ar1 = np.array(arr1)
    srr = sum(arr0, arr1)
    srr_list = srr.tolist()


for aar, frac in zip(fracs, (srr)): 

    plt.loglog(aar/aar.sum(), label=f'{frac=}')
    plt.xlabel("Dimension")
    plt.ylabel("Variance")

plt.savefig(f'{saveROOT}plot_twog.png')
plt.legend()
plt.close()

# step0
# read data and denoise
# subset neurons
