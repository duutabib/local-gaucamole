import os
import pdb
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
nroot = "/Users/duuta/ppp/data/stringer/"
saveROOT="/Users/duuta/ppp/plots/makeplots/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
print("Here the data files ", data_names)
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)




print("frantically reading all files ...")
# read files all 
N = len(picklepaths)

ss_arr = [] # arr for  
maxrows = 2366

print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()

    print(f'unboxing  {fpath} for resp, spon, istim...')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    resp_ = h.dupSignal(resp, istim)

    print("computing...covariances...")
    ss_resp = u.shuff_cvPCA(resp_, nshuff=10)
    ss_resp_ = ss_resp.mean(axis=0)
    
    ss_  = ss_resp_/ss_resp_.sum()
    ss_ = ss_[:maxrows]
    ss_arr.append(ss_)


plt.loglog(np.transpose(ss_arr), label='all recordings')
plt.xlim([1, 2800])
plt.ylim([10**-5, 10**-1])
plt.ylabel("Variance", fontsize='x-large')
plt.xlabel("PC Dimension", fontsize='x-large')
plt.title("Plot 2D", fontsize='x-large')
plt.savefig(f"{saveROOT}plot_twod_check.png")
plt.close()
