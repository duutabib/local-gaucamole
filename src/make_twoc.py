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
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gridspec


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/live_data/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)
saveROOT="/Users/duuta/ppp/plots/makeplots/"


print("frantically reading all files ...")
# read files all 
N = len(picklepaths)

tss0 = 0 # store totals of normalized PCA
maximgs = 2366

print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()

    print(f'working hard on {fpath}')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    print('duplicating signal.......')
    resp_ = h.dupSignal(resp, istim)

    print("computing...cross validated PCA...")
    ss0 = u.shuff_cvPCA(resp_, nshuff=10)
    ss0 = ss0.mean(axis=0)
    ss0 = ss0[:maximgs]
    
    
    print('getting the shapes of arrs ...')
    print(f'here is the shape of the computed cvPCA {ss0.shape}')

    tss0 += ss0
    
    #print(f'the shape for the running totals {tss0.shape}')
    
    print('normalizing power laws...')
    ss0a, ypredss0, _ = u.get_powerlaw(ss0/ss0.sum(), np.arange(11, 5e2).astype('int'))
    print('alpha value', ss0a)
    
    plt.loglog(np.arange(0, ss0.shape[0])+1, ss0/ss0.sum(), label='saw')
    plt.loglog(np.arange(0, ss0.shape[0])+1, ypredss0, label='wanted')
    plt.savefig(f'{saveROOT}plot_{i}.png')
    plt.close()
    
    
tss = ss0
print('writing total to all data npy')
print('shape', tss0.shape)

print("making twoc....")
ntss0 = tss0.shape[0]
av0, ypred0, _ = u.get_powerlaw(tss0/tss0.sum(), np.arange(11, 5e2).astype('int'))
fav = round(av0, 2)

print('normalized total results', av0, np.max(ypred0), ypred0)

print("av0", fav)
print('plotting tss0 which is the sum of the ss0')
plt.loglog(np.arange(0, ntss0)+1, tss0/tss0.sum(), color='blue',  label='observed')
plt.loglog(np.arange(0, ntss0)+1, ypred0, label=f'{fav=}', color='black')
plt.text(10**2, 10**-2, r'$\alpha=1.04$', color='blue', horizontalalignment='center', size='xx-large')
plt.xlabel("PC Dimension", fontsize='xx-large')
plt.ylabel("Variance", fontsize='xx-large')
plt.xlim([10**0, 2800])
plt.ylim([10**-5, 10**0])
plt.title("Plot 2C")
plt.legend()
plt.savefig(f"{saveROOT}plot_twoc.png")
print("saved  plot twoc...")

# plotted the loglog totals: ypred0 is off... missed the scaling of tss0; alpha is correct.
# fixed the scaling of tss0 in get_powerlaw... 
# no clipping here... still the plot falls short; data doesn't reach full extent. 
# changed u.shuff_cPCA() call to u.cPCA(); u.cPCA returns zero withut a shuffle.
# set nshuff=10

