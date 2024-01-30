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
import spontHelpers as h
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gridspec
from scipy.stats import skew 


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/live_data/"
saveROOT = "/Users/duuta/ppp/dashBoards/dash-spont/"

data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
print("where are data files ", data_names)
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)


title_dict = {'spont': 'Spontaneous', 'resp': 'Response'}
tag = 'spont'
#dasheader = f'Dashboard for {title_dict[tag]} Activity'


print("frantically reading all files ...")
# read files all 
N = len(picklepaths)


print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")
    set_filename = f"Dashboard of {title_dict[tag]} Activity: " + data_names[i]

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()


    print(f'working hard on {fpath}')
    _, spon, istim = h.unbox(data)
    #resp = h.denoise_resp(resp, spon) # can't denoise spont
    print('min spon', np.min(spon), np.max(spon))
    print('min spon>0', np.min(spon[spon>0]))

    spon0 = np.clip(spon, 1, np.inf)
    
    logd = np.log10(spon0)
    print(skew(logd))
    
    print("done with clipped version")

    #resp_ = h.dupSignal(resp, istim)

    #spon1 += np.random.randn(*spon1.shape)*1e-16
        
    sspon = h.ssplit(spon0)
    

    print("preping data for structures...")
    #Jresp = np.array(random.sample(list(resp), max_rows))

    tcluster = time.time()
    row_order, col_order = h.ro_orders(spon)
    print(f"time for clustering {time.time() - tcluster}")


    print("computing...covariances...")
    ss_spont = u.shuff_cvPCA(sspon, nshuff=10)
    ss_spont_ = ss_spont.mean(axis=0)
    ss_ = ss_spont_/ss_spont_.sum()


    print("computing power laws...")
    alpha_fit, ypred, b_  = u.get_powerlaw(ss_, np.arange(10, 1e2).astype('int'))	
    alf = round(alpha_fit, 2)


    print('generating dashboard...')

    h.imboard(spon0, row_order, col_order, ss_spont_, ypred, alpha_fit, b_, set_filename)
    plt.savefig(f"{saveROOT}dash{i}-{tag}.png")

    print(f"Saved file...{i}")
    plt.close()
