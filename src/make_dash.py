import os
import sys
import time
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
saveROOT = "/Users/duuta/ppp/dashBoards/dash-resp/"

data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
print("where are data files ", data_names)
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)


title_dict = {'spont': 'Spontaneous', 'resp': 'Response'}
tag = 'resp'
#dasheader = f'Dashboard for {title_dict[tag]} Activity'


print("frantically reading all files ...")
# read files all 
N = len(picklepaths)
# max_rows = 200

start_time = time.time()
print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")
    set_filename = f"Dashboard of {title_dict[tag]} Activity: " + data_names[i]
    
    # load pickle files
    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()


    print(f'working hard on {fpath}')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    resp_ = h.dupSignal(resp, istim)

    #resp += np.random.randn(*resp.shape)*1e-16

    print("preping data for structures...")
    # resp = np.array(random.sample(list(resp), max_rows))

    tcluster = time.time()
    row_order, col_order = h.ro_orders(resp)
    print(f"time for clustering {time.time() - tcluster}")

    print("computing...covariances...")
    ss_resp = u.shuff_cvPCA(resp_, nshuff=10)
    ss_resp_ = ss_resp.mean(axis=0)
    
    ss_  = ss_resp_/ss_resp_.sum()

    print("computing power laws...")
    alpha_fit, ypred, b_  = u.get_powerlaw(ss_, np.arange(10, 5e2).astype('int'))	
    alf = round(alpha_fit, 2)
    

    h.imboard(resp, row_order, col_order, ss_resp_, ypred, alpha_fit, b_,  set_filename)
    plt.savefig(f"{saveROOT}dash{i}-{tag}.png")


    print(f"Saved file...{i}")
    plt.close()

print(f"duration {time.time() - start_time}")
