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
nroot = "/Users/duuta/ppp/data/stringer/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
print("Here the data files ", data_names)
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)
pdb.set_trace()



title_dict = dict({'spont': 'Spontaneous', 'resp': 'Response'})
tag = 'resp'
#dasheader = f'Dashboard for {title_dict[tag]} Activity'

print("frantically reading all files ...")
# read files all 
N = len(picklepaths)
max_rows = 200

ivals = [] # arr for alpha vals
ss_arr = [] # arr for  
ssp_arr = [] # arr for partial set of neurons
fracs = [1.000, 0.500, 0.250, 0.125, 0.062, 0.031, 0.016]

print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")
    set_filename = f"Dashboard of {title_dict[tag]} Activity: " + data_names[i]

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)


    print(f'working hard on {fpath}')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    resp_ = h.dupSignal(resp, istim)

    #resp += np.random.randn(*resp.shape)*1e-16

    print("preping data for structures...")
    resp = np.array(random.sample(list(resp), max_rows))

    tcluster = time.time()
    row_order, col_order = h.ro_orders(resp)
    print(f"time for clustering {time.time() - tcluster}")

    print("computing...covariances...")
    ss_resp = u.shuff_cvPCA(resp_)
    ss_resp_ = ss_resp.mean(axis=0)
    
    ss_  = ss_resp_/ss_resp_.sum()
    ss_arr.append(ss_)

    print("computing power laws...")
    alpha_fit, ypred = u.get_powerlaw(ss_, np.arange(10, 1e2).astype('int'))	
    alf = round(alpha_fit, 2)
    
    ivals.append(alf)

    h.imboard(resp, row_order, col_order, ss_resp_, ypred, alpha_fit, set_filename)
    plt.savefig(f"dash{i}-{tag}.png")

    ofile.close()

    print(f"Saved file...{i}")
    plt.close()


print("Here are the vals of alpha", ivals)
print("making plot 2E")
plt.hist(ivals, bins=[0.9, 1.0, 1.1])
plt.xlabel("values of alpha")
plt.ylabel("No of recordings")
plt.title("Plot 2E")
plt.savefig('plot2E.png')
plt.close()


print("making plot 2D")
plt.loglog(ss_arr, label='all recordings')
plt.ylabel("Variance")
plt.xlabel("PC Dimension")
plt.title("Plot 2D")
plt.savefig("plot2D.png")
plt.close()


nn= ss_resp_.shape[0]
print("making plot 2G")
for frac in fracs:
    rsxy = random.sample(list(ss_resp_), math.ceil(frac*nn))
    nrsxy = len(rsxy)
    plt.loglog(np.arange(0, nrsxy) + 1, rsxy, label=f'frac=')

plt.ylabel("Variance")
plt.xlabel("Dimension")
plt.title("Plot 2G")
plt.savefig("plot2G.png")
