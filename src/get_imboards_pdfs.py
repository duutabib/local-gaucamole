import os
import sys
import pdb
import scipy
import pickle
import random
import timeit
import numpy as np
import helpers as h
from glob import glob
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gridspec


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/"
oroot = "/Users/duuta/ppp/data/stringer/dash-spont/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)



title_dict = {'spont': 'Spontaneous', 'resp': 'Response'}
tag = 'spont'
#dasheader = f'Dashboard for {title_dict[tag]} Activity'

if __name__ == "__main__":
    print("frantically reading all files ...")
    # read files all 
    N = len(picklepaths)
    max_rows = np.inf

    print(f"there are {N} files")
    for i, fpath in enumerate(picklepaths):
        print(f"reading data {i}................")
        set_filename = f"Dashboard of {title_dict[tag]} Activity: " + data_names[i]
        
        ofile = open(fpath, 'rb')
        data = pickle.load(ofile)['stim']


        print(f'working hard on {fpath}')
        #data = rlist[0][i]

        resp = data[tag]
        #resp = np.array(random.sample(list(resp), max_rows))

        resp += np.random.randn(*resp.shape)*1e-16

        print("preping data for structures...")
        tcluster = time.time()
        row_order, col_order = h.ro_orders(resp)
        print(f"time for clustering {time.time() - tcluster}")

        print("computing...covariances...")
        ss_resp = h.test_vars(resp)

        print("computing power laws...")
        alpha_fit, ypred = h.get_alpha_fit(ss_resp, np.arange(10, 1e2).astype('int'))	

        h.imboard(resp, row_order, col_order, ss_resp, ypred, alpha_fit, set_filename)
        plt.savefig(f"{data_root}dash{i}-{tag}.png")

        ofile.close()

        print(f"Saved file...{i}")
