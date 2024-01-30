import os
import sys
import pdb
import scipy
import random
import numpy as np
import helpers as h
from glob import glob
from scipy.io import loadmat
from mpl_toolkits import mplot3d
from scipy.stats import skewnorm 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.stats import skew, kurtosis
import matplotlib.gridspec as Gridspec
from scipy.cluster import hierarchy as H
sys.path.append("/Users/duuta/ppp/msc/")
from rread import read_file, read_files
import pdb


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/"
data_files = list(glob(f"{data_root}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
picklepath = list(glob(f"{data_root}data_*.pickle"))


title_dict = dict({'spont': 'Spontaneous', 'resp': 'Response'})
tag = 'spont'
#dasheader = f'Dashboard for {title_dict[tag]} Activity'

if __name__ == "__main__":
    print("frantically reading all files ...")
    # read files all 
    rlist = read_files(data_files)
    N = len(data_files)
    max_rows = 200

    print(f"there are {N} files")
    for i in range(N):
        print(f"reading data {i}................")
        set_filename = f"Dashboard of {title_dict[tag]} Activity: " + data_names[i]

        print(f'working hard on {data_files[i]}')
        data = rlist[0][i]

        resp = data[tag]
        resp = np.array(random.sample(list(resp), max_rows))

        resp += np.random.randn(*resp.shape)*1e-16

        print("preping data for structures...")
        row_order, col_order = h.ro_orders(resp)

        print("computing...covariances...")
        ss_resp = h.test_vars(resp)

        print("computing power laws...")
        alpha_fit, ypred = h.get_alpha_fit(ss_resp, np.arange(10, 1e2).astype('int'))	

        h.imboard(resp, row_order, col_order, ss_resp, ypred, alpha_fit, set_filename)
        plt.savefig(f"/Users/duuta/ppp/data/PDFs/dash{i}-{tag}.png")

        print(f"Saved file...{i}")
