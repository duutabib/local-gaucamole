import os
import pdb
import time
import pickle
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



# need to write a more versitile read function.
def read_files(file0, file1):
    """returns two data arrays (M X N)"""

    print(f"reading data ................")
    ofile0 = open(file0, 'rb')
    ofile1 = open(file1, 'rb')

    data0 = pickle.load(ofile0)
    data1 = pickle.load(ofile1)
    
    ofile0.close() 
    ofile1.close()
    
    return data0, data1


def clean_unpack(data0):
    """returns two clean and zscored signal"""
    resp0, spon0, istim0  = h.unbox(data0)

    resp0 = h.denoise_resp(resp0, spon0)

    resp0 = h.dupSignal(resp0, istim0)

    return resp0


def compute_cvPCA(resp0):
    ss0 = u.shuff_cvPCA(resp0, nshuff=10)
    ss0 = ss0.mean(axis=0)
    ss0 = ss0/ss0.sum()

    return ss0


if __name__ == "__main__":

    d0, d1 = read_files(picklepaths[0], picklepaths[1])
    dc0 = clean_unpack(d0)
    dc1 = clean_unpack(d1)

    s0 = compute_cvPCA(dc0)
    s1 = compute_cvPCA(dc1)
    
    s0_alp, s0_ypred = u.get_powerlaw(s0, np.arange(11, 5e2).astype('int'))
    s1_alp, s1_ypred = u.get_powerlaw(s1, np.arange(11, 5e2).astype('int'))

    fig = plt.figure()
    ax = plt.axes()

    plt.loglog(np.arange(0, s0.shape[0])+1, s0, label='region0' )
    plt.loglog(np.arange(0, s0.shape[0])+1, s1, label='region1' )
    ax.axline((0, 0), slope=1, ls='---')
    plt.ylabel("Variance", fontsize=20)
    plt.xlabel("PC Dimension", fontsize=20)
    plt.title("OverlayPlot for data from diff brain regions", fontsize=20)
    plt.savefig(f"{saveROOT}ovelay_diff_regions.png")
    plt.close()

    

