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
from scipy.stats import skew
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as Gridspec


print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/stringer/dpickle/"
nroot = "/Users/duuta/ppp/data/stringer/"
data_files = list(glob(f"{nroot}natimg2800_M*.mat"))
data_names = [fname.split('/')[-1].strip('.mat').strip("natimg2800_") for fname in data_files]
picklepaths = list(glob(f"{data_root}data_*.pickle"))
print('here are all pickle paths', picklepaths)



#title_dict = {'spont': 'Spontaneous', 'resp': 'Response'}
#tag = 'resp'

print("frantically reading all files ...")
# read files all 
N = len(picklepaths)
max_rows = 200


maxskew_lims = []
maxmean_lims = []
maxstd_lims  = []
maxmas_lims  = []
maxresp_lims = []

minskew_lims = []
minmean_lims = []
minstd_lims  = []
minmas_lims  = []
minresp_lims = []


print(f"there are {N} files")
for i, fpath in enumerate(picklepaths):
    print(f"reading data {i}................")

    ofile = open(fpath, 'rb')
    data = pickle.load(ofile)
    ofile.close()


    print(f'working hard on {fpath}')
    resp, spon, istim = h.unbox(data)
    resp = h.denoise_resp(resp, spon)
    
    #resp_ = h.dupSignal(resp, istim)

    #resp += np.random.randn(*resp.shape)*1e-16

    #resp = np.array(random.sample(list(resp), max_rows))
    

    print("computing the log of the distribution")
    logr = np.log10(np.clip(resp, 1, np.inf))
    
    print("computing stats...")
    mean_logr = np.mean(logr)
    skew_logr = skew(logr)
    std_logr = np.std(logr)
    mas = np.mean(resp, axis=0) 
    
    #mean_lims.append(mean_logr)
    #skew_lims.append(skew_logr)
    #std_lims.append(std_logr)
    #mas_lims.append(mas)
   
    print("getting the min and max...")
    minmean_lims.append(np.min(mean_logr))
    maxmean_lims.append(np.max(mean_logr))
    minskew_lims.append(np.min(skew_logr))
    maxskew_lims.append(np.max(skew_logr))
    minstd_lims.append(np.min(std_logr))
    maxstd_lims.append(np.max(std_logr))
    minmas_lims.append(np.min(mas))
    maxmas_lims.append(np.max(mas))
   


# max & mins
print(f'mean_logr min: {np.min(minmean_lims)}, max: {np.max(maxmean_lims)}')
print(f'std min: {np.min(minstd_lims)}, max: {np.max(maxstd_lims)}')
print(f'skew_logr min: {minskew_lims}, max: {maxskew_lims}')
print(f'mas_logr min: {np.min(minmas_lims)}, max: {np.max(maxmas_lims)}')
print(f'resp min: {np.min(resp)}, max: {np.max(resp)}')
