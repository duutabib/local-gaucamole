import math
import pdb
import pickle
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import skew

import helpers_data00 as h
import utils as u

print("franctically getting all file paths...")
data_root = "/Users/duuta/ppp/data/data00/pickled/"
saveROOT = "/Users/duuta/ppp/dashBoards/data00/test/"

# need to exactly match the titles
names_root = glob("/Users/duuta/ppp/data/data00/*spont_*.mat")
nameslist = [gname.split("/")[-1].strip("spont_") for gname in names_root]
print(nameslist)


data_files = glob(f"{data_root}*.pickle")
print(data_files)
data_names = [
    fname.split("/")[-1].strip(".pickle").strip("spont_") for fname in data_files
]


title_dict = {"spont": "Spontaneous", "resp": "Response"}
tag = "spont"
# dasheader = f'Dashboard for {title_dict[tag]} Activity'


print("frantically reading all files ...")
# read files all
N = len(data_files)
max_rows = 8000
max_cols = 8000


print(f"there are {N} files")
for i, fpath in enumerate(data_files):
    print(f"reading data {i}................")
    set_filename = f"Dashboard of {title_dict[tag]} Activity: " + nameslist[i]

    # load pickle files
    ofile = open(fpath, "rb")
    data = pickle.load(ofile)
    ofile.close()

    print(f"working hard on {fpath}")
    spont = data["Fsp"]
    print(spont.shape)
    print("Here is the load", spont.shape)
    spon = spont[:max_rows, :max_cols]
    print("Shape after sampling", spon.shape)
    print("min spon", np.min(spon), np.max(spon))
    print("min spon>0", np.min(spon[spon > 0]))

    spon0 = np.clip(spon, 1, np.inf)

    logd = np.log10(spon0)
    print(skew(logd))

    print("done with clipped version")

    # resp_ = h.dupSignal(resp, istim)

    # spon1 += np.random.randn(*spon1.shape)*1e-16

    sspon = h.dupSpont(spon0)

    print("preping data for structures...")
    # Jresp = np.array(random.sample(list(resp), max_rows))

    tcluster = time.time()
    row_order, col_order = h.ro_orders(spon)
    print(f"time for clustering {time.time() - tcluster}")

    print("computing...covariances...")
    ss_spont = u.shuff_cvPCA(sspon, nshuff=10)
    ss_spont_ = ss_spont.mean(axis=0)
    ss_ = ss_spont_ / ss_spont_.sum()

    print("computing power laws...")
    alpha_fit, ypred, b_ = u.get_powerlaw(ss_, np.arange(10, 1e2).astype("int"))
    alf = round(alpha_fit, 2)

    print("generating dashboard...")

    h.imboard(
        spon0, row_order, col_order, ss_spont_, ypred, alpha_fit, b_, set_filename
    )
    plt.savefig(f"{saveROOT}dash{i}-{tag}.png")

    print(f"Saved file...{i}")
    plt.close()
