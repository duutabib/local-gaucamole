import csv
import glob
import math
import multiprocessing
import pdb
import sys
import time

import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
from scipy.stats import skew

data_root = "/Users/duuta/ppp/data/stringer/live_data/"
outfile_root = "/Users/duuta/ppp/data/"


def read_file(file_path):
    """return file given a path"""
    y = loadmat(file_path, simplify_cells=True)
    x = y["stim"]["spont"]
    mu = x.mean()
    filter0 = x > mu
    filter1 = ~filter0
    x0 = ma.masked_array(x, mask=filter0, fill_value=0).data
    x1 = ma.masked_array(x, mask=filter1, fill_value=0).data

    return x0, x1


def read_split_on_x(file_path, max_rows=200):
    """return data given file path"""
    y = loadmat(file_path, simplify_cells=True)
    x00 = y["stim"]["spont"]
    curr_rows = x00.shape[0]
    quota = max_rows / curr_rows
    if quota < 0.5:
        max_rows = math.ceil(0.5 * curr_rows)
    x0 = x00[:max_rows,]
    x1 = x00[max_rows:curr_rows,]

    return x0, x1


def compute_features(arr, stype=0):
    """return features for row"""
    new_row = []
    for row in arr:
        mu, std, sk = np.mean(row), np.std(row), skew(row)
        new_row.append([mu, std, sk, stype])

    return new_row


def read_split_on_y(file_path, max_rows=200):
    """return data given file path"""
    y = loadmat(file_path, simplify_cells=True)
    x00 = y["stim"]["spont"]
    curr_rows = x00.shape[0]
    quota = max_rows / curr_rows
    if quota < 0.5:
        max_rows = math.ceil(0.5 * max_rows)
    x0 = x00[:, :max_rows]
    x1 = x00[:, max_rows:curr_rows]

    return x0, x1


def read_files(file_paths):
    """return list data objects"""
    slist, rlist = [], []
    with multiprocessing.Pool(processes=1) as pool:
        data0, data1 = zip(*pool.map(read_split_on_x, file_paths))
        slist.append(data0)
        rlist.append(data1)

        return slist, rlist


def data_lru():
    """cache data for repeated calls"""
    # complete implementation later today
    pass


if __name__ == "__main__":
    # manually fixing sys.modules
    import cProfile

    if sys.modules["__main__"].__file__ == cProfile.__file__:
        import rread

        globals().update(vars(rread))
        sys.modules["__main__"] = rread

    # set some variables
    start_time = time.time()
    stim_types = [0, 1]

    # data header
    print("setting the header of output csv........")
    header = ["mean", "std", "skew", "type"]

    # get all data files
    print("getting all file names to load into mem")
    file_list = glob.glob(f"{data_root}natimg2800_M*.mat")
    lenfile_list = len(file_list)

    # load read data in mem
    slist, rlist = read_files(file_list)
    print("spont", slist[0])
    print("resp", rlist[0])

    print("here is the data...")
    print("frantically read all the data files into memory")

    with open(f"{outfile_root}pipeline_spilt_on_y.csv", "w+") as csvFile:
        tablet = csv.writer(csvFile)
        tablet.writerow(header)

        for arr0 in slist[0]:
            _row0 = compute_features(arr0, stype=0)
            tablet.writerows(_row0)

        for arr1 in rlist[0]:
            _row1 = compute_features(arr1, stype=1)
            tablet.writerows(_row1)
        print(f"finished reading and writing files in {time.time() - start_time}")
