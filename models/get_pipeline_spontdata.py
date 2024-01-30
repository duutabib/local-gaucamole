import csv
import glob
import multiprocessing
import random
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
    print(f"currently reading {file_path}")
    return x0, x1


def read_split_on_x(file_path, max_rows):
    """return data given file path"""
    y = loadmat(file_path, simplify_cells=True)
    x00 = y["stim"]["spont"]
    mu = x00.mean()
    mask = x00[:max_rows,] > mu
    mask0 = ~mask
    x0 = ma.masked_array(x00, mask=mask, fill_value=0).data
    x1 = ma.masked_array(x00, mask=mask0, fill_value=0).data

    return x0, x1


def read_files(file_paths):
    """return list data objects"""
    slist, rlist = [], []
    with multiprocessing.Pool(processes=1) as pool:
        data0, data1 = zip(*pool.map(read_file, file_paths))
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
    file_list = list(glob.glob(f"{data_root}natimg2800_M*.mat"))
    lenfile_list = len(file_list)

    # load read data in mem
    slist, rlist = read_files(file_list)
    print("here is the data...")
    print("frantically read all the data files into memory")

    with open(f"{outfile_root}pipeline_spont.csv", "w+") as csvFile:
        tablet = csv.writer(csvFile)
        tablet.writerow(header)

        for arr0, arr1 in zip(slist[0], rlist[0]):
            for row0, row1 in zip(arr0, arr1):
                mu0, std0, sk0 = np.mean(row0), np.std(row0), skew(row0)
                mu1, std1, sk1 = np.mean(row1), np.std(row1), skew(row1)
                _row0 = [mu0, std0, sk0, stim_types[0]]
                _row1 = [mu1, std1, sk1, stim_types[1]]
                print(f"updating data file with {mu0=} {std0=} {sk0=} {stim_types[0]=}")
                print(f"updating data file with {mu1=} {std1=} {sk1=} {stim_types[1]=}")
                tablet.writerow(_row0)
                tablet.writerow(_row1)
        print(f"finished reading and writing files in {time.time() - start_time}")
