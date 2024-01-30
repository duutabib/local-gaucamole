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


def read_centers(file_path):
    """return centered data"""
    data = loadmat(file_path, simplify_cells=True)
    x, y, z = data["med"].T

    return x, y, z


def read_files(file_paths):
    """return list data objects"""
    slist, rlist, tlist = [], [], []
    with multiprocessing.Pool(processes=1) as pool:
        data0, data1, data2 = zip(*pool.map(read_centers, file_paths))
        slist.append(data0)
        rlist.append(data1)
        tlist.append(data2)

        return slist, rlist, tlist


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
    max_iter = 10000
    stim_types = [0, 1, 2]

    # data header
    print("setting the header of output csv........")
    header = ["mean", "std", "skew", "type"]

    # get all data files
    print("getting all file names to load into mem")
    file_list = list(glob.glob(f"{data_root}natimg2800_M*.mat"))
    lenfile_list = len(file_list)

    # load read data in mem
    slist, rlist, tlist = read_files(file_list)
    print("here is the data...")
    print("frantically read all the data files into memory")

    with open(f"{outfile_root}pipeline_centers.csv", "w+") as csvFile:
        tablet = csv.writer(csvFile)
        tablet.writerow(header)

        for arr0, arr1, arr2 in zip(slist, rlist, tlist):
            for row0, row1, row2 in zip(arr0, arr1, arr2):
                mu0, std0, sk0 = np.mean(row0), np.std(row0), skew(row0)
                mu1, std1, sk1 = np.mean(row1), np.std(row1), skew(row1)
                mu2, std2, sk2 = np.mean(row2), np.std(row2), skew(row2)
                _row0 = [mu0, std0, sk0, stim_types[0]]
                _row1 = [mu1, std1, sk1, stim_types[1]]
                _row2 = [mu2, std2, sk2, stim_types[2]]
                print(f"updating data file with {mu0=} {std0=} {sk0=} {stim_types[0]=}")
                print(f"updating data file with {mu1=} {std1=} {sk1=} {stim_types[1]=}")
                print(f"updating data file with {mu2=} {std2=} {sk2=} {stim_types[2]=}")
                tablet.writerow(_row0)
                tablet.writerow(_row1)
                tablet.writerow(_row2)
        print(f"finished reading and writing files in {time.time() - start_time}")
