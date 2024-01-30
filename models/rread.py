import csv
import glob
import multiprocessing
import random
import sys
import time

import numpy as np
from scipy.io import loadmat
from scipy.stats import skew

data_root = "/Users/duuta/ppp/data/stringer/"
img_root = "/Users/duuta/ppp/msc/pngs/"
outfile_root = "/Users/duuta/ppp/data/"


def read_file(file_path, **args):
    """return file given a path"""
    y = loadmat(file_path, simplify_cells=True)
    return y["stim"]


def read_files(file_paths):
    """return list data objects"""
    datalist = []
    with multiprocessing.Pool(processes=3) as pool:
        data = pool.map(read_file, file_paths)
        datalist.append(data)
        return datalist


def basic_clean(arr):
    curr_arr = arr[arr > 0]
    minvalue = np.min(curr_arr)
    # set min value for array
    curr_arr = np.clip(curr_arr, minvalue, np.inf)

    return curr_arr


def data_lru():
    """cache data for repeated calls"""
    # complete implementation later today
    pass


if __name__ == "__main__":
    # manually fixing sys.modules for cProfiling
    import cProfile

    if sys.modules["__main__"].__file__ == cProfile.__file__:
        import rread

        globals().update(vars(rread))
        sys.modules["__main__"] = rread

    # declare some variables
    start_time = time.time()
    stim_types = [0, 1]  # stimules types
    max_elements = 3
    max_rows = 100

    # data header
    print("setting the header of output csv........")
    header = ["mean", "std", "skew", "type"]

    # get all data files
    print("getting all file names to load into mem")
    file_list = list(glob.glob(f"{data_root}natimg2800_M*.mat"))[:max_elements]

    # load read data in mem
    data_holder = read_files(file_list)
    print("frantically read all the data files into memory")

    with open("pipeline.csv", "w+") as f:
        rite = csv.writer(f, delimiter=",")
        rite.writerow(header)
        for arr in data_holder[0][:max_elements]:
            print(arr)
            for _ in range(max_rows):
                curr_data_spont = arr["spont"][:max_rows]
                curr_data_resp = arr["resp"][:max_rows]
                cc_spont = basic_clean(curr_data_spont)
                cc_resp = basic_clean(curr_data_resp)
                ar0 = random.sample(list(cc_spont), 1)
                ar1 = random.sample(list(cc_resp), 1)
                mu0, std_0, sk_spont = np.mean(ar0), np.std(ar0), skew(ar0, axis=1)
                mu1, std_1, sk_resp = np.mean(ar1), np.std(ar1), skew(ar1, axis=1)
                _row0 = [mu0, std_0, sk_spont[0], stim_types[0]]
                _row1 = [mu1, std_1, sk_resp[0], stim_types[1]]
                rite.writerow(_row0)
                rite.writerow(_row1)
                print(
                    f"updating data file with {mu0=} {std_0=} {sk_spont[0]=} {stim_types[0]=}"
                )
                print(
                    f"updating data file with {mu1=} {std_1=} {sk_resp[0]=} {stim_types[1]=}"
                )
            print(f"finished reading and writing files in {time.time() - start_time}")
