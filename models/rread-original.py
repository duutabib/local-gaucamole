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
outfile_root = "/Users/duuta/ppp/data/"


def read_file(file_path):
    """return file given a path"""
    y = loadmat(file_path, simplify_cells=True)
    x0 = y["stim"]["spont"]
    x1 = y['stim']["resp"]
    print(f"currently reading {file_path}")
    return x0, x1


def read_files(file_paths):
    """return list data objects"""
    datalist = []
    with multiprocessing.Pool(processes=1) as pool:
        data = pool.map(read_file, file_paths)
        datalist.append(data)
        return datalist


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

    # declare some variables
    start_time = time.time()
    max_iter = 100000
    stim_types = [0, 1]

    # data header
    print("setting the header of output csv........")
    header = ["mean", "std", "skew", "type"]

    # get all data files
    print("getting all file names to load into mem")
    file_list = list(glob.glob(f"{data_root}natimg2800_M*.mat"))
    lenfile_list = len(file_list)

    # load read data in mem
    red_datalist = read_files(file_list)
    print("frantically read all the data files into memory")

    with open(f"{outfile_root}pipeline.csv", "w+") as csvFile:
        tablet = csv.writer(csvFile)
        tablet.writerow(header)

        for _ in range(max_iter):
            print(f"row index is {ri}")
            curr_data0, curr_data1= red_datalist[0][ri]
            ar0 = random.sample(list(curr_data0), 1)
            ar1 = random.sample(list(curr_data1), 1)
            ar0 = np.log10(np.clip(ar0, 1, np.inf))
            ar1 = np.log10(np.clip(ar1, 1, np.inf))
            mu, std_, sk = np.mean(ar), np.std(ar), skew(ar, axis=1)
            _row = [mu, std_, sk[0], stim_types[0]]
            print(f"updating data file with {mu=} {std_=} {sk[0]=} {target=}")
            tablet.writerow(_row)
        print(f"finished reading and writing files in {time.time() - start_time}")
