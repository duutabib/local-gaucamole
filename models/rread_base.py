import csv
import sys
import glob
import time
import random
import numpy as np
import multiprocessing
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from scipy.stats import skew, kurtosis

data_root = "/Users/duuta/ppp/data/stringer/"
outfile_root = "/Users/duuta/ppp/data/"

def read_file(file_path, **args):
    """ return file given a path"""
    y = loadmat(file_path, simplify_cells=True)
    return y['stim']


def read_files(file_paths):
    """ return list data objects """
    datalist = []
    with multiprocessing.Pool(processes=1) as pool:
        data = pool.map(read_file, file_paths)
        datalist.append(data)
        return datalist 

def data_lru():
    """cache data for repeated calls"""
    # complete implementation later today
    pass

if __name__ =="__main__":
    # manually fixing sys.modules for cProfiling
    #import cProfile
    #if sys.modules['__main__'].__file__ == cProfile.__file__:
    #    import rread
    #    globals().update(vars(rread))
    #    sys.modules['__main__']= rread


    # declare some variables
    start_time = time.time()
    max_iter = 100000
    max_elements = 7
    max_rows = 100
    target_set = list(range(7))

    # data header 
    print("setting the header of output csv........")
    header = ['mean', 'std', 'skew', 'target']

    # get all data files
    print("getting all file names to load into mem")
    file_list = list(glob.glob(f"{data_root}natimg2800_M*.mat"))
    lenfile_list = len(file_list)

    # load read data in mem
    data_holder = read_files(file_list)
    print("frantically read all the data files into memory")

    with open(f'{outfile_root}scdata_file.csv', 'w+') as csvFile:
        tablet = csv.writer(csvFile)
        tablet.writerow(header)
        c = 0
        for arr, i in zip(data_holder[0][:max_elements], range(max_elements)): 
            curr_data = arr['spont'][:max_rows]
            niter = curr_data.shape[0]
            for _ in range(1):
                target = i 
                curr_data0= curr_data[curr_data>0]
                minvalue = np.min(curr_data0)
                # set min value for array 
                curr_data_clip = np.clip(curr_data, minvalue, np.inf)
                plt.plot(curr_data_clip)
                plt.savefig(f"curr_data_{i}.png")

                #ar = random.sample(list(curr_data), 1)
                #xas = range(len(ar))
                #plt.plot(xas, ar)
                #plt.savefig(f'dist{c}.png')
                #c += 1
                #ar = np.log10(np.clip(ar, 1, np.inf))
                #mu, std_, sk = np.mean(ar), np.std(ar), skew(ar, axis=1)
                #_row = [mu, std_, sk[0], target]
                #print(f"updating data file with {mu=} {std_=} {sk[0]=} {target=}")
                #tablet.writerow(_row)
            print(f"finished reading and writing files in {time.time() - start_time}")

