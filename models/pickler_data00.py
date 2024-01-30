import glob
import multiprocessing
import pickle
import time

from scipy.io import loadmat

inroot = "/Users/duuta/ppp/data/data00/"
outroot = "/Users/duuta/ppp/data/data00/pickled/"


def rfile(file_path, **args):
    """return file given a path"""
    y = loadmat(file_path, simplify_cells=True)
    return y


def qread(function, flist, pnu):
    with multiprocessing.Pool(processes=pnu) as pool:
        y = pool.map(function, flist)
    return y


if __name__ == "__main__":
    # declare some variables
    start_time = time.time()
    max_elements = 7

    # get all data files
    print("getting all file names to load into mem")
    flist = glob.glob(f"{inroot}*spont_*.mat")[:max_elements]

    dholder = qread(rfile, flist, max_elements)

    for j, data in enumerate(dholder):
        with open(f"{outroot}data_{j}.pickle", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"dumped data {j} for pickle")
    print(f"dur for the pickle dumps {time.time() - start_time}")
