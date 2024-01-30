import csv
import sys
import glob
import time
import random
from itertools import islice
import numpy as np
import multiprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold

infile_root = "/Users/duuta/ppp/data/"

max_rows = 100
max_cols = 3
tmax_cols = 1

def read_file_into_array(file_path):
    """ return file given a path"""
    X = np.empty(shape=(max_rows, max_cols), dtype='float')
    y = np.empty(shape=(max_rows, tmax_cols), dtype='int')

    print(X.shape)
    print(y.shape)

    with open(file_path, "r") as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for i, row in enumerate(islice(reader, 100)):
            X[i, :] = row[:3]
            y[i, :] = int(row[3])

        y = y.ravel()
        print(f'...rows and cols completed')
    return X, y 


def data_lru():
    """cache data for repeated calls"""
    # complete implementation later today
    pass

if __name__ =="__main__":

    # declare some variables
    start_time = time.time()
    
    file_path = infile_root+'scdata_file.csv'
    
    # create hold outs for CV responses
    kfold = RepeatedKFold(n_splits = 5, n_repeats=10, random_state=None)
    skfold = StratifiedKFold(n_splits = 5, n_repeats= 10, random_state=None)

    # read data 
    X, y = read_file_into_array(file_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True,)

    # init model and pass in data
    print("quickly init model & passing in train data")
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    
    # get model features
    print("franctically getting model features...")
    mparams = model.get_params()
    mpredict = model.predict(X_test) 
    # residue 
    mscore = model.score(X_test, y_test)
    
    print(f"raw perf {mscore=}")
    
    # cross validate 
    cv_res = cross_validate(model, X, y)
    print("here cv scores:", cv_res['test_score'])
    

    # get predictions
    cv_predictions = cross_val_predict(model, X, y, cv=5)
    # print('Here are the predictions', cv_predictions)

    # finish time
    dur = time.time() - start_time
    print(f"took {dur} secs...")

