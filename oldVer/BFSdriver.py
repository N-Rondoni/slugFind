import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from spikefinder_eval import _downsample
"""
This file iterates through all data sets, both train and test, and supplies the run information to MPCmain.py
In essense, this file produces command line inputs such as

python3 MPCmain.py 0 1 test
"""

dsets = [1, 2, 3, 4, 5]
status = ['test', 'train'] 


for stat in status:
    if stat == 'train':
        dsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # there are more training sets than test sets. 

    for dset in dsets:
        # load in data so you know how many rows are in a dset
        file_path = 'data/' + str(dset) + '.' + str(stat) + '.calcium.csv'
        data1 = pd.read_csv(file_path).T 
        data1 = np.array(data1)
        mDat, nDat = np.shape(data1) 
        # loop through rows, each corresponds to a neuron. 
        i = 0
        while i < mDat:
            start  = time.time()
            print("Beginning solve on", str(stat) ,"data set", str(dset) + ", neuron",  i)
            
            # check for NaNs
            naninds = np.isnan(data1[i,:])
            #if the below evaluates to true, there are NaNs in the dataset.
            NaNpresent = np.any(naninds)

            if NaNpresent == True:
                print("This neuron's data contained NaNs! Solving up until NaNs begin... ") 
                
            # run solver, if NaNs present main will simulate up until they begin. 
            os.system("python3 MPCmain.py " + str(i) + " " + str(dset) + " " + str(stat))
            end = time.time()
            print("previous solve for neuron", i, "completed in", (end - start)/60, "minutes")
            print("------------------------------------------------------------------------")
            i = i + 1

