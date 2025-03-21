import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
"""
This file iterates through all matrices of calcium/spike data in the folder "data", 
each row (individual neuron) is checked for NaNs, then saved as a new file 
This minimizes impacts on memory when loading in the future, and avoids NaN issues.

This new 1d array is saved into the folder "data/processed"

Note: For spikefinder data, once there is a NaN in timeseries, all future values will be NaN.
"""



states = ['test', 'train']
for stat in states:
    if stat == 'test':
        dsets = [1, 2, 3, 4, 5]
    if stat == 'train':
        dsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
    for dset in dsets:
        # load in true spikes
        file_path1 = 'data/' + str(dset) + '.' + str(stat) + '.calcium.csv'
        file_path2 = 'data/' + str(dset) + '.' + str(stat) + '.spikes.csv'
        
        spikeDat = pd.read_csv(file_path2).T 
        spikeDat = np.array(spikeDat)
        
        calcDat = pd.read_csv(file_path1).T 
        calcDat = np.array(calcDat)
        mDat, nDat = np.shape(calcDat)

        mSpike,nSpike = spikeDat.shape       

        # Remove NaNs, save rows as individual files. 
        i = 0
        while i < mSpike:
            naninds = np.isnan(calcDat[i,:])
            NaNpresent = np.any(naninds)
            if NaNpresent == True:
                subsetAmount = ((np.where(naninds == True))[0][0]) - 1 #index of first Nan, less one. 
            else:
                subsetAmount = np.max(np.shape(calcDat))

            # with NaNs out, pull that row and save as a new file.
        
            spikeDatSingle = spikeDat[i, :subsetAmount]
            calcDatSingle = calcDat[i, :subsetAmount]
        

            file_path1R = 'data/processed/node'+ str(i) + '_dset' + str(dset) + '.' + str(stat) + '.calcium'
            file_path2R = 'data/processed/node'+ str(i) + '_dset' + str(dset) + '.' + str(stat) + '.spikes'
            np.save(file_path1R, calcDatSingle)
            np.save(file_path2R, spikeDatSingle)

            i = i + 1 




       
