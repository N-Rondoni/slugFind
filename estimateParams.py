import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

def sigmoid(signal):
    sig = 1/(1 + np.exp(-1*(signal - 1)))
    return sig



if __name__=="__main__":
    start = time.time() # begin timer
 
    # passed in from driver.py or manually in command line, 
    # row denotes a neuron's index, dset the dset number, and stat the status of either "train" or "test". 
    row = int(sys.argv[1])    
    dset = int(sys.argv[2])
    stat = str(sys.argv[3])

    # load in data
    file_path = 'data/processed/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.calcium.npy'
    
    CI_Meas = np.load(file_path)
    n = len(CI_Meas)

    # apply sigmoidal filter to help minimize noise
    CI_Meas = sigmoid(CI_Meas) - 0.15

    # set up timevec, recordings were resampled to 100 hz
    imRate = 1/100
    tEnd = n*(imRate) 
    print("Simulating until final time", f"{tEnd/60:.3f}", "minutes, consisting of", n, "data points")
    timeVec = np.linspace(0, tEnd, n, endpoint = False) #used in interp call
    

    # load in actual ground truth spike data
    file_path2 = 'data/processed/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'
    spikeDatRaw = np.load(file_path2)
    spikeDatRaw = spikeDatRaw[:n] 
    

    # define initial conditions
    Ca_0 = 5    
    CiF_0 = CI_Meas[0]  
    x0 = np.array([Ca_0, CiF_0])

    # Solve ODEs with CI_meas, real spike data involve



    stop = time.time()
    runTime = (stop - start)/60
    print(f"{runTime:.3f}")
