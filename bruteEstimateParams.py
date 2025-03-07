import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from spikefinder_eval import _downsample
from parameterUtils import paramValuesBFS
from scipy.integrate import solve_ivp


def sigmoid(signal):
    sig = 1/(1 + np.exp(-1*(signal - 1)))
    return sig


def CRN(t, A):
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        A : vector of the state variables: Ca^{2+}, CI^* respectively. 
                A = [x, z]
        t : time
        p : vector of the parameters:
                p = [kf, kr, alpha, gamma]
        s :  must be one of the impulse functions above, e.g., impulseSquare(t)
             remember to change this in plotting functions below as well. 
    """
    x, z = A
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, L = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    #s = impulseSineSimple(t)
    s = np.interp(t, timeVec, spikeDatRaw)
    
    # augmented system with time evolution of grad x, z
    # rhs of ODES d/dt[x, z, gx, gz]
    du = [alpha*s - gamma*x + kr*z - kf*x*(L-z), 
          kf*x*(L-z) - kr*z]

    return du

def minInd(bb):
    # returns indices of minimal element, 2x2 array. 
    amin = np.argmin(bb)
    index =  [amin // np.shape(bb)[0], amin % np.shape(bb)[1]]
    return index

if __name__=="__main__":
    start = time.time() # begin timer
    alphas = np.linspace(0, 5, 5)
    gammas  = np.linspace(0, 20, 5)
    loss = np.zeros((np.shape(alphas)[0], np.shape(gammas)[0]))
    

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
    #print("Simulating until final time", f"{tEnd/60:.3f}", "minutes, consisting of", n, "data points")
    timeVec = np.linspace(0, tEnd, n, endpoint = False) #used in interp call
    

    # load in actual ground truth spike data
    file_path2 = 'data/processed/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'
    spikeDatRaw = np.load(file_path2)
    spikeDatRaw = spikeDatRaw[:n] 
    

    # define initial conditions
    Ca_0 = 5    
    CiF_0 = CI_Meas[0]  
    x0 = np.array([Ca_0, CiF_0])

    # replace this with a meshgrid call for generalizations
    for k in range(len(alphas)):
        for j in range(len(gammas)):
            #print(alphas[k], betas[j])
            
            alpha = alphas[k]
            gamma = gammas[j]
            kf, kr, L = paramValuesBFS(dset, CiF_0)
           
            # pack up parameters and ICs
            p = [kf, kr, alpha, gamma, L]


            # Solve ODEs with CI_meas, real spike data involved    
            sol = solve_ivp(CRN, [0, tEnd], x0, t_eval=timeVec)

           
            Ca_f =  sol.y[0, :]
            CiF_f = sol.y[1, :]

            # compute loss wrt sigmoid of CI_meas, CI_sim. Utilizing 2norm. 
            #print("Relative MSE of measured calcium tracking tracking:", np.linalg.norm(CI_Meas - CiF_f)/len(CiF_f))

            loss[k, j]  = np.linalg.norm(CI_Meas - CiF_f)/len(CiF_f)


    stop = time.time()
    runTime = (stop - start)/60
    print("Total runtime", f"{runTime:.3f}", "minutes.", "Looped over",len(alphas)*len(gammas), "parameters.")
    print(loss)
    print("minimum difference:", np.min(loss))
    index = minInd(loss)
    
    print(loss[index[0], index[1]])
    
    print("created with alpha =", alphas[index[0]], "gamma =", gammas[index[1]])
    saveLoc = 'data/paramEstimation/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '_params'

    np.save(saveLoc, index)
