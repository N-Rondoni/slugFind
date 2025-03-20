import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from spikefinder_eval import _downsample
from parameterUtils import paramValues
from scipy.integrate import solve_ivp
from scipy.integrate import quad


def sigmoid(signal):
    sig = 1/(1 + np.exp(-1*(signal - 1)))
    return sig


def CRN_alpha(t, A):
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
    x, z, ga_x, ga_z  = A
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, L = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    #s = impulseSineSimple(t)
    s = np.interp(t, timeVec, spikeDatRaw)
    
    # augmented system with time evolution of grad x, z
    # rhs of ODES d/dt[x, z, ga_x, ga_z]
    
    du = [alpha*s - gamma*x + kr*z - kf*x*(L-z), 
          kf*x*(L-z) - kr*z,
          s - gamma*ga_x + kr*ga_z - kf*ga_x*L + kf*ga_x*z + kf*x*ga_z, 
          kf*ga_x*L - kf*ga_x*z - kf*x*ga_z - kr*ga_z] 
    
    return du


# alpha*s - gamma*gx + kr*gz - kf*gz*L + kf*gx*z + kf*x*gz,
#        kf*gx*L - kf*gx*z - kf*x*gz - kr*gz] 
    


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
    vs = 0.15
    CI_Meas = sigmoid(CI_Meas) - vs
    #if dset == 

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
    gx_0 = 0.5
    gz_0 = 0.5

    kf, kr, alpha, gamma, L = paramValues(dset, CiF_0)
    # suppose instead params are pulled from lit
    #kf = 0.032
    #kr = 8
    
    kf = 0.1
    kr = 10

    alpha = np.random.uniform(1, 30)
    alpha_list = np.arange(3, 40, 15)
    print(alpha_list)

    gamma = 1 #dsets  6, ... require a higher value of gamma or alpha will go negative. A RESULT OF forgetting vertical shift

    error_prev = 0
    error_min = 100
    paramsOut = [] # to be filled with final alpha of gradient descent

    numStep = 200
    for alpha in alpha_list:
        #print(alpha)
        for i in range(numStep):
            print("Beginning grad descent step", i)
            

            x0 = np.array([Ca_0, CiF_0, gx_0, gz_0])

            
            # pack up parameters and ICs
            p = [kf, kr, alpha, gamma, L]


            # Solve ODEs with CI_meas, real spike data involved    
            sol = solve_ivp(CRN_alpha, [0, tEnd], x0, t_eval=timeVec)
         
            Ca_f =  sol.y[0, :]
            CiF_f = sol.y[1, :]
            grad_X = sol.y[2, :]
            grad_Z = sol.y[3, :]
       
            CiF_f = sigmoid(CiF_f) - vs
            
            timeStep = timeVec[1] - timeVec[0]
            
            ga_L = np.sum(-2*(CI_Meas - CiF_f)*grad_Z*timeStep) 
            #print("gradient wrt alpha", ga_L)
            # step
            rho = 1  # learning rate
            alpha = alpha - rho*ga_L
        
            
            print("alpha:", alpha)

            # compute loss wrt sigmoid of CI_meas, CI_sim. Utilizing 2norm. 
            error_current = np.linalg.norm(CI_Meas - CiF_f)#/len(CiF_f)
            error_dif = error_current - error_prev
            error_prev = error_current

            error_MSE =  np.mean((CI_Meas - CiF_f)**2)

            if error_current < error_min:
                error_min = error_current
                alpha_best = alpha
            if (i+1) % numStep == 0:
                alpha_fin = alpha
                paramsOut = np.append(paramsOut, alpha_fin)

            print("2norm of measured calcium tracking tracking:", error_current)
            print("difference in error between previous step:", error_dif)
            print("MSE of measured calcium tracking tracking:", error_MSE)
            #print("Raw Differences, summed", np.sum((CI_Meas - CiF_f)**2)*timeStep)
            print("#____________________________________________________________________#")
    
    print("Min error found: ", error_min)
    print("Created by using alpha = ", alpha_best,"gamma = ", gamma,  "kf = ", kf, "kr = ", kr)
    # prepend best alpha to paramsOut
    paramsOut = np.insert(paramsOut, 0, alpha_best, axis=0)
    print(paramsOut)
    saveLoc = 'data/paramEstimation/alphas_node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '_params_' + 'gamma_'+ str(gamma) +  'kf_' + str(kf) + 'kr_' + str(kr)

    np.save(saveLoc, paramsOut)

    stop = time.time()
    runTime = (stop - start)/60
    print("Total run took", f"{runTime:.3f}", " minutes.")
