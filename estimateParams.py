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
    x, z, ga_x, ga_z, gk_x, gk_z  = A
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, L = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    #s = impulseSineSimple(t)
    s = np.interp(t, timeVec, spikeDatRaw)
    
    # augmented system with time evolution of grad x, z
    # gradients taken wrt alpha, kr for optimization. Denoted ga_x and gk_x respectively
    # rhs of ODES d/dt[x, z, ga_x, ga_z, gk_x, gk_z]
    
    du = [alpha*s - gamma*x + kr*z - kf*x*(L-z), 
          kf*x*(L-z) - kr*z,
          s - gamma*ga_x + kr*ga_z - kf*ga_x*L + kf*ga_x*z + kf*x*ga_z, 
          kf*ga_x*L - kf*ga_x*z - kf*x*ga_z - kr*ga_z,
          -gamma*gk_x + kr*gk_z + z - kf*gk_x*L + kf*gk_x*z + kf*x*gk_z,
          kf*gk_x*L - kf*gk_x*z - kf*x*gk_z - z - kr*gk_z] 
    
    return du


# alpha*s - gamma*gx + kr*gz - kf*gz*L + kf*gx*z + kf*x*gz,
#        kf*gx*L - kf*gx*z - kf*x*gz - kr*gz] 
    


def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    #initialize weights  
    beta = np.random.randn(X.shape[1])
    for i in range(iterations):
        y_pred = X.dot(beta)
        error = y_pred - y
        gradient = 2 * X.T @ (y_pred - y) / X.shape[0]
        #approximate the gradient and update weights 
        beta -= learning_rate * gradient
        if i % 100 == 0:
            loss = np.mean(error ** 2)
            print(f"Iteration {i}: loss = {loss}, beta = {beta}")    
    return beta


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
    #print("Simulating until final time", f"{tEnd/60:.3f}", "minutes, consisting of", n, "data points")
    timeVec = np.linspace(0, tEnd, n, endpoint = False) #used in interp call
    

    # load in actual ground truth spike data
    file_path2 = 'data/processed/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'
    spikeDatRaw = np.load(file_path2)
    spikeDatRaw = spikeDatRaw[:n] 
    
    # define initial conditions
    Ca_0 = 5    
    CiF_0 = CI_Meas[0]  
    gx_alpha_0 = 0.5
    gz_alpha_0 = 0.5
    gx_kr_0 = 0.5
    gz_kr_0 = 0.5
    
    kf, kr, alpha, gamma, L = paramValues(dset, CiF_0)

    #alpha = 5
    #kr = 4
    alpha = np.random.uniform(5, 20)
    kr = np.random.uniform(2,10)
    alpha = 0.5
    kr = 2

    gamma = 10
    kf = 0.1

    error_prev = 0

    numStep = 200
    for i in range(numStep):
        print("Beginning grad descent step", i)
        
        x0 = np.array([Ca_0, CiF_0, gx_alpha_0, gz_alpha_0, gx_kr_0, gz_kr_0])

        
        # pack up parameters and ICs
        p = [kf, kr, alpha, gamma, L]


        # Solve ODEs with CI_meas, real spike data involved    
        sol = solve_ivp(CRN_alpha, [0, tEnd], x0, t_eval=timeVec)
     
        Ca_f =          sol.y[0, :]
        CiF_f =         sol.y[1, :]
        grad_X_alpha =  sol.y[2, :]
        grad_Z_alpha =  sol.y[3, :]
        grad_X_kr =     sol.y[4, :]
        grad_Z_kr =     sol.y[5, :]

        CiF_f = sigmoid(CiF_f) 

        timeStep = timeVec[1] - timeVec[0]
        

        g_L_alpha = np.sum(-2*(CI_Meas - CiF_f)*grad_Z_alpha*timeStep)
        g_L_kr =  np.sum(-2*(CI_Meas - CiF_f)*grad_Z_kr*timeStep)

        # compute gradient w.r.t each param, the same??

        # step
        rho_alpha = 1  # learning rates
        rho_kr =  1    # learning rate
        alpha = alpha - rho_alpha*g_L_alpha
        kr = kr - rho_kr*g_L_kr
    
        
        print("alpha:", alpha)
        print("kr:", kr)
        # compute loss wrt sigmoid of CI_meas, CI_sim. Utilizing 2norm. 
        error_current = np.linalg.norm(CI_Meas - CiF_f)#/len(CiF_f)
        error_dif = error_current - error_prev
        error_prev = error_current


        print("Relative MSE of measured calcium tracking tracking:", error_current)
        print("difference in error between previous step:", error_dif)
        print("#________________________________________________#")
        #del(sol) 



    stop = time.time()
    runTime = (stop - start)/60
    print(f"{runTime:.3f}")
