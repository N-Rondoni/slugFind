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
    row = 0#int(sys.argv[1])    
    dset = 3#int(sys.argv[2])
    stat = "train"#str(sys.argv[3])

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
    gx_0 = 0.5
    gz_0 = 0.5

    kf, kr, alpha, gamma, L = paramValues(dset, CiF_0)
    # suppose instead params are pulled from lit
    kf = 0.032
    kr = 8

    alpha = np.random.uniform(1, 20)
    gamma = 1
    error_prev = 0

    numStep = 100
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
   
        CiF_f = sigmoid(CiF_f) 

        
        ga_L = np.sum(-2*(CI_Meas - CiF_f)*grad_Z*0.1) # should be 0.01, doesn't matter? 

        # step
        rho = .1  # learning rate
        alpha = alpha - rho*ga_L
        #gamma = gamma - rho*gL
        
        print("alpha:", alpha)
        print("gamma:", gamma)
        # compute loss wrt sigmoid of CI_meas, CI_sim. Utilizing 2norm. 
        error_current = np.linalg.norm(CI_Meas - CiF_f)#/len(CiF_f)
        error_dif = error_current - error_prev
        error_prev = error_current


        print("Relative MSE of measured calcium tracking tracking:", error_current)
        print("difference in error between previous step:", error_dif)
        print("#_____________________________#")
        #del(sol) 



    stop = time.time()
    runTime = (stop - start)/60
    print(f"{runTime:.3f}")
