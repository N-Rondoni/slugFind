""""
# Purpose:     This file is functionally equivalent to MPCmain.py, but makes lovely plots that would otherwise slow down the process when iterating through all datasets. 
               Infers underlying neuronal firing as a function of calcium imaging data. Dependent on ODEs derived from chemical reaction network.
#              This file contains the main contributions of the work "Predicting Neuronal Firing from Calcium Imaging Using a Control Theoretic Approach"
# Author:      Nicholas A. Rondoni
# Use example: In command line, enter
#              MPCmain.py 0 1 test
#              This will simulate spiking times for neuron 0, dataset 1, test, of spikefinder challenge and compare it to the recorded ground truth. 
# Note:        driver.py will loop over all neurons in every dataset, but this file can be used for individual neurons.  
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import do_mpc
from casadi import *
import pandas as pd
import seaborn as sns
from datetime import date
import spikefinder_eval as se
from spikefinder_eval import _downsample
from parameterUtils import paramValues

def tvp_fun(t_now):
    for k in range(n_horizon + 1):
        tvp_template['_tvp', k, 'Ci_m'] = np.interp(t_now + k*t_step, timeVec, CI_Meas)
    return tvp_template

def tvp_fun_sim(t_now):
    tvp_template1['Ci_m'] = np.interp(t_now, timeVec, CI_Meas)
    return tvp_template1

def plotTwoLines(x, y1, y2):
    plt.figure(1)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$ (mol)')
    #plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$ (mol)')
    plt.plot(x, y2, linewidth = 2, label = r'$CI^{*}$ (mol)')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'MPC_fig1_node' + str(row) + '.png'
    
def plotFourLines(x, y1, y2, y3, y4):
    plt.figure(2)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$ (mol)')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$ (mol)')
    plt.plot(x, y3, '--', linewidth = 1, label = r'$CI^{*}$ (mol)')
    plt.plot(x, y4, '--', linewidth = 1, label = r'$s$ (Hz)')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration (mol), Rate (Hz)', fontsize = 14)
    plt.legend()
    filename = 'MPC_fig2_node' + str(row) + '.png'
    
def plotErr(x, y):
    plt.figure(2)
    plt.plot(x, y, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(i) + '.png'


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
    print("Simulating until final time", tEnd/60, "minutes, consisting of", n, "data points")
    timeVec = np.linspace(0, tEnd, n, endpoint = False) #used in interp call
    

    # define initial conditions
    Ca_0 = 5    
    CiF_0 = CI_Meas[0]  
    x0 = np.array([Ca_0, CiF_0])

    # set up do-mpc model
    model_type = 'continuous' 
    model = do_mpc.model.Model(model_type)
    
    # states struct, optimization variables
    Ca = model.set_variable('_x', 'Ca')
    CiF = model.set_variable('_x', 'CiF')

    # define parameters 
    penalty = 0.01
    # pull dset dependent params from parameterUtils.py
    kf, kr, alpha, gamma, L = paramValues(dset, CiF_0)
            
    tstep = 1/100 # tstep of solver, can be reworked to be different from imRate, requires interpolation. 
   
    s = model.set_variable('_u', 's')         # control variable
    CI_m = model.set_variable('_tvp', 'Ci_m') # time varying parameter, measured CI_m


    # define ODE eqns, calcium ion and calcium indicator fluoresced respectively.  
    model.set_rhs('Ca', alpha*s - gamma*Ca + kr*CiF - kf*Ca*(L - CiF))
    model.set_rhs('CiF', (kf*Ca*(L - CiF) - kr*CiF))
   
    model.setup()
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters, can change collocation/state discretization here.
    setup_mpc = {
            'n_horizon': 6,  # pretty short horizion
            't_step': tstep, # (s)
            'n_robust': 0,
            'store_full_solution': True,
            }
  
    mpc.settings.supress_ipopt_output() # supresses output of solver

    mpc.set_param(**setup_mpc)
    n_horizon = 6
    t_step = tstep
    n_robust = 0 

    # define objective, which is to miminize the difference between Ci_m and CiF
    mterm =(sigmoid(model.x['CiF']) - model.tvp['Ci_m'])**2     
    lterm = mterm


    mpc.set_objective(mterm = mterm, lterm = lterm)
    mpc.set_rterm(s = penalty) # sets a penalty on changes in s, defined at top of main for ease
    # see latex doc for description, but essentialy adds penalty*(s_k - s_{k-1}) term.

    # make sure the objective/cost updates with CI_measured and time.    
    tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(tvp_fun)

    # enforce constraints
    mpc.bounds['lower', '_u', 's'] = 0 # spikes cannot be negative
    
    # once mpc.setup() is called, no model parameters can be changed.
    mpc.setup()
    
    # Estimator: assume all states can be directly measured
    estimator = do_mpc.estimator.StateFeedback(model)

    # Simulator
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
            'integration_tool': 'cvodes', # look into this
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': tstep, # (s) mean step is 6.11368547250401 in data
            }
    simulator.set_param(**params_simulator)
    
    # account for tvp
    tvp_template1 = simulator.get_tvp_template() # must differ from previous template name
    simulator.set_tvp_fun(tvp_fun_sim)

    simulator.setup()
          
    # set for controller, simulator, and estimator
    mpc.x0 = x0  
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()

    # finally perform closed loop simulation
    n_steps = int(tEnd/tstep) # ensures we hit final time
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    end = time.time()
    print("Solve completed in",  (end-start)/60, "minutes")

    # pull final solutions for ease of use, _f denotes final version of that param
    Ca_f = mpc.data['_x'][:, 0]     # calcium ion
    CiF_f = mpc.data['_x'][:, 1]    # calcium indicator fluoresced
    t_f = mpc.data['_time']         # time vector 
    s = mpc.data['_u']              # spiking signal

    # compute quantity of calcium indicator unbound (not fluoresced)
    Ci_f = L - CiF_f

    # sigmoid calcium indicator so it is comoparable to calcium ion. 
    CiF_f = sigmoid(CiF_f)

    plotTwoLines(t_f, Ca_f, CiF_f)
    plotFourLines(t_f, Ca_f, Ci_f, CiF_f, s)


    # check error between Ci_M and Ci_sim
    CI_Meas_interp = np.interp(t_f, timeVec, CI_Meas) # interp call only does something if tstep isn't the same as imRate
    CI_Meas_interp = CI_Meas_interp[:, 0]             # reshaping

    s = np.array(s[:,0])                        # reshape s for future computations    
    s_interp = np.interp(timeVec, t_f[:,0], s)  # interp command only does something if tstep isn't the same as imRate

    print("Relative MSE of measured calcium tracking tracking:", np.linalg.norm(CI_Meas_interp - CiF_f)/len(CiF_f))

    # load in actual ground truth spike data
    file_path2 = 'data/processed/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'
    spikeDatRaw = np.load(file_path2)
    spikeDatRaw = spikeDatRaw[:n] 
    
    
    ##---------------------------------------------------------------------------------##
    # POST PROCESS BELOW, saving, downsampling, and correlation comutations.            #
    ##---------------------------------------------------------------------------------##
    
    # save raw simulated spiking signal for later eval
    saveLoc = 'data/processed/solutions/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.sVals'
    s = s_interp
    #np.save(saveLoc, s) only save in MPCmain to ensure you know where files come from. 

    # decide what factor to downsample by 
    factor = 4
    spikeDat = _downsample(spikeDatRaw, factor)
    s = _downsample(s, factor)
    m1 = min([len(s), len(spikeDat)]) # needed if interpolated lengths differ. Should never happen if tstep = imRate. 
    spikeDat = spikeDat[0:m1]
    newTime = np.linspace(0, t_f[-1], m1)#, endpoint = True) # only used for plotting

   
    # finally scale so viewing is more clear
    s = (np.max(spikeDat)/np.max(s))*s # correlation coeff. invariant wrt scaling. 

    # compute correlation coefficient 
    corrCoef = np.corrcoef(s[100:], spikeDat[100:])[0, 1] # toss first second of recording, can contain bad transient dynamics
    print("Corr Coef:", corrCoef)

    # plotting routines below 
    neuron = row
    plt.figure(4)
    plt.plot(t_f, CiF_f, label=r'$CI^{*}_{Sim}$') ## subtracting baseline
    plt.plot(t_f, CI_Meas_interp, label=r'$CI^{*}_{Meas}$', alpha = 0.7)
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'Tracking_dset'+ str(dset) + "_neuron" + str(neuron)
    plt.savefig(filename)
        
    subL, subH = 2000, 4000
    #subL, subH = 800, 1200

    plt.figure(5)
    plt.plot(t_f[subL:subH], CiF_f[subL:subH], label=r'$CI^{*}_{Sim}$') ## subtracting baseline
    plt.plot(t_f[subL:subH], CI_Meas_interp[subL:subH], label=r'$CI^{*}_{Meas}$', alpha = 0.7)
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'Tracking_subset_dset'+ str(dset) + "_neuron" + str(neuron)
    plt.savefig(filename)

    plt.figure(6)
    subL, subH = int(subL/factor), int(subH/factor)
    plt.plot(newTime[subL:subH], s[subL:subH], label=r'Simulated Rate')
    plt.plot(newTime[subL:subH], spikeDat[subL:subH], label="Recorded Spike Rate", alpha = 0.7)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.legend()
    plt.title("Expected and Recorded spikes")#, bin size of " + str(1000*binSizeTime) + " ms")
    filename = 'Spikes_dset'+ str(dset) + "_neuron" + str(neuron)
    plt.savefig(filename)
        
    fig, axs = plt.subplots(3, 1)
    color2 = "orange"
    color1 = "lightseagreen"
    gridAxis = "both"
    axs[0].plot(t_f[int(subL*factor):int(factor*subH)], CiF_f[int(factor*subL):int(factor*subH)], label=r'$CI^{*}_{Sim}$', color = color1) ## subtracting baseline
    axs[0].plot(t_f[int(subL*factor):int(factor*subH)], CI_Meas_interp[int(factor*subL):int(factor*subH)], label=r'$CI^{*}_{Meas}$', color = color2)
    #axs[0].set_xlabel(r'$t$', fontsize = 16)
    axs[0].set_ylabel(r'$CI^*$', fontsize = 16)
    axs[0].grid(axis = gridAxis)
    axs[0].legend(loc = 'upper right')#(fontsize = 14)
   
  
    CorrCoef = np.corrcoef(s[subL:subH], spikeDat[subL:subH])[0, 1] # lessed for dset5, neuron 5
    print("Subset's corr:", CorrCoef)
    s = s/np.max(s[subL:subH])
    spikeDat = spikeDat/np.max(spikeDat[subL:subH])

    axs[1].plot(newTime[subL:subH], spikeDat[subL:subH], label=r'$s_{meas}$', color = color2)
    axs[1].set_ylabel(r'Recorded Spikes', fontsize = 16)        
    axs[1].legend(fontsize = 14, loc = 'upper right')
    axs[1].grid(axis = gridAxis)

    axs[2].plot(newTime[subL:subH], s[subL:subH], label=r'$s_{sim}$', color = color1)
    axs[2].set_xlabel(r'$t$', fontsize = 18)
    axs[2].set_ylabel(r'Simulated Spikes', fontsize = 16)        
    axs[2].grid(axis = gridAxis)
    plt.legend(fontsize = 14, loc = 'upper right')
    fig.suptitle(r"MPC Approximation of Firing Rates", fontsize = 18)
    plt.tight_layout()


    plt.show()
