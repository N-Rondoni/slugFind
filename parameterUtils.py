"""
This file holds parameters supplied to MPCmain.py

Dependent on data set. 
"""
import numpy as np


def paramValues(dset, CiF_0):
    if dset in [1, 2, 3]: 
        kf = 0.1
        kr = 10 
        alpha = 16.666666 
        gamma = 1.3666666    # passive diffusion
        L = CiF_0 + 100      # total amount of calcium indicator, assumes L units of unflor. calcium indicator
    
    if dset == 5: # params found from BFS, worked well on 5 for sure.
        kf = 0.2
        kr = 10 
        alpha = 10 
        gamma = 0.73333            
        L = CiF_0 + 100

    if dset == 4:
        kf = 0.1
        kr = 10 
        alpha = 10 
        gamma = 0.73333   
        L = CiF_0 + 100 

    if dset in [6, 7, 8, 9, 10]: 
        kf = 0.1
        kr = 10 
        alpha = 16.666666 
        gamma = 1.3666666   
        L = CiF_0 + 100  

    return kf, kr, alpha, gamma, L


def paramValuesBFS(dset, CiF_0):
    if dset in [1, 2, 3]: 
        kf = 0.1
        kr = 10 
        alpha = 16.666666 
        gamma = 1.3666666    # passive diffusion
        L = CiF_0 + 100      # total amount of calcium indicator, assumes L units of unflor. calcium indicator
    
    if dset == 5: # params found from BFS, worked well on 5 for sure.
        kf = 0.2
        kr = 10 
        alpha = 10 
        gamma = 0.73333            
        L = CiF_0 + 100

    if dset == 4:
        kf = 0.1
        kr = 10 
        alpha = 10 
        gamma = 0.73333   
        L = CiF_0 + 100 

    if dset in [6, 7, 8, 9]: 
        kf = 0.1
        kr = 10 
        alpha = 16.666666 
        gamma = 1.3666666   
        L = CiF_0 + 100  

    return kf, kr, L


def learnedParams(dset):#, row, CiF_0):
    row = 0
    kf = 0.1
    kr = 10
    stat = "train"
    
    # These two are the old dataset, before I realized to subtract vs before grad desc. 
    #gamma = 10
    #saveLoc = 'data/oldParamEstimation/alphas_node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) +  '_params_' + 'gamma_'+ str(gamma) +  'kf_' + str(kf) + 'kr_' + str(kr) + '.npy' 

    gamma = 1
    saveLoc = 'data/paramEstimation/alphas_node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) +  '_params_' + 'gamma_'+ str(gamma) +  'kf_' + str(kf) + 'kr_' + str(kr) + '.npy' 
    # format: alphas_node0_dset1.train_params_gamma_1kf_0.1kr_10.npy    
    alphas = np.load(saveLoc)
    #print(alphas)
    return alphas[1]


