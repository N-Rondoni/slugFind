"""
This file holds parameters supplied to MPCmain.py

Dependent on data set. 
"""
import numpy as np

def learnedParams(dset):#, row, CiF_0):
        
    ### within this section is  a version that reads out where estimateParams saves to, can be used instead of manually entering resultant alpha from grad desc.
    
    #row = 0
    #kf = 0.1
    #kr = 10
    #stat = "train"
    #gamma = 1

    #saveLoc = 'data/paramEstimation/alphas_node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) +  '_params_' + 'gamma_'+ str(gamma) +  'kf_' + str(kf) + 'kr_' + str(kr) + '.npy' 
    # format: alphas_node0_dset1.train_params_gamma_1kf_0.1kr_10.npy    
    #alphas = np.load(saveLoc)
    #alpha = alpha[1]
    #print(alphas)

    ### end auto read out section

    if dset == 1:
        alpha = 22.797186679494867
    if dset == 2:
        alpha = 10.310464536404607
    if dset == 3:
        alpha = 45.705568102028074
    if dset == 4:
        alpha = 7.902930472647228
    if dset == 5:
        alpha = 14.753531979163323
    if dset == 6:
        alpha = 32.59883461997574
    if dset == 7:
        alpha = 53.3323013248404
    if dset == 8:
        alpha = 55.8358458488713
    if dset == 9:
        alpha = 21.71118311993751
    if dset == 10: 
        alpha = 10.766203812279688 
        
    return alpha



def paramValues(dset, CiF_0):
    # old parameters, used in some figures of the paper when noted as such.
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



