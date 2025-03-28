import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time
import do_mpc
from casadi import *
import pandas as pd
#import seaborn as sns
from datetime import date
import spikefinder_eval as se
from spikefinder_eval import _downsample
from VPdistance import VPdis
from scipy.stats import spearmanr

save = True#False
ftype = "png"

def plotCorrelations(factors, corrCoefs, neuron, dset):
    plt.figure()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting with enhancements
    ax1.plot(factors, corrCoefs, marker='o', linestyle='-', color='b', markersize=8, linewidth=2, label='Correlation Coefficients')
    
    # Primary x-axis and y-axis labels
    ax1.set_title("Correlations between Simulated and Recorded Spikes", fontsize=20, fontweight='bold')
    ax1.set_xlabel("Downsampling Factor", fontsize=16)
    ax1.set_ylabel("Correlation Coefficient", fontsize=16)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adding annotations for data points
    for i, txt in enumerate(corrCoefs):
        ax1.annotate(f'{txt:.2f}', (factors[i], corrCoefs[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
    ax1.legend(fontsize=14)

    # Secondary x-axis for bin width
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Make sure the limits match
    bin_widths = [40, 83.0, 167.0, 333.0]  # Specific bin width values
    ax2.set_xticks(factors)
    ax2.set_xticklabels([f'{bw:.1f}' for bw in bin_widths])
    ax2.set_xlabel("Bin width (ms)", fontsize=20)
   
    fig.tight_layout()
    plt.show()

    print('dset:', dset, 'neuron:', neuron, "corr:", corrCoefs[0])
    if save == True:
        filename = 'CorrCoef_dset'+ str(dset) + "_neuron" + str(neuron)
        plt.savefig(filename + '.' + ftype, format = ftype)
                    

def plotSignalsSubset(t, simSignal, trueSignal, sStart, sStop, neuron, dset):
    plt.figure()
    plt.plot(t[sStart:sStop], simSignal[sStart:sStop], label=r'Simulated Rate')
    plt.plot(t[sStart:sStop], trueSignal[sStart:sStop], label="Recorded Spike Rate", alpha = 0.8)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.title("Subset of Expected and Recorded Spikes, dataset " + str(dset) + " neuron " + str(neuron))
    plt.legend()
    
    if save == True:
        filename = 'Spikes_subset_dset'+ str(dset) + "_neuron" + str(neuron)
        plt.savefig(filename + '.' + ftype, format = ftype)
        
def plotSignals(t, simSignal, trueSignal, neuron, dset, stat):
    plt.figure()
    plt.plot(t, simSignal, label=r'Simulated Rate')
    plt.plot(t, trueSignal, label="Recorded Spike Rate", alpha = 0.8)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.title("Expected and Recorded Spikes")#, bin size of " + str(1000*binSizeTime) + " ms")
    plt.legend()

    if save == True:
        filename = 'Spikes_fullSolve_dset'+ str(dset) + "_neuron" + str(neuron)
        plt.savefig(filename + '.' + ftype, format = ftype)
        os.system("mv "+ filename + "." + ftype + " /Users/nrondoni/Pictures/spikeFinder/" + stat) # only use this photo location if you are Nick  

def NaNChecker(dset, row, stat):
    # check for NaNs in calcium dataset
    file_path = 'data/' + str(dset) + '.' + str(stat) + '.calcium.csv'
    #file_path2 = 'data/' + str(dset) + '.' + str(stat) + '.spikes.csv'

    data1 = pd.read_csv(file_path).T 
    data1 = np.array(data1)
    mDat, nDat = np.shape(data1)
    # loop through rows, each corresponds to a neuron. 
    
    # check for NaNs
    naninds = np.isnan(data1[row,:])
    #if the below evaluates to true, there are NaNs in the dataset.
    NaNpresent = np.any(naninds)

    return NaNpresent 


def saturatedChecker(signal):
    endLen = 10
    last = signal[-endLen:-1]
    last = last/np.max(last)
    tracker = 0
    #print(last)
    for ii in range(len(last)-1):
        dif = np.abs(last[ii] - last[ii+1])
        if dif < 0.001:
            tracker  = tracker + 1
            #print("tracker  hit!", counter)        
    if tracker >= int(0.80*endLen):
        print("probably an issue here")
    #print(tracker)

if __name__=="__main__":
   
    # load in actual truth data
    states = ['test', 'train']
    tempSum = 0
    counter = 0
    downsampledCorScor = []
    downsampledCorScor1 = []
    downsampledCorScor2 = []
    downsampledCorScor3 = []
    downsampledCorScor4 = []
    downsampledCorScor5 = []
    downsampledCorScor6 = []
    downsampledCorScor7 = []
    downsampledCorScor8 = []
    downsampledCorScor9 = []
    downsampledCorScor10 = []
    
    downsampledCorScorTest = []
    downsampledCorScor1Test = []
    downsampledCorScor2Test = []
    downsampledCorScor3Test = []
    downsampledCorScor4Test = []
    downsampledCorScor5Test = []
    
    allVPDs = []

    for stat in states:
       #dsets = [1, 2, 3, 4, 5]
        dsets = [1, 2, 3, 4, 5]
        if stat == "train":
            #dsets = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            dsets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for dset in dsets:
            # load in true spikes
            file_path2 = 'data/' + str(dset) + '.' + str(stat) + '.spikes.csv' # load in undownsampled data just to know number of rows.
            spikeDat = pd.read_csv(file_path2).T 
            spikeDat = np.array(spikeDat)

            mSpike,nSpike = spikeDat.shape
            # set imrate depending on dset
                        
            imRate = 1/100
                    
            i = 0
            while i < mSpike:
                spikeDatRaw = np.load('data/processed/node' + str(i) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy')
                simSpikesRaw = np.load('data/processed/solutions/node' + str(i) + '_dset' + str(dset) + '.' + str(stat) + '.sVals.npy')
            
                nSpike = np.shape(spikeDatRaw)[0]
                n = np.shape(simSpikesRaw)[0]
                

                finalTime = n*(imRate)
                               

                # create corr coeff
                factors = [4] # can add more factors to this list if you'd like to see other downsampled values. Be careful!
    
                corrCoefs = np.zeros(np.shape(factors))
                VPDs = np.zeros(np.shape(factors))
                for j in range(len(factors)):
                    factor = factors[j]
                    spikeDatDown = _downsample(spikeDatRaw, factor)
                    simSpikeDown = _downsample(simSpikesRaw, factor)
                    corrCoefs[j] = np.corrcoef(spikeDatDown, simSpikeDown)[0, 1] 
                    print("pearson", corrCoefs[j])
                    print("spearman", spearmanr(spikeDatDown, simSpikeDown).statistic)
                    corrCoefs[j] = spearmanr(spikeDatDown, simSpikeDown).statistic

               
    
                    print('dset:', dset, 'neuron:', i, "corr:", corrCoefs[0])
            
                    saturatedChecker(simSpikeDown)

                    # split Victur-Purpura computations into two (can run on subsets then add results, getting same score). VPD(a + b) = VPD(a) + VPD(b)
                    # this must be done for certain data sets if you have less than 16GB ram.    
                    Nreduced = int(len(spikeDatDown)/2)
                    VPDtemp1 = 0#VPdis(spikeDatDown[0:Nreduced], simSpikeDown[0:Nreduced], 1) 
                    VPDtemp2 = 0#VPdis(spikeDatDown[Nreduced:-1], simSpikeDown[Nreduced:-1], 1) 
                    sumVPD = VPDtemp1 + VPDtemp2
                    VPDs[j] = sumVPD                    
                
                    if j == 0:
                        tempSum = tempSum + corrCoefs[0]
                        counter = counter + 1   

                # set up time to match, note final time is still computed with undownsampled n. Only use this time Vec for testing to be safe.
                n1 = min([len(spikeDatDown), len(simSpikeDown)])
                t_down = np.linspace(0, finalTime, n1)
                timeVec = np.linspace(0, finalTime, n)
                neuron = i
               
               
                # finally call plot functions
                #plotCorrelations(factors, corrCoefs, neuron, dset)
                downsampledCorScor = np.append(downsampledCorScor, corrCoefs[0])
                allVPDs = np.append(allVPDs, VPDs[0])
                print("Victor-Purpura Distance:", VPDs[0])
                if dset == 1:
                    if stat == "train":
                        downsampledCorScor1 = np.append(downsampledCorScor1, corrCoefs[0])
                    elif stat == "test":
                        downsampledCorScor1Test = np.append(downsampledCorScor1Test, corrCoefs[0])
                if dset == 2:
                    if stat == "train":
                        downsampledCorScor2 = np.append(downsampledCorScor2, corrCoefs[0])
                    elif stat == "test":
                        downsampledCorScor2Test = np.append(downsampledCorScor2Test, corrCoefs[0])
                if dset == 3:
                    if stat == "train":
                        downsampledCorScor3 = np.append(downsampledCorScor3, corrCoefs[0])
                    elif stat == "test":
                        downsampledCorScor3Test = np.append(downsampledCorScor3Test, corrCoefs[0])
                if dset == 4:
                    if stat == "train":
                        downsampledCorScor4 = np.append(downsampledCorScor4, corrCoefs[0])
                    elif stat == "test":
                        downsampledCorScor4Test = np.append(downsampledCorScor4Test, corrCoefs[0])
                if dset == 5:
                    if stat == "train":
                        downsampledCorScor5 = np.append(downsampledCorScor5, corrCoefs[0])
                    elif stat == "test":
                        downsampledCorScor5Test = np.append(downsampledCorScor5Test, corrCoefs[0])
                if dset == 6:
                    if stat == "train":
                        downsampledCorScor6 = np.append(downsampledCorScor6, corrCoefs[0])
                if dset == 7:
                    if stat == "train":
                        downsampledCorScor7 = np.append(downsampledCorScor7, corrCoefs[0])
                if dset == 8:
                    if stat == "train":
                        downsampledCorScor8 = np.append(downsampledCorScor8, corrCoefs[0])
                if dset == 9:
                    if stat == "train":
                        downsampledCorScor9 = np.append(downsampledCorScor9, corrCoefs[0])
                if dset == 10:
                    if stat == "train":
                        downsampledCorScor10 = np.append(downsampledCorScor10, corrCoefs[0])
                    



                # instead of mixing training/testing data since they're not very different for this
                # method, keep them distinct. Allows for better comparisons to other methods. 
                #if dset == 2:
                #    downsampledCorScor2 = np.append(downsampledCorScor2, corrCoefs[0])
                #if dset == 3:
                #    downsampledCorScor3 = np.append(downsampledCorScor3, corrCoefs[0])
                #if dset == 4:
                #    downsampledCorScor4 = np.append(downsampledCorScor4, corrCoefs[0])
                #if dset == 5:
                #    downsampledCorScor5 = np.append(downsampledCorScor5, corrCoefs[0])
                #if dset == 6:
                #    downsampledCorScor6 = np.append(downsampledCorScor6, corrCoefs[0])
                #if dset == 7: 
                #downsampledCorScor7 = np.append(downsampledCorScor7, corrCoefs[0])
                #if dset == 8:
                #    downsampledCorScor8 = np.append(downsampledCorScor8, corrCoefs[0])
                #if dset == 9:
                #    downsampledCorScor9 = np.append(downsampledCorScor9, corrCoefs[0])
                #if dset == 10:
                #    downsampledCorScor10 = np.append(downsampledCorScor10, corrCoefs[0])


                #plotSignals(t_down[50:], simSpikeDown[50:], spikeDatDown[50:], neuron, dset) # THESE ARE DOWNSAMPLES VALUES
                #subStart, subStop = 200, 400
                #plotSignalsSubset(t_f, simSpikeDown, spikeDatDown, subStart, subStop, neuron, dset) # UNCOMMENT TO PLOT DOWNSAMPLED VALUES
                subStart, subStop = 2000, 4000
                
                plotSignals(timeVec, simSpikesRaw, spikeDatRaw, neuron, dset, stat)
                

                #plotSignalsSubset(timeVec, simSpikesRaw, spikeDatRaw, subStart, subStop, neuron, dset)

                #print(np.shape(t_f[subStart:subStop]), np.shape(simSpikeDown[subStart:subStop]), np.shape(spikeDatDown[subStart:subStop]))
                #print(np.shape(t_f), np.shape(simSpikeDown), np.shape(spikeDatDown))

                i = i + 1 

        print("average:", tempSum/counter)


    avg1 = np.mean(downsampledCorScor1)
    avg2 = np.mean(downsampledCorScor2)
    avg3 = np.mean(downsampledCorScor3)
    avg4 = np.mean(downsampledCorScor4)
    avg5 = np.mean(downsampledCorScor5)
    #
    avg1Test = np.mean(downsampledCorScor1Test)
    avg2Test = np.mean(downsampledCorScor2Test)
    avg3Test = np.mean(downsampledCorScor3Test)
    avg4Test = np.mean(downsampledCorScor4Test)
    avg5Test = np.mean(downsampledCorScor5Test)

    #
    avg6 = np.mean(downsampledCorScor6)
    avg7 = np.mean(downsampledCorScor7)
    avg8 = np.mean(downsampledCorScor8)
    avg9 = np.mean(downsampledCorScor9)
    avg10 = np.mean(downsampledCorScor10)
    averages = [avg1Test, avg2Test, avg3Test, avg4Test, avg5Test, avg1, avg2, avg3, avg4, avg5, avg6, avg7, avg8, avg9, avg10]
    print(averages)

    print("mean of means", np.mean(averages))
    print("median of means", np.median(averages))
    print("All cors:", downsampledCorScor)
    print("Median of whole set:", np.median(downsampledCorScor))
    print("Mean of whole set:", np.mean(downsampledCorScor))
    #print("All VP distances:", allVPDs)
    #np.save("data/allVPDs", allVPDs)
    #np.save("data/allScores", downsampledCorScor)
    #np.save("data/allScoresDset1", downsampledCorScor1)
    #np.save("data/allScoresDset2", downsampledCorScor2)
    #np.save("data/allScoresDset3", downsampledCorScor3)
    #np.save("data/allScoresDset4", downsampledCorScor4)
    #np.save("data/allScoresDset5", downsampledCorScor5)
    #np.save("data/allScoresDset6", downsampledCorScor6)
    #np.save("data/allScoresDset7", downsampledCorScor7)
    #np.save("data/allScoresDset8", downsampledCorScor8)
    #np.save("data/allScoresDset9", downsampledCorScor9)
    #np.save("data/allScoresDset10", downsampledCorScor10)

    #print(allVPDs)
    #plt.show()

