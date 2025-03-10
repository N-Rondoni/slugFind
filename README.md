slugFind, or MPC approach as it is referred to in its accompanying manuscript, infers neuronal spiking as a function of calcium imaging data.
 
 Corresponding Author: Nicholas A. Rondoni
         nrondoni@ucsc.edu 
         
 Please reach out if you have any questions about this software implementation! 


 This directory contains the major software contributions of the paper entitled 
 "Predicting Neuronal Firing from Calcium Imaging Using a Control Theoretic Approach".

#--------------------------------------------------------------------------------------------#
 
 In summary, the ground up workflow should be
 
 -dataPrep.py
 
 -can run: MPCmain.py 0 1 "test" from command line for specific node/dataset/condition. In this example, node 0 dset 1 condition "test".
 
 Alternatively, driver.py will iterate through all nodes/datasets/conditions
 
 -processing.py computes and saves all correlation scores, as well as VPD distances
 
 -plotFriend.py visualizes the output of processing.py 
 
 -for visualizations of spiking as a function of calcium data on a subset, use fancyPlottingMPCmain.py
 
#--------------------------------------------------------------------------------------------#

 All raw .csv files, e.g., 1.test.calcium.csv,  were provided from the spikefinder challenge and can be found within the subfolder "data"
 
 We did not in any way collect this source data, and credit is due to Berens et. al.'s paper in Plos Computational Biology entitled
 
 "Community-based benchmarking improves spike rate inference from two-photon calcium imaging data", found at https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006157#sec014
 
 for compiling and providing it. 

#--------------------------------------------------------------------------------------------#

 BELOW PROVIDES MORE ROBUST DETAILS AND DESCRIBES DATA ORGANIZATION STRUCTURE, IF NEEDED: 

 Within the folder "data" is a subfolder named "processed" (path data/processed)
 
 This folder contains all rows of each .csv, such as 1.test.calcium.csv, saved into its own file. This is accomplished by the file dataPrep.py, 
 
 which will produce nodeX_dset1.test.calcium.npy and nodeX_dset1.test.spikes.npy
 
 dataPrep.py should be ran first if beginning with raw csv. 
 
 For example, dataPrep.py will save into this "processed" folder "node0_dset1.test.calcium.npy", denoting the calcium trace for the 0th row of dset 1, test. 
 
 dataPrep cuts off datasets after the first NaN appears, which is OK for the spikefinder challenge since after the first NaN the rest won't be numbers either.
 

 with "processed" populated, MPCmain.py will perform the necessary computations to infer underlying spiking times, 
 
 then print the correlation score when comparing to the ground truth data, which can be found in "data" with names such as  1.test.spikes.csv 
 
 for future analysis, these inferred spikes are saved into the folder "data/processed/solutions" as files named in the format node0_dset1.test.sVals.npy
 
 driver.py can be used to loop through all neurons, datasets, and train/test conditions. 

 the following codes examine the output of MPCmain, plotting it and comparing the output to other state of the art methods. 
 
 At this point, processing.py should be ran. This can take a while due to the computational cost of computing VP distances.
 
 If speed is desired, I suggest commenting out the lines where VP distance is computed  (lines 163 - 170).
 
 processing.py will save all correlation scores within the data folder. 

 For visualization, run plotFriend.py, which will generate most the figures used in the paper. 
 
 For visulaization of specific subsets of spiking activity, run fancyPlottingMPCmain.py 
 
 fancyPlottingMPCmain.py is exactly MPCmain.py, but produces fancy plots which would slow down driver.py if looped over. 
