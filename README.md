slugFind, or MPC approach as it is referred to in its accompanying manuscript, infers neuronal spiking as a function of calcium imaging data.
 
 Corresponding Author: Nicholas A. Rondoni
         nrondoni@ucsc.edu 
         
 Please reach out if you have any questions about this software implementation! 


 This directory contains the major software contributions of the paper entitled 
 "Predicting Neuronal Firing from Calcium Imaging Using a Control Theoretic Approach"
 published through PLOS at https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012603

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
 
 In summary, the ground up workflow should be
 
 -dataPrep.py

 -estimateDriver.py will iterate through available datasets in the spikefinder challenge, or one may point estimateParams.py at their specific file.

 -make sure your learned alpha is stored in paramUtils.py (either hardcode or point to the location estimateParams saved alpha at). 
 
 -can run: MPCmain.py 0 1 "test" from command line for specific node/dataset/condition. In this example, node 0 dset 1 condition "test".

 Alternatively, driver.py will iterate through all nodes/datasets/conditions

 -can also point MPCmain.py to your own calcium recording, should you have such data
 
 -processing.py computes and saves all correlation scores, as well as VPD distances
 
 -plotFriend.py visualizes the output of processing.py 
 
 -for visualizations of spiking as a function of calcium data on a subset, use fancyPlottingMPCmain.py
 
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

 All raw .csv files, e.g., 1.test.calcium.csv,  were provided from the spikefinder challenge and can be found within the subfolder "data"
 
 We did not in any way collect this source data, and credit is due to Berens et. al.'s paper in Plos Computational Biology entitled

 "Community-based benchmarking improves spike rate inference from two-photon calcium imaging data", 
 found at https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006157
 
 for compiling and providing it. 

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
FURTHER DETAILS AND DATA ORGANIZATION

Within the folder "data" is a subfolder named "processed" (path data/processed). This folder
contains all rows of each .csv (e.g., 1.test.calcium.csv) saved into its own file. This is
accomplished by dataPrep.py, which produces files like nodeX_dset1.test.calcium.npy and
nodeX_dset1.test.spikes.npy. dataPrep.py should be run first if beginning with raw csv. For
example, it will save "node0_dset1.test.calcium.npy" into the "processed" folder, denoting the
calcium trace for the 0th row of dset 1, test. dataPrep cuts off datasets after the first NaN
appears, which is OK for the Spikefinder challenge since after the first NaN the rest won't be
numbers either.

With "processed" populated, MPCmain.py will infer underlying spiking times and print the
correlation score when comparing to the ground truth data (found in "data" with names such as
1.test.spikes.csv). For future analysis, inferred spikes are saved into "data/processed/solutions"
as files named in the format node0_dset1.test.sVals.npy. driver.py can be used to loop through
all neurons, datasets, and train/test conditions.

The following codes examine the output of MPCmain, comparing the output to other state-of-the-art
methods. Run processing.py to compute and save all correlation scores within the data folder. This
can take a while due to the computational cost of computing VP distances - if speed is desired,
comment out lines 163-170. For visualization, run plotFriend.py, which will generate most of the
figures used in the paper. For visualization of specific subsets of spiking activity as a function
of calcium data, run fancyPlottingMPCmain.py. It is exactly MPCmain.py, but produces fancy plots
which would slow down driver.py if looped over.
