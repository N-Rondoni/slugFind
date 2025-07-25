import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from dataParser import pcc, medianFromDset, meanFromDset
import pandas as pd
""""
This file plots the correlations and other files produced by processing.py 

if the plot is not made here, it is made in fancyPlottingMPCmain.py

This code will likely be cleaned up a bit, while retaining the same functionality. 
"""



cors = np.load("data/allScores.npy")
cors1 = np.load("data/allScoresDset1.npy")
cors2 = np.load("data/allScoresDset2.npy")
cors3 = np.load("data/allScoresDset3.npy")
cors4 = np.load("data/allScoresDset4.npy")
cors5 = np.load("data/allScoresDset5.npy")
cors6 =  np.load("data/allScoresDset6.npy")
cors7 =  np.load("data/allScoresDset7.npy")
cors8 =  np.load("data/allScoresDset8.npy")
cors9 =  np.load("data/allScoresDset9.npy")
cors10 = np.load("data/allScoresDset10.npy")
allVPDs = np.load("data/allVPDs.npy")

stm_allVPDs = np.load("data/stm_allVPDs.npy")
oasis_allVPDs = np.load("data/oasis_allVPDs.npy")

stm_allCors = np.load("data/allScoresSTM.npy")
oasis_allCors = np.load("data/allScoresOasis.npy")

#print("stm, VPD median", np.median(stm_allVPDs))
#print("oasis, VPD median",  np.median(oasis_allVPDs))


labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","All"]
plt.figure(1)
plt.title(r"Correlation scores by data set, $n_{all} = $" + str(np.shape(cors)[0]), fontsize = 18)
plt.xlabel("Data Set", fontsize = 14)
plt.ylabel("Correlation Coefficient", fontsize = 14)
plt.boxplot([cors1, cors2, cors3, cors4, cors5, cors6, cors7, cors8, cors9, cors10, cors], labels = labels) #labels depreciated, will break in newer ver


plt.figure(3)
binVals= np.linspace(0,1,11)
stmMeanCor = np.mean(stm_allCors)                  #.4224426 # mean across test and train, found in stmProcessing.py 
oasisMeanCorInfo = np.mean(oasis_allCors)          #0.59 
oasisMeanCorNoInfo = 0.3397627                     # found in naive oasis processing folder
oasisInfoColor = "magenta"
oasisNoInfoColor = "green"
plt.hist(cors, color = 'c', edgecolor = 'k')
plt.axvline(np.mean(cors), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(stmMeanCor, color = 'r', linestyle="dashed", alpha = 0.65)
plt.axvline(oasisMeanCorInfo, color = oasisInfoColor, linestyle="dashed", alpha = 0.65)
plt.axvline(oasisMeanCorNoInfo, color = oasisNoInfoColor, linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =16)
plt.xlabel("Correlation Coefficient", fontsize = 16)
plt.title("Correlation Coefficients, All Datasets", fontsize = 20)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(cors)*.89, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(cors)))
plt.text(stmMeanCor*1.01, max_ylim*0.9, 'STM Mean: {:.2f}'.format(float(stmMeanCor)), color = 'r')
plt.text(oasisMeanCorInfo*1.01, max_ylim*0.5, 'Oasis Mean: {:.2f}'.format(float(oasisMeanCorInfo)), color = oasisInfoColor)
plt.text(oasisMeanCorNoInfo*.8, max_ylim*0.9, 'Oasis* Mean: {:.2f}'.format(float(oasisMeanCorNoInfo)), color = oasisNoInfoColor)
#print("mean", np.mean(cors))

# plotting VP distances
#print("# of allVPDs", np.shape(allVPDs))
#print("# of all VPDs less than 3500", np.shape(allVPDs[allVPDs<=3500]))
naive_oasisVPDmedian = 901.39 
stmVPDmedian = np.median(stm_allVPDs)
oasisVPDmedian = np.median(oasis_allVPDs)
plt.figure(4)
plt.hist(allVPDs[allVPDs<=3500], color='coral', edgecolor="k",bins=20)
plt.axvline(oasisVPDmedian, color=oasisInfoColor, linestyle="dashed", alpha = 0.65, label="NND")
plt.axvline(np.median(allVPDs), color='k', linestyle="dashed", alpha = 0.65, label="MPC")
plt.axvline(stmVPDmedian, color='r', linestyle="dashed", alpha = 0.65, label="STM")
plt.axvline(naive_oasisVPDmedian, color=oasisNoInfoColor, linestyle="dashed", alpha = 0.65, label = "NND*")
plt.ylabel("Count", fontsize =16)
plt.xlabel("Victor-Purpura Distance", fontsize = 16)
plt.title("Victor-Purpura Distances", fontsize = 20)
min_ylim, max_ylim = plt.ylim()
plt.legend()
#plt.text(np.median(allVPDs)*(-.1), max_ylim*0.96, 'Median: {:.2f}'.format(np.median(allVPDs)))
#plt.text(oasisVPDmedian*1.1, max_ylim*0.5, 'Oasis* Median: {:.2f}'.format(oasisVPDmedian), color=oasisNoInfoColor)
#plt.text(stmVPDmedian*1.1, max_ylim*0.7, 'STM Median: {:.2f}'.format(stmVPDmedian), color='r')


plt.figure(6)
plt.hist(allVPDs[allVPDs <= 2000])
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distances for datasets less than 2000", fontsize = 18)

file_path1 = 'data/results_22_06_17.csv'
file_path2 = 'data/spikefinder.csv'
df = pd.read_csv(file_path1)
stm_df = df[(df['algo'] == 'stm')]
#print("stm average on test data:", stm_df['value'].mean()) # used to place mean line in earlier plot


# This is an alternative (and maybe more intuitive way) to pull df for partiular method 
# ends up functionally equivalent to what is done with function pcc
df2 = pd.read_csv(file_path2)
#df2 = df2.reset_index(drop=True)
# Filter for algo_algorithm == 'stm' and label_label == 'corr'
filtered_df = df2[(df2['algo_algorithm'] == 'stm') & (df2['label_label'] == 'corr')]
# Compute mean of label_value
#print("stm average on test and train data, all csv:", filtered_df['label_value'].mean())
#print("stm median on test and train data, all csv:", filtered_df['label_value'].median())
#print("filtered_DF", filtered_df)


oasis_df =df2[(df2['algo_algorithm'] == 'linear gaussian') & (df2['label_label'] == 'corr')] # just means on each dataset
#print(oasis_df)
#print(oasis_df['label_value'].mean())
#nonan = oasis_df['label_value']
#print(np.shape(nonan))
#nonan = nonan[~np.isnan(nonan)]
#print("oasis mean", nonan.mean())

stm_df = pcc(file_path2, 'stm') # pulls all data, test and train
#print("stm_df", stm_df)
# can also do the above mean, median calls with the below logic. 
#j = 0
#runSum = 0
#accumed = []
#for key in stm_df:
#    #print(key)
#    temp = stm_df[key]
#    res = float(temp[0])
#    accumed = np.append(accumed, res)
#    runSum = runSum + res
#   j = j + 1

#print("average for stm across all datasets:", runSum/j)
#print("median across all for stm:", np.median(accumed))
#print("stm dset 7 mean:", float(stm_df['7.train.spikes'][0]))
#print("stm dset 3 mean:", float(stm_df['3.test.spikes'][0]))


#print("filtered df min", filtered_df)
#print("dset 5 min", np.min(stm_df['9.train.spikes']))
#print("AAAAAAAAA IS GOD REAL", stm_df['9.train.spikes'])



meanS1 = meanFromDset(df, 'stm', 1)
meanS2 = meanFromDset(df, 'stm', 2)
meanS3 = meanFromDset(df, 'stm', 3)
meanS4 = meanFromDset(df, 'stm', 4)
meanS5 = meanFromDset(df, 'stm', 5)

# this meanFromDset function will break there isn't both train and test (like dsets 7, 8, 9). 
# for datasets with just train, will take mean directly from df. 

figNo = 8
plt.figure(figNo)
simDat = cors1
methodsMean = meanS1
dset = 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Median: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Median: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors2
methodsMean = meanS2
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*0.7, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.05, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors3
methodsMean = meanS3
dset = dset + 1
dset = 3
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*0.6, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat))) #fontsize = 14) too big
plt.text(methodsMean*1.05, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r') #fontsize = 14)
figNo = figNo + 1

plt.figure(figNo)
simDat = cors4
methodsMean = meanS4
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors5
methodsMean = meanS5
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1


plt.figure(figNo)
simDat = cors7
methodsMean = float(stm_df['7.train.spikes'][0])
dset = 7
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.05, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*.76, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors8
methodsMean = float(stm_df['8.train.spikes'][0])
dset = 8
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.05, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.05, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1


# same as figure that compares all 4 means, but attempting to include a legend instead
plt.figure(figNo)
binVals= np.linspace(0,1,11)
#stmMeanCor = 0.3866859940274038 # found with computations below, only on test data. 
#stmMeanCor = .4244426
#oasisMeanCorInfo = 0.45 #$ from jneursci stringer paper
#oasisMeanCorNoInfo = 0.3397627 
oasisInfoColor = "magenta"#"hotpink"#"darkorchid"#"hotpink"
oasisNoInfoColor = "green"#"Seagreen"#"indigo"#"seagreen"
plt.hist(cors, color = 'c', edgecolor = 'k')#, bins = binVals)
plt.axvline(oasisMeanCorNoInfo, color = oasisNoInfoColor, linestyle="dashed", alpha = 0.65, label = "NND*")
plt.axvline(np.mean(cors), color='k', linestyle="dashed", alpha = 0.65, label="MPC")
plt.axvline(stmMeanCor, color = 'r', linestyle="dashed", alpha = 0.65, label="STM")
plt.axvline(oasisMeanCorInfo, color = oasisInfoColor, linestyle="dashed", alpha = 0.65, label = "NND")
plt.legend()
plt.ylabel("Count", fontsize =16)
plt.xlabel("Correlation Coefficient", fontsize = 16)
plt.title("Correlation Coefficients, All Datasets", fontsize = 20)
min_ylim, max_ylim = plt.ylim()
figNo = figNo + 1


# violin plot with color median
fig, axs = plt.subplots()
parts = axs.violinplot([cors1, cors2, cors3, cors4, cors5, cors6, cors7, cors8, cors9, cors10, cors], showmeans=False, showmedians=True)
parts['cmedians'].set_color('deeppink')
axs.set_title("Violin Plot of Correlation Coefficient by Dataset", fontsize = 18)
axs.set_xticks([y + 1 for y in range(len([cors1, cors2, cors3, cors4, cors5, cors6, cors7, cors8, cors9, cors10, cors]))], labels=labels)
axs.set_ylabel("Correlation Coefficient", fontsize = 14)
axs.set_xlabel("Dataset", fontsize = 14)
axs.yaxis.grid(True)
#fig.suptitle(r"Correlation Scores by Data Set", fontsize = 18)
plt.tight_layout()
figNo = figNo + 1


plt.show()

