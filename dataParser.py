import csv
from collections import defaultdict
"""
 CSV processing tools, reports correlation coefficeints and their mean/median values
 Computes these medians, means from spikefinder challenge in plotFriend.py 
"""

# parse correlation coeff, pcc
def pcc(file_path, algorithm_name):
    corr_data = defaultdict(list) 
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            if row['algo_algorithm'] == algorithm_name and row['label_label'] == 'corr':
                dataset_name = row['dset_dataset']  # Use the dataset name instead of the ID
                correlation_value = float(row['label_value'])
                corr_data[dataset_name].append(correlation_value)
    
    return dict(corr_data)


# this is good
def medianFromDset(df, algorithm, dset):
    # Filter the DataFrame based on the algorithm and dataset
    filtered_df = df[(df['algo'] == algorithm) & (df['dset'] == dset)]
    
    # Compute the median of the 'value' column for both 'train' and 'test' data
    overall_med = filtered_df['value'].median()
    
    return overall_med


def meanFromDset(df, algorithm, dset):
    # Filter the DataFrame based on the algorithm and dataset
    filtered_df = df[(df['algo'] == algorithm) & (df['dset'] == dset)]
    
    # Compute the mean of the 'value' column for both 'train' and 'test' data 
    overall_med = filtered_df['value'].mean()
    
    return overall_med 





