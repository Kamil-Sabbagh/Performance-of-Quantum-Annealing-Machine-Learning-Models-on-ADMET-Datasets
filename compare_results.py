# Let's generate the CSV file with the combined data and perform a statistical analysis to check the relationships.

import pandas as pd
import os
import numpy as np
from scipy import stats

# Step 1: Load the data
previous_benchmarks = pd.read_csv("previous_benchmarks.csv")
qboost_results = pd.read_csv("Qboost_results.csv")
qsvm_results = pd.read_csv("QSVM_results2.csv")

# Step 2: Clean the data
previous_benchmarks['dataset_name'] = previous_benchmarks['dataset_path'].apply(lambda x: x.split('/')[-1])
qboost_results['dataset_name'] = qboost_results['dataset_path'].apply(lambda x: x.split('/')[-1])
qsvm_results['dataset_name'] = qsvm_results['dataset_path'].apply(lambda x: x.split('/')[-1])

# Step 3: Filter datasets that are common across all three methods
common_datasets = set(previous_benchmarks['dataset_name']).intersection(
    qboost_results['dataset_name'],
    qsvm_results['dataset_name']
)

# Filter data to only include these common datasets
previous_benchmarks = previous_benchmarks[previous_benchmarks['dataset_name'].isin(common_datasets)]
qboost_results = qboost_results[qboost_results['dataset_name'].isin(common_datasets)]
qsvm_results = qsvm_results[qsvm_results['dataset_name'].isin(common_datasets)]

# Step 4: Merge the data on 'dataset_name'
merged_data = pd.merge(
    pd.merge(previous_benchmarks[['dataset_name', 'auc_roc']], qboost_results[['dataset_name', 'auc_roc']], on='dataset_name', suffixes=('_prev', '_qboost')),
    qsvm_results[['dataset_name', 'auc_roc']], on='dataset_name'
)

merged_data.rename(columns={'auc_roc': 'auc_qsvm'}, inplace=True)

# Step 5: Read original training dataset statistics (train.csv) without balancing
num_train_examples = []
num_positive_cases = []
num_negative_cases = []
difference_cases = []

for dataset in merged_data['dataset_name']:
    dataset_folder = f"alldatasets/{dataset}"  # Assumes datasets are stored in 'alldatasets/{dataset_name}'
    
    if os.path.exists(dataset_folder):
        train_file = os.path.join(dataset_folder, 'train.csv')
        
        if os.path.exists(train_file):
            # Load the training dataset
            train_df = pd.read_csv(train_file)
            
            # Count the number of examples in the training set
            total_train_examples = len(train_df)
            
            # Count positive and negative cases in the training set (assuming the column 'Y' holds the class labels)
            total_positive_cases = np.sum(train_df['Y'] == 1)
            total_negative_cases = np.sum(train_df['Y'] == 0)
            
            # Calculate the difference between positive and negative cases
            difference = total_positive_cases - total_negative_cases
            
            num_train_examples.append(total_train_examples)
            num_positive_cases.append(total_positive_cases)
            num_negative_cases.append(total_negative_cases)
            difference_cases.append(difference)
        else:
            num_train_examples.append(0)
            num_positive_cases.append(0)
            num_negative_cases.append(0)
            difference_cases.append(0)
    else:
        num_train_examples.append(0)
        num_positive_cases.append(0)
        num_negative_cases.append(0)
        difference_cases.append(0)

# Add the dataset statistics to the merged_data dataframe
merged_data['num_train_examples'] = num_train_examples
merged_data['num_positive_cases'] = num_positive_cases
merged_data['num_negative_cases'] = num_negative_cases
merged_data['difference_cases'] = difference_cases

# Save the merged data to a CSV file
csv_file_path = 'merged_auc_and_dataset_stats.csv'
merged_data.to_csv(csv_file_path, index=False)

# Step 6: Statistical Analysis
# Calculate the correlation between dataset statistics and AUC-ROC values for each method
correlation_results = {
    'Method': ['Previous Benchmarks', 'QBoost', 'QSVM'],
    'Train Size Correlation': [
        stats.pearsonr(merged_data['num_train_examples'], merged_data['auc_roc_prev'])[0],
        stats.pearsonr(merged_data['num_train_examples'], merged_data['auc_roc_qboost'])[0],
        stats.pearsonr(merged_data['num_train_examples'], merged_data['auc_qsvm'])[0]
    ],
    'Positive Cases Correlation': [
        stats.pearsonr(merged_data['num_positive_cases'], merged_data['auc_roc_prev'])[0],
        stats.pearsonr(merged_data['num_positive_cases'], merged_data['auc_roc_qboost'])[0],
        stats.pearsonr(merged_data['num_positive_cases'], merged_data['auc_qsvm'])[0]
    ],
    'Negative Cases Correlation': [
        stats.pearsonr(merged_data['num_negative_cases'], merged_data['auc_roc_prev'])[0],
        stats.pearsonr(merged_data['num_negative_cases'], merged_data['auc_roc_qboost'])[0],
        stats.pearsonr(merged_data['num_negative_cases'], merged_data['auc_qsvm'])[0]
    ],
    'Difference (Pos-Neg) Correlation': [
        stats.pearsonr(merged_data['difference_cases'], merged_data['auc_roc_prev'])[0],
        stats.pearsonr(merged_data['difference_cases'], merged_data['auc_roc_qboost'])[0],
        stats.pearsonr(merged_data['difference_cases'], merged_data['auc_qsvm'])[0]
    ]
}

# Convert the correlation results to a DataFrame
correlation_df = pd.DataFrame(correlation_results)

# Display the correlation results
import matplotlib.pyplot as plt

# Step 7: Save the correlation results as a CSV file
correlation_csv_file_path = 'correlation_results.csv'
correlation_df.to_csv(correlation_csv_file_path, index=False)

# Step 8: Create the AUC-ROC plot and save as PDF
plt.figure(figsize=(16, 5))
bar_width = 0.25
r1 = range(len(merged_data))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Creating bar plots for AUC scores
plt.bar(r1, merged_data['auc_roc_prev'], color='b', width=bar_width, edgecolor='grey', label='Previous Benchmarks AUC')
plt.bar(r2, merged_data['auc_roc_qboost'], color='g', width=bar_width, edgecolor='grey', label='QBoost AUC')
plt.bar(r3, merged_data['auc_qsvm'], color='r', width=bar_width, edgecolor='grey', label='QSVM AUC')

# Adding labels and title
plt.xlabel('Dataset Name', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(merged_data))], merged_data['dataset_name'], rotation=90)
plt.ylabel('AUC-ROC Score', fontweight='bold')
plt.title('Comparison of AUC-ROC Scores across Methods')
plt.legend()

# Save the plot as a PDF
auc_roc_pdf_path = 'auc_roc_plot.pdf'
plt.tight_layout()
plt.savefig(auc_roc_pdf_path)

# Step 9: Create the Dataset Statistics plot and save as PDF
plt.figure(figsize=(16, 6))
r1 = range(len(merged_data))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]

# Creating bar plots for dataset statistics
plt.bar(r1, merged_data['num_train_examples'], color='c', width=bar_width, edgecolor='grey', label='Total Train Examples')
plt.bar(r2, merged_data['num_positive_cases'], color='m', width=bar_width, edgecolor='grey', label='Positive Cases')
plt.bar(r3, merged_data['num_negative_cases'], color='y', width=bar_width, edgecolor='grey', label='Negative Cases')
plt.bar(r4, merged_data['difference_cases'], color='orange', width=bar_width, edgecolor='grey', label='Difference (Positive-Negative)')

# Adding labels and title
plt.xlabel('Dataset Name', fontweight='bold')
plt.xticks([r + 1.5 * bar_width for r in range(len(merged_data))], merged_data['dataset_name'], rotation=90)
plt.ylabel('Number of Cases', fontweight='bold')
plt.title('Training Dataset Statistics: Total, Positive, Negative, and Difference')
plt.legend()

# Save the plot as a PDF
dataset_stats_pdf_path = 'dataset_stats_plot.pdf'
plt.tight_layout()
plt.savefig(dataset_stats_pdf_path)