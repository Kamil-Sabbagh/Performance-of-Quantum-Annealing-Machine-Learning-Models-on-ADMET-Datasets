import neal
import math
from dwave.system import LeapHybridSampler
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import DataLoader
import utils
import argparse
from CoustumDataLoader import load_data_from_folder
def parseArguments():
    parser = argparse.ArgumentParser(description="Change Config File")
    
    parser.add_argument("--type",default = 'SA',choices = ['SA','HQPU','QPU'],help = 'Sampler Type')
    parser.add_argument("--dataFile",default= 'banknote_1',help = 'Dataset File')
    parser.add_argument("--trainingPoints",type = int,default= 20,help = 'Training Points')
    parser.add_argument("--validationPoints",type = int,default= 10,help = 'validation Points')
    parser.add_argument("--C",type = int,default= 3,help = 'Regularisation Parameter')
    parser.add_argument("--xi",type = int,default= 0.001,help = 'QUBO Penalty Parameter')
    parser.add_argument("--gamma",type = int,default= 16,help = 'Rbf Gamma Parameter')
    parser.add_argument("--B",type = int,default= 2,help = 'Base for encoding Variable')
    parser.add_argument("--K",type = int,default= 2,help = 'Number of variables used for encoding')
    parser.add_argument('--fig',default=False,action='store_true',help = "Plot Figure")
    args = parser.parse_args()
    args_config = vars(args)
    print(args_config)
    return args_config

def save_results_to_csv(file_name, row):
    """Save a single row of results to a CSV file."""
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

def main():
    args = parseArguments()

        # Define the CSV file and write the header
    results_file = 'QSVM_results.csv'
    header = ['dataset', 'type','f_score', 'accuracy', 'auc']
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    for dataset in os.listdir("alldatasets"):
        print(f"Working with {dataset} data set!")
        data, t, X_valid, y_valid, X_test, y_test = load_data_from_folder(f"alldatasets/{dataset}/")
        #data,t = DataLoader.load_data(args['dataFile'],args['trainingPoints'],args['validationPoints'])
 
        _SVM = SVM(args['B'],args['K'],args['C'],args['gamma'],args['xi'],len(data),args['type'])
        alpha, b = _SVM.train_SVM(data, t)

        #if args['fig']: utils.plot_figure(_SVM,alpha,data,t,b,args['trainingPoints'],args['type'])

        precision,recall,f_score,accuracy, auc_roc = utils.compute_metrics(_SVM,alpha,X_test,y_test,b)
        print(f'{precision=} {recall=} {f_score=} {accuracy=} {auc_roc=}')

         # Save the results to the CSV file
        results_row = [dataset, args['type'], accuracy, f_score, accuracy, auc_roc]
        save_results_to_csv(results_file, results_row)
        break

if __name__ =='__main__':
    from svm import SVM
    main()