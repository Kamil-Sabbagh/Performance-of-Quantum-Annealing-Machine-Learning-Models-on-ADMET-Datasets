import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from qboost.qboost import QBoostClassifier, qboost_lambda_sweep
from CoustumDataLoader import load_data_from_folder
from QSVM.svm import SVM
from QSVM import utils
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle

def load_or_process_dataset(dataset_path):
    """
    Load the dataset from the processed directory if available, 
    otherwise process the dataset and store it in the processed directory.

    Args:
        dataset_path (str): The path to the dataset folder.

    Returns:
        tuple: Processed training, validation, and test datasets.
    """
    # Create processed dataset directory if it doesn't exist
    if "processed_datasets" not in os.listdir():
        os.makedirs("processed_datasets")
    
    # Path for the processed dataset
    processed_dataset_path = f"processed_datasets/{os.path.basename(dataset_path)}.npz"
    
    # Load processed dataset if it exists
    if os.path.exists(processed_dataset_path):
        print(f"Loading processed dataset from {processed_dataset_path}")
        data = np.load(processed_dataset_path)
        return data['X_train'], data['t'], data['X_valid'], data['y_valid'], data['X_test'], data['y_test']
    
    # Process dataset if it hasn't been processed yet
    else:
        print(f"Processing dataset from {dataset_path}")
        data, t, X_valid, y_valid, X_test, y_test = load_data_from_folder(f"{dataset_path}")
        np.savez(processed_dataset_path, X_train=data, t=t, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)
        return data, t, X_valid, y_valid, X_test, y_test

def run_experiment_qboost(dataset_path, output_csv, args):
    """
    Run the QBoost experiment on the given dataset and store the results in a CSV file.

    Args:
        dataset_path (str): The path to the dataset.
        output_csv (str): The path to the CSV file where results will be stored.
        args: Command line arguments.
    """
    print(f"Processing dataset: {dataset_path}...")

    # Load data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_or_process_dataset(dataset_path)

    # Convert labels to -1 and 1 for binary classification
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    y_valid = np.where(y_valid == 0, -1, 1)

    # Normalize lambdas for cross-validation
    normalized_lambdas = np.linspace(0.0000, 0.0005, 5)
    lambdas = normalized_lambdas / X_train.shape[1]

    # Perform cross-validation if specified, otherwise use default lambda
    if args.cross_validation:
        print(f'Performing cross-validation with {len(lambdas)} lambda values... This may take a few minutes.')
        lambdas = normalized_lambdas / X_train.shape[1]
    else:
        lambdas = [0.00001 / X_train.shape[1]]
    
    # Perform lambda sweep using QBoost
    qboost, lam = qboost_lambda_sweep(X_train, y_train, X_valid, y_valid, lambdas, verbose=args.verbose)

    # Report baseline metrics
    qboost.report_baseline(X_test, y_test)

    # Compute and display metrics
    metrics = qboost.score(X_test, y_test)

    print('Metrics on test set:')
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    # Store results in a dictionary and save to CSV
    results = {
        "Method": "QBoost",
        "Dataset": os.path.basename(dataset_path),
        "Accuracy": metrics['accuracy'],
        "F1 Score": metrics['f1_score'],
        "AUC-ROC": metrics['auc_roc'],
        "Best_lamda:": lam
    }
    
    df = pd.DataFrame([results])

    # Create or append results to the output CSV
    if not os.path.exists(output_csv):
        print(f"Creating new output CSV file: {output_csv}")
        df.to_csv(output_csv, index=False)
    else:
        print(f"Appending results to existing CSV file: {output_csv}")
        df.to_csv(output_csv, mode='a', header=False, index=False)
    
    print(f"Results stored in {output_csv}\n")

def run_experiment_QSVM(dataset_path, output_csv, args):
    """
    Run the QSVM experiment on the given dataset and store the results in a CSV file.

    Args:
        dataset_path (str): The path to the dataset.
        output_csv (str): The path to the CSV file where results will be stored.
        args: Command line arguments.
    """
    print(f"Working with {dataset_path} data set!")
    
    # Load or process the dataset
    data, t, X_valid, y_valid, X_test, y_test = load_or_process_dataset(dataset_path)
    dataset_path = dataset_path.split('/')[-1]

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
            'B': [2],
            'K': [2],
            'C': [0.1, 1, 10],
            'gamma': [0.1, 0.5, 10],
            'xi': [0.001, 0.01, 0.1]
    }
    
    # Initialize best parameters for QSVM
    best_params = {'B': 2, 'C': 10, 'K': 2, 'gamma': 0.1, 'xi': 0.1}
    
    # Perform cross-validation if specified
    if args.cross_validation:
        n_initial = len(ParameterGrid(param_grid))
        n_candidates = n_initial
        min_candidates = 2
        reduction_factor = 2
        best_auc = -np.inf

        history_file = f"{output_csv}_history.csv"
        
        # Load or create a history dataframe to track hyperparameters
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
        else:
            history_df = pd.DataFrame(columns=['dataset_path', 'B', 'K', 'C', 'gamma', 'xi', 'auc_roc'])

        # Perform successive halving cross-validation
        while n_candidates >= min_candidates:
            param_list = list(ParameterGrid(param_grid))
            param_list = shuffle(param_list)[:n_candidates]
            round_results = []

            for params in tqdm(param_list):
                print(f"Evaluating parameters: {params}")
                match = (history_df['dataset_path'] == dataset_path) & \
                        (history_df['B'] == params['B']) & \
                        (history_df['K'] == params['K']) & \
                        (history_df['C'] == params['C']) & \
                        (history_df['gamma'] == params['gamma']) & \
                        (history_df['xi'] == params['xi'])

                # Check if these parameters have been evaluated before
                if match.any():
                    auc_roc_valid = history_df[match]['auc_roc'].values[0]
                    print(f"Found existing result in history: Validation AUC-ROC: {auc_roc_valid}")
                else:
                    _SVM = SVM(params['B'], params['K'], params['C'], params['gamma'], params['xi'], len(data), "HQPU")
                    print("Started training!")
                    alpha, b = _SVM.train_SVM(data, t)
                    print("Finished training!")
                    _, _, _, _, auc_roc_valid = utils.compute_metrics(_SVM, alpha, X_valid, y_valid, b)
                    print(f"Validation AUC-ROC: {auc_roc_valid}")

                    new_row = pd.DataFrame([{
                        'dataset_path': dataset_path, 'B': params['B'], 'K': params['K'], 'C': params['C'],
                        'gamma': params['gamma'], 'xi': params['xi'], 'auc_roc': auc_roc_valid
                    }])
                    history_df = pd.concat([history_df, new_row], ignore_index=True)
                    history_df.to_csv(history_file, index=False)

                round_results.append((params, auc_roc_valid))

            # Sort by AUC-ROC and halve candidates
            round_results.sort(key=lambda x: x[1], reverse=True)
            top_candidates = round_results[:len(round_results) // reduction_factor]

            if top_candidates[0][1] > best_auc:
                best_auc = top_candidates[0][1]
                best_params = top_candidates[0][0]

            n_candidates = len(top_candidates)

            param_grid = {
                'B': list(set([p['B'] for p, _ in top_candidates])),
                'K': list(set([p['K'] for p, _ in top_candidates])),
                'C': list(set([p['C'] for p, _ in top_candidates])),
                'gamma': list(set([p['gamma'] for p, _ in top_candidates])),
                'xi': list(set([p['xi'] for p, _ in top_candidates]))
            }

            print(f"Best parameters found: {best_params} with AUC-ROC: {best_auc}")

    # Final training and testing on the best parameters
    _SVM = SVM(best_params['B'], best_params['K'], best_params['C'], best_params['gamma'], best_params['xi'], len(data), "HQPU")
    print("Started final training")
    
    alpha, b = _SVM.train_SVM(data, t)
    print("Finished final training")

    precision, recall, f_score, accuracy, auc_roc = utils.compute_metrics(_SVM, alpha, X_test, y_test, b)
    print(f'{precision=} {recall=} {f_score=} {accuracy=} {auc_roc=}')

    results_row = pd.DataFrame([{
        'Dataset': dataset_path, 'type': "HQPU", 'accuracy': accuracy, 
        'f_score': f_score, 'auc_roc': auc_roc, 'best_params': best_params
    }])
    if not os.path.exists(output_csv):
        print(f"Creating new output CSV file: {output_csv}")
        results_row.to_csv(output_csv, mode='a', header=True, index=False)
    else:
        print(f"Appending results to existing CSV file: {output_csv}")
        results_row.to_csv(output_csv, mode='a', header=False, index=False)

def main():
    """
    Main function to run experiments for QBoost and QSVM.
    It reads the dataset folder and runs experiments on each dataset.
    """
    parser = argparse.ArgumentParser(description="The main code to run the experiments")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cross-validation', action='store_true',
                       help='Use cross-validation to estimate the value of the regularization parameter')
    
    args = parser.parse_args()

    dataset_folder = "alldatasets"
    qboos_output_csv = "Qboost_results.csv"
    qsvm_output_csv = "QSVM_results.csv"
    
    # Iterate over datasets and run experiments
    for dataset in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, dataset)
        print("Qboost:")
        run_experiment_qboost(dataset_path, qboos_output_csv, args)
        print("QSVM")
        run_experiment_QSVM(dataset_path, qsvm_output_csv, args)

if __name__ == "__main__":
    main()