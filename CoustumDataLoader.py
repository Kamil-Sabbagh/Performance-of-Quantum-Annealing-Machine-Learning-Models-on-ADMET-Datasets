import pandas as pd
import anndata
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def process_data(adata, train_df, valid_df, test_df):
    """
    Processes the data by extracting features for each drug from anndata object
    and assigns them to the appropriate dataset (train, test, or validation) 
    based on the provided dataframes.

    Args:
        adata (AnnData): Annotated data object containing the features.
        train_df (DataFrame): DataFrame containing training data with drug names and labels.
        valid_df (DataFrame): DataFrame containing validation data with drug names and labels.
        test_df (DataFrame): DataFrame containing test data with drug names and labels.

    Returns:
        tuple: Processed feature matrices (X) and corresponding labels (y) for training, test, and validation sets.
    """
    X_train, X_test, X_valid = [], [], []
    y_train, y_test, y_valid = [], [], []

    # Extract all feature keys from the annotated data object
    feature_keys = list(adata.obsm.keys())

    for i, drug in enumerate(adata.obs.smiles):
        drug_features = []

        # Collect features for each drug across all feature keys
        for key in feature_keys:
            drug_features.append(adata.obsm[key][i])

        # Concatenate all collected features
        drug_features = np.concatenate(drug_features)

        # Assign the features and labels to the appropriate dataset
        if drug in train_df["Drug"].values:
            X_train.append(drug_features)
            y_train.append(train_df[train_df["Drug"] == drug]["Y"].values[0])
        elif drug in test_df["Drug"].values:
            X_test.append(drug_features)
            y_test.append(test_df[test_df["Drug"] == drug]["Y"].values[0])
        elif drug in valid_df["Drug"].values:
            X_valid.append(drug_features)
            y_valid.append(valid_df[valid_df["Drug"] == drug]["Y"].values[0])
        else:
            print(f"Drug not found: {drug}")

    return (np.array(X_train), np.array(y_train),
            np.array(X_test), np.array(y_test),
            np.array(X_valid), np.array(y_valid))

class DrugDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for handling drug feature data.

    Args:
        X (array): Feature matrix for the dataset.
        y (array): Labels for the dataset.
    """
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a single data sample (features and label) at the specified index.
        """
        return self.X[idx], self.y[idx]

def create_dataloaders(X_train, y_train, X_test, y_test, X_valid, y_valid, batch_size=32):
    """
    Creates DataLoader objects for training, validation, and testing datasets.

    Args:
        X_train, X_test, X_valid (array): Feature matrices for training, test, and validation sets.
        y_train, y_test, y_valid (array): Labels for training, test, and validation sets.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.

    Returns:
        tuple: DataLoader objects for training, test, and validation datasets.
    """
    train_dataset = DrugDataset(X_train, y_train)
    test_dataset = DrugDataset(X_test, y_test)
    valid_dataset = DrugDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader

def load_data_from_folder(path, variance_threshold=0.98, balance_data=True):
    """
    Loads and processes drug dataset files from the specified folder. Performs feature extraction, 
    normalization, and dimensionality reduction using PCA. Optionally balances the training data.

    Args:
        path (str): Folder path containing the dataset files.
        variance_threshold (float, optional): The amount of variance to preserve in PCA. Defaults to 0.98.
        balance_data (bool, optional): Whether to balance the training dataset. Defaults to True.

    Returns:
        tuple: Processed feature matrices (X) and corresponding labels (y) for training, test, and validation sets.
    """
    # File paths
    folder_path = path
    hdf_file = 'smiles_descriptors.hdf'
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'

    # Step 1: Load the annotated HDF5 file
    adata = anndata.read_h5ad(os.path.join(folder_path, hdf_file))

    # Step 2: Load the CSV files for train, validation, and test sets
    train_df = pd.read_csv(os.path.join(folder_path, train_file))
    valid_df = pd.read_csv(os.path.join(folder_path, valid_file))
    test_df = pd.read_csv(os.path.join(folder_path, test_file))

    # Step 3: Process the data to extract features and labels
    X_train, y_train, X_test, y_test, X_valid, y_valid = process_data(adata, train_df, valid_df, test_df)

    # Step 4: Replace NaN values in the data with a small number
    small_number = 1e-10
    X_train = np.where(np.isnan(X_train), small_number, X_train)
    y_train = np.where(np.isnan(y_train), small_number, y_train)
    X_test = np.where(np.isnan(X_test), small_number, X_test)
    y_test = np.where(np.isnan(y_test), small_number, y_test)
    X_valid = np.where(np.isnan(X_valid), small_number, X_valid)
    y_valid = np.where(np.isnan(y_valid), small_number, y_valid)

    # Step 5: Normalize the data using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)

    # Step 6: Apply PCA for dimensionality reduction while preserving 98% variance
    pca = PCA(n_components=variance_threshold)
    X_train = pca.fit_transform(X_train)
    X_valid = pca.transform(X_valid)
    X_test = pca.transform(X_test)

    # Step 7: Balance the classes in the training dataset if specified
    if balance_data:
        X_train, y_train = balance_classes(X_train, y_train)

    # Step 8: Display class distribution in training, validation, and test sets
    def count_classes(y):
        unique, counts = np.unique(y, return_counts=True)
        return dict(zip(unique, counts))

    y_train_counts = count_classes(y_train)
    y_valid_counts = count_classes(y_valid)
    y_test_counts = count_classes(y_test)

    print("Class distribution in y_train:", y_train_counts)
    print("Class distribution in y_valid:", y_valid_counts)
    print("Class distribution in y_test:", y_test_counts)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def balance_classes(X, y):
    """
    Balances the classes in the dataset by oversampling the minority class.

    Args:
        X (array): Feature matrix.
        y (array): Labels corresponding to the feature matrix.

    Returns:
        tuple: Balanced feature matrix and labels.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_0_count, class_1_count = counts[0], counts[1]

    # Identify minority and majority classes
    if class_0_count < class_1_count:
        minority_class = 0
        majority_class = 1
    else:
        minority_class = 1
        majority_class = 0

    # Get indices of each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Duplicate the minority class to balance the dataset
    num_to_duplicate = abs(class_0_count - class_1_count)
    duplicated_minority_indices = np.random.choice(minority_indices, size=num_to_duplicate, replace=True)

    # Combine the original data with duplicated samples
    X_balanced = np.vstack((X, X[duplicated_minority_indices]))
    y_balanced = np.concatenate((y, y[duplicated_minority_indices]))

    return X_balanced, y_balanced