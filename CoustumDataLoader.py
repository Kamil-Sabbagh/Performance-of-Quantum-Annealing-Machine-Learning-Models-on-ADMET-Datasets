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
    X_train, X_test, X_valid = [], [], []
    y_train, y_test, y_valid = [], [], []

    # Get all feature keys
    feature_keys = list(adata.obsm.keys())

    for i, drug in enumerate(adata.obs.smiles):
        # Initialize a list to store features for this drug
        drug_features = []

        # Collect features from all keys
        for key in feature_keys:
            drug_features.append(adata.obsm[key][i])
        
        # Concatenate all features
        drug_features = np.concatenate(drug_features)

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
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_dataloaders(X_train, y_train, X_test, y_test, X_valid, y_valid, batch_size=32):
    train_dataset = DrugDataset(X_train, y_train)
    test_dataset = DrugDataset(X_test, y_test)
    valid_dataset = DrugDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader
    
def load_data_from_folder(path, variance_threshold=0.98, balance_data=True):
    # Paths to the files
    folder_path = path
    hdf_file = 'smiles_descriptors.hdf'
    train_file = 'train.csv'
    valid_file = 'valid.csv'
    test_file = 'test.csv'

    # Step 1: Load the HDF file using anndata
    adata = anndata.read_h5ad(os.path.join(folder_path, hdf_file))


    # Step 2: Load the CSV files
    train_df = pd.read_csv(os.path.join(folder_path, train_file))
    valid_df = pd.read_csv(os.path.join(folder_path, valid_file))
    test_df = pd.read_csv(os.path.join(folder_path, test_file))

    # Step 3: Process the data (this function should be defined elsewhere)
    X_train, y_train, X_test, y_test, X_valid, y_valid = process_data(adata, train_df, valid_df, test_df)

    # Step 4: Handle NaN values by replacing them with a small number
    small_number = 1e-10
    X_train = np.where(np.isnan(X_train), small_number, X_train)
    y_train = np.where(np.isnan(y_train), small_number, y_train)
    X_test = np.where(np.isnan(X_test), small_number, X_test)
    y_test = np.where(np.isnan(y_test), small_number, y_test)
    X_valid = np.where(np.isnan(X_valid), small_number, X_valid)
    y_valid = np.where(np.isnan(y_valid), small_number, y_valid)

    # Step 5: Merge the datasets (features only) for PCA
    #X_all = np.vstack((X_train, X_valid, X_test))
    #y_all = np.concatenate((y_train, y_valid, y_test))

    # Step 6: Normalize the merged data
    scaler = StandardScaler()
    #X_all_normalized = scaler.fit_transform(X_all)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_valid = scaler.fit_transform(X_valid)

    # Step 7: Apply PCA, keeping 98% of the variance
    pca = PCA(n_components=variance_threshold)
    X_train = pca.fit_transform(X_train)
    X_valid = pca.transform(X_valid)
    X_test = pca.transform(X_test)

    # Step 9: Split the data into 70% train, 10% validation, and 20% test
    #X_train_pca, X_temp_pca, y_train, y_temp = train_test_split(X_all_pca, y_all, test_size=0.3, random_state=42)
    #X_valid_pca, X_test_pca, y_valid, y_test = train_test_split(X_temp_pca, y_temp, test_size=2/3, random_state=42)

    # Step 8: Balance the data if required
    if balance_data:
        X_train, y_train = balance_classes(X_train, y_train)

    # Step 10: Check and compare the number of 0s and 1s in y_train, y_valid, and y_test
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
    Balances the classes in the dataset by duplicating the underrepresented class.
    """
    unique, counts = np.unique(y, return_counts=True)
    class_0_count, class_1_count = counts[0], counts[1]

    # Determine the minority and majority classes
    if class_0_count < class_1_count:
        minority_class = 0
        majority_class = 1
    else:
        minority_class = 1
        majority_class = 0

    # Find indices of each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate how many duplicates are needed to balance the classes
    num_to_duplicate = abs(class_0_count - class_1_count)

    # Duplicate the minority class samples
    duplicated_minority_indices = np.random.choice(minority_indices, size=num_to_duplicate, replace=True)

    # Combine original data with duplicated minority class samples
    X_balanced = np.vstack((X, X[duplicated_minority_indices]))
    y_balanced = np.concatenate((y, y[duplicated_minority_indices]))

    return X_balanced, y_balanced