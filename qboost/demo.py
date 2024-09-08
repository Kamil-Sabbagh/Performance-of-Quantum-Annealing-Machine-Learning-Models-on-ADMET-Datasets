#    Copyright 2018 D-Wave Systems Inc.

#    Licensed under the Apache License, Version 2.0 (the "License")
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http: // www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
try:
    import matplotlib.pyplot as plt
except ImportError:
    # Not required for demo
    pass

from sklearn.preprocessing import StandardScaler
from qboost import QBoostClassifier, qboost_lambda_sweep, QBoostOvRClassifier
from datasets import make_blob_data, get_handwritten_digits_data
from sklearn.preprocessing import LabelEncoder
from CoustumDataLoader import load_data_from_folder

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run QBoost example",
                                     epilog="Information about additional options that are specific to the data set can be obtained using either 'demo.py blobs -h' or 'demo.py digits -h'.")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cross-validation', action='store_true',
                        help='use cross-validation to estimate the value of the regularization parameter')
    parser.add_argument('--lam', default=0.0001, type=float,
                        help='regularization parameter (default: %(default)s)')

    # Note: required=True could be useful here, but not available
    # until Python 3.7
    subparsers = parser.add_subparsers(
        title='dataset', description='dataset to use', dest='dataset')

    sp_blobs = subparsers.add_parser('blobs', help='blobs data set')
    sp_blobs.add_argument('--num-samples', type=int, default=2000,
                          help='number of samples (default: %(default)s)')
    sp_blobs.add_argument('--num-features', type=int, default=10,
                          help='number of features (default: %(default)s)')
    sp_blobs.add_argument('--num-informative', type=int, default=2,
                          help='number of informative features (default: %(default)s)')

    sp_digits = subparsers.add_parser(
        'digits', help='handwritten digits data set')
    sp_digits.add_argument('--digit1', type=int, default=0, choices=range(10),
                           help='first digit to include (default: %(default)s)')
    sp_digits.add_argument('--digit2', type=int, default=1, choices=range(10),
                           help='second digit to include (default: %(default)s)')
    sp_digits.add_argument('--plot-digits', action='store_true',
                           help='plot a random sample of each digit')
    
    sp_userBinaryData = subparsers.add_parser('binary', help='user-provided binary data set')
    sp_userBinaryData.add_argument('--csv-file', type=str, required=True,
                                   help='path to the CSV file')
    sp_userBinaryData.add_argument('--label-column', type=str, required=True,
                                   help='name of the column containing the labels')

    sp_userMultiClassData = subparsers.add_parser('multi-class', help='user-provided multi-class data set')
    sp_userMultiClassData.add_argument('--csv-file', type=str, required=True,
                                   help='path to the CSV file')
    sp_userMultiClassData.add_argument('--label-column', type=str, required=True,
                                   help='name of the column containing the labels')

    args = parser.parse_args()

    if args.dataset == 'blobs':
        
        # Parameters for blob data
        n_samples = args.num_samples
        n_features = args.num_features
        n_informative = args.num_informative

        # Generate blob data
        X_blob, y_blob = make_blob_data(
            n_samples=n_samples, n_features=n_features, n_informative=n_informative)

        X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(
            X_blob, y_blob, test_size=0.4)

        # Load data from the specified folder
        X_data_train, y_data_train, X_data_test, y_data_test, _, _ = load_data_from_folder("alldatasets/PAMPA_NCATS/")

        # Change the values of y_data_train and y_data_test from zeros to ones
        # Ensure y values are -1 and 1
        y_data_train = np.where(y_data_train == 0, -1, 1)
        y_data_test = np.where(y_data_test == 0, -1, 1)


        # Compare shapes
        print(f"Shape of Blob Training Data: {X_blob_train.shape}, Labels: {y_blob_train.shape}")
        print(f"Shape of Folder Training Data: {X_data_train.shape}, Labels: {y_data_train.shape}")
        print(f"Shape of Blob Testing Data: {X_blob_test.shape}, Labels: {y_blob_test.shape}")
        print(f"Shape of Folder Testing Data: {X_data_test.shape}, Labels: {y_data_test.shape}")

        # Compare value ranges and distributions
        print(f"Value range in Blob Training Data: {X_blob_train.min()} to {X_blob_train.max()}")
        print(f"Value range in Folder Training Data: {X_data_train.min()} to {X_data_train.max()}")

        #print(f"Label distribution in Blob Training Data: {np.bincount(y_blob_train)}")
        #print(f"Label distribution in Folder Training Data: {np.bincount(y_data_train)}")

        # (Optional) Standardize both datasets for better comparison
        scaler = StandardScaler()
        X_blob_train_scaled = scaler.fit_transform(X_blob_train)
        X_data_train_scaled = scaler.fit_transform(X_data_train)

        print(f"Value range in Scaled Blob Training Data: {X_blob_train_scaled.min()} to {X_blob_train_scaled.max()}")
        print(f"Value range in Scaled Folder Training Data: {X_data_train_scaled.min()} to {X_data_train_scaled.max()}")

        # Proceed with the original code logic using either the Blob data or the loaded data
        if args.cross_validation:
            # Cross-validation using Blob data
            normalized_lambdas = np.linspace(0.0, 0.5, 10)
            lambdas = normalized_lambdas / n_features
            print('Performing cross-validation on Blob data...')
            qboost_blob, lam_blob = qboost_lambda_sweep(X_blob_train, y_blob_train, lambdas, verbose=args.verbose)

            # Cross-validation using Folder data
            lambdas = normalized_lambdas / X_data_train.shape[1]
            print('Performing cross-validation on Folder data...')
            qboost_data, lam_data = qboost_lambda_sweep(X_data_train, y_data_train, lambdas, verbose=args.verbose)
        else:
            # No cross-validation, use default lambda
            qboost_blob = QBoostClassifier(X_blob_train, y_blob_train, args.lam)
            qboost_data = QBoostClassifier(X_data_train, y_data_train, args.lam)

        # Report results
        print("Results for Blob Data:")
        qboost_blob.report_baseline(X_blob_test, y_blob_test)
        metrics_blob = qboost_blob.score(X_blob_test, y_blob_test)
        print(f"Accuracy: {metrics_blob['accuracy']:.3f}")
        print(f"Precision: {metrics_blob['precision']:.3f}")
        print(f"Recall: {metrics_blob['recall']:.3f}")
        print(f"F1 Score: {metrics_blob['f1_score']:.3f}")
        print(f"auc_roc: {metrics_blob['auc_roc']:.3f}")

        print("Confusion Matrix:")
        print(metrics_blob['confusion_matrix'])

        print("Results for Folder Data:")
        qboost_data.report_baseline(X_data_test, y_data_test)
        metrics_data = qboost_data.score(X_data_test, y_data_test)
        print(f"Accuracy: {metrics_data['accuracy']:.3f}")
        print(f"Precision: {metrics_data['precision']:.3f}")
        print(f"Recall: {metrics_data['recall']:.3f}")
        print(f"F1 Score: {metrics_data['f1_score']:.3f}")
        print(f"auc_roc: {metrics_data['auc_roc']:.3f}")

        print("Confusion Matrix:")
        print(metrics_data['confusion_matrix'])    

    elif args.dataset == 'digits':
        if args.digit1 == args.digit2:
            raise ValueError("must use two different digits")

        X, y = get_handwritten_digits_data(args.digit1, args.digit2)
        n_features = np.size(X, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4)
        print('Number of features:', np.size(X, 1))
        print('Number of training samples:', len(X_train))
        print('Number of test samples:', len(X_test))

        if args.cross_validation:
            # See Boyda et al. (2017), Eq. (17) regarding normalization
            normalized_lambdas = np.linspace(0.0, 1.75, 10)
            lambdas = normalized_lambdas / n_features
            print('Performing cross-validation using {} values of lambda, this make take several minutes...'.format(len(lambdas)))
            qboost, lam = qboost_lambda_sweep(
                X_train, y_train, lambdas, verbose=args.verbose)
        else:
            qboost = QBoostClassifier(X_train, y_train, args.lam)

        if args.verbose:
            qboost.report_baseline(X_test, y_test)

        print('Number of selected features:',
              len(qboost.get_selected_features()))

        metrics = qboost.score(X_test, y_test)

        print('Metrics on test set:')
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print()
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

        if args.plot_digits:
            digits = load_digits()

            images1 = [image for image, target in zip(
                digits.images, digits.target) if target == args.digit1]
            images2 = [image for image, target in zip(
                digits.images, digits.target) if target == args.digit2]

            f, axes = plt.subplots(1, 2)

            # Select a random image from each set to show:
            i1 = np.random.choice(len(images1))
            i2 = np.random.choice(len(images2))
            for ax, image in zip(axes, (images1[i1], images2[i2])):
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

            plt.show()

    elif args.dataset == 'binary':
        # Load data from the specified folder
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_from_folder("alldatasets/PAMPA_NCATS/")
        X_train = np.concatenate((X_train, X_valid))
        y_train = np.concatenate((y_train, y_valid))
        y_train = np.where(y_train == 0, -1, 1)
        y_test = np.where(y_test == 0, -1, 1)

        # Perform cross-validation if enabled
        if args.cross_validation:
            # See Boyda et al. (2017), Eq. (17) regarding normalization
            normalized_lambdas = np.linspace(0.0, 0.5, 10)
            lambdas = normalized_lambdas / X_train.shape[1]
            print('Performing cross-validation using {} values of lambda, this may take several minutes...'.format(len(lambdas)))
            qboost, lam = qboost_lambda_sweep(X_train, y_train, lambdas, verbose=args.verbose)
        else:
            qboost = QBoostClassifier(X_train, y_train, args.lam)

        #if args.verbose:
        qboost.report_baseline(X_test, y_test)

        metrics = qboost.score(X_test, y_test)

        print('Metrics on test set:')
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print(f"auc_roc: {metrics['auc_roc']:.3f}")
        print()
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])


    elif args.dataset == 'multi-class':
        data = pd.read_csv(args.csv_file)
        X = data.drop(columns=[args.label_column]).values
        y = data[args.label_column].values

        # Encode labels to integers
        le = LabelEncoder()
        y = le.fit_transform(y)

        print("Classes found:", le.classes_)
        for i, class_name in enumerate(le.classes_):
            print(f"Class {i}: {class_name}")

        n_features = np.size(X, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4)
        print('Number of features:', n_features)
        print('Number of training samples:', len(X_train))
        print('Number of test samples:', len(X_test))

        if args.cross_validation:
            # See Boyda et al. (2017), Eq. (17) regarding normalization
            normalized_lambdas = np.linspace(0.0, 1.75, 10)
            lambdas = normalized_lambdas / n_features
            print('Performing cross-validation using {} values of lambda, this make take several minutes...'.format(len(lambdas)))
            qboost, lam = qboost_lambda_sweep(
                X_train, y_train, lambdas, verbose=args.verbose)
        else:
            qboost = QBoostOvRClassifier(lam=args.lam)
            qboost.fit(X_train, y_train)

        if args.verbose:
            # Assuming qboost has a method for verbose reporting in a multi-class setting
            for i, clf in enumerate(qboost.classifiers_):
                print(f"Classifier for class {le.classes_[i]}:")
                clf.report_baseline(X_test, y_test)

        print('Number of selected features:')
        for i, clf in enumerate(qboost.classifiers_):
            print(f"Class {le.classes_[i]} selected features:", clf.get_selected_features())

        metrics = qboost.score(X_test, y_test)

        print('Metrics on test set:')
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1 Score: {metrics['f1_score']:.3f}")
        print()
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])

    elif not args.dataset:
        parser.print_help()
