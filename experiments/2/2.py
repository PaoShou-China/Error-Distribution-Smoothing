import numpy as np
from sklearn.linear_model import Lasso
from itertools import combinations_with_replacement
import time
import os

# Function to load data
def load_data(features_path, labels_path):
    features = np.loadtxt('data/processed/2/' + features_path)
    labels = np.loadtxt('data/processed/2/' + labels_path)
    return features, labels

# Build polynomial feature library and return feature names
def build_feature_library(X, poly_order=2):
    n_samples, n_features = X.shape
    library = [np.ones((n_samples, 1))]  # Add constant term
    feature_names = ['1']  # List of feature names
    base_names = [f'x{i + 1}' for i in range(n_features)]
    for order in range(1, poly_order + 1):
        for indices in combinations_with_replacement(range(n_features), order):
            term_name = '*'.join([base_names[i] for i in indices])
            term = np.prod(X[:, indices], axis=1, keepdims=True)
            library.append(term)
            feature_names.append(term_name)
    return np.hstack(library), feature_names

# SINDy algorithm implementation
def sindy(X, y, alpha=0.2, poly_order=2):
    Theta, feature_names = build_feature_library(X, poly_order)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    Xi = np.zeros((Theta.shape[1], y.shape[1]))
    for i in range(y.shape[1]):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        model.fit(Theta, y[:, i])
        Xi[:, i] = model.coef_
    return Xi, Theta, feature_names

# Load datasets
data_paths = [
    ('train_features.txt', 'train_labels.txt', 'test_features.txt', 'test_labels.txt'),
    ('sampled_features_dim0.txt', 'sampled_labels_dim0.txt', 'test_features.txt', 'test_labels.txt'),
    ('sampled_features_dim1.txt', 'sampled_labels_dim1.txt', 'test_features.txt', 'test_labels.txt'),
    ('sampled_features_dim2.txt', 'sampled_labels_dim2.txt', 'test_features.txt', 'test_labels.txt'),
    ('eds_features_dim0.txt', 'eds_labels_dim0.txt', 'test_features.txt', 'test_labels.txt'),
    ('eds_features_dim1.txt', 'eds_labels_dim1.txt', 'test_features.txt', 'test_labels.txt'),
    ('eds_features_dim2.txt', 'eds_labels_dim2.txt', 'test_features.txt', 'test_labels.txt'),
    ('smogn_features_dim0.txt', 'smogn_labels_dim0.txt', 'test_features.txt', 'test_labels.txt'),
    ('smogn_features_dim1.txt', 'smogn_labels_dim1.txt', 'test_features.txt', 'test_labels.txt'),
    ('smogn_features_dim2.txt', 'smogn_labels_dim2.txt', 'test_features.txt', 'test_labels.txt')
]

for train_features_path, train_labels_path, test_features_path, test_labels_path in data_paths:
    X_train, y_train = load_data(train_features_path, train_labels_path)
    X_test, y_test = load_data(test_features_path, test_labels_path)

    # Record start time
    start_time = time.time()

    # Apply SINDy algorithm
    Xi, Theta, feature_names = sindy(X_train, y_train, alpha=0.01, poly_order=2)

    # Record end time
    end_time = time.time()
    training_time = end_time - start_time

    print(f"\nEquations identified from {train_features_path} and {train_labels_path}:")
    # Select output variable names based on dataset type
    if 'dim' in train_features_path:
        dim = int(train_features_path.split('dim')[-1].split('.')[0])
        var_names = [f'dx{dim}/dt']
    elif 'train_features' in train_features_path:
        # For train_features and train_labels, process dim=0,1,2 in order
        for dim in range(3):
            var_names = [f'dx{dim}/dt']
            for i, (name, coeffs) in enumerate(zip(var_names, Xi.T[dim:dim+1])):
                eq = ' + '.join([f'{coeff:.3f}*{feature_names[j]}' for j, coeff in enumerate(coeffs) if abs(coeff) > 1e-3])
                print(f"{name} = {eq}")
            # Validate model
            Theta_test, _ = build_feature_library(X_test, poly_order=2)
            y_pred = Theta_test @ Xi
            error = y_pred[:, dim:dim+1] - y_test[:, dim:dim+1]
            error_norm = np.abs(error).flatten()
            rmse = np.sqrt(np.mean(error_norm**2))
            max_err = np.max(error_norm)
            print(f"RMSE on test data (dim {dim}): {rmse:.6f}")
            print(f"Maximum error on test data (dim {dim}): {max_err:.6f}")
            print(f"Training time: {training_time:.3f} seconds")
        continue
    else:
        var_names = ['dx/dt', 'dy/dt', 'dz/dt']
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    for i, (name, coeffs) in enumerate(zip(var_names, Xi.T)):
        eq = ' + '.join([f'{coeff:.3f}*{feature_names[j]}' for j, coeff in enumerate(coeffs) if abs(coeff) > 1e-3])
        print(f"{name} = {eq}")
    # Validate model
    Theta_test, _ = build_feature_library(X_test, poly_order=2)
    y_pred = Theta_test @ Xi
    if 'dim' in train_features_path:
        # For dim datasets, only compare the corresponding dimension error
        dim = int(train_features_path.split('dim')[-1].split('.')[0])
        error = y_pred - y_test[:, dim:dim+1]
        error_norm = np.abs(error).flatten()
        rmse = np.sqrt(np.mean(error_norm**2))
        max_err = np.max(error_norm)
        print(f"RMSE on test data (dim {dim}): {rmse:.6f}")
        print(f"Maximum error on test data (dim {dim}): {max_err:.6f}")
    else:
        # For full datasets, calculate error for each dimension
        for dim in range(3):
            error = y_pred[:, dim:dim+1] - y_test[:, dim:dim+1]
            error_norm = np.abs(error).flatten()
            rmse = np.sqrt(np.mean(error_norm**2))
            max_err = np.max(error_norm)
            print(f"RMSE on test data (dim {dim}): {rmse:.6f}")
            print(f"Maximum error on test data (dim {dim}): {max_err:.6f}")
        # Also calculate overall error
        error_norm = np.linalg.norm(y_pred - y_test, axis=1)
        rmse = np.sqrt(np.mean(error_norm**2))
        max_err = np.max(error_norm)
        print(f"Overall RMSE on test data: {rmse:.6f}")
        print(f"Overall Maximum error on test data: {max_err:.6f}")
    print(f"Training time: {training_time:.3f} seconds")