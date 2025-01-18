import numpy as np
from sklearn.linear_model import Lasso
from itertools import combinations_with_replacement
import time


# Function to load data from files
def load_data(features_path, labels_path):
    """Load features and labels from specified paths."""
    features = np.loadtxt('lorenz_dataset/' + features_path)
    labels = np.loadtxt('lorenz_dataset/' + labels_path)
    return features, labels


# Function to build a polynomial feature library and return feature names
def build_feature_library(X, poly_order=2):
    """Build a polynomial feature library and return feature names."""
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
def sindy(X, y, alpha=0.1, poly_order=2):
    """Implement the SINDy algorithm."""
    # Build feature library and feature names
    Theta, feature_names = build_feature_library(X, poly_order)

    # Initialize coefficient matrix
    Xi = np.zeros((Theta.shape[1], y.shape[1]))

    # Perform Lasso regression for each output dimension
    for i in range(y.shape[1]):
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
        model.fit(Theta, y[:, i])
        Xi[:, i] = model.coef_

    return Xi, Theta, feature_names


# Data paths
data_paths = [
    ('train_features.txt', 'train_labels.txt', 'test_features.txt', 'test_labels.txt'),
    ('sampled_features.txt', 'sampled_labels.txt', 'test_features.txt', 'test_labels.txt'),
    ('managed_features.txt', 'managed_labels.txt', 'test_features.txt', 'test_labels.txt')
]

for train_features_path, train_labels_path, test_features_path, test_labels_path in data_paths:
    X_train, y_train = load_data(train_features_path, train_labels_path)
    X_test, y_test = load_data(test_features_path, test_labels_path)

    # Record start time
    start_time = time.time()

    # Apply SINDy algorithm
    Xi, Theta, feature_names = sindy(X_train, y_train, alpha=0.01, poly_order=2)  # Adjust alpha to control sparsity

    # Record end time
    end_time = time.time()

    # Calculate training time
    training_time = end_time - start_time

    # Print identified differential equations
    print(f"Equations identified from {train_features_path} and {train_labels_path}:")
    for i, (name, coeffs) in enumerate(zip(['dx/dt', 'dy/dt', 'dz/dt'], Xi.T)):
        eq = ' + '.join([f'{coeff:.3f}*{feature_names[j]}' for j, coeff in enumerate(coeffs) if abs(coeff) > 1e-3])
        print(f"{name} = {eq}")

    # Validate the model
    Theta_test, _ = build_feature_library(X_test, poly_order=2)
    y_pred = Theta_test @ Xi
    error_norm = np.linalg.norm(y_pred - y_test, axis=1)
    # Calculate root mean square error (RMSE)
    rmse = np.sqrt(np.mean(error_norm**2))

    # Calculate maximum error (Max Error)
    max_err = np.max(error_norm)

    print(f"RMSE on test data: {rmse:.6f}")
    print(f"Maximum error on test data: {max_err:.6f}")
    print(f"Training time: {training_time:.3f} seconds\n")