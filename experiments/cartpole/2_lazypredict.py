import numpy as np
from lazypredict.Supervised import LazyRegressor
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, max_error

def load_data(features_path, labels_path):
    print(f"Loading data from: {features_path}, {labels_path}")
    features = np.loadtxt('data/processed/3/' + features_path)
    labels = np.loadtxt('data/processed/3/' + labels_path)
    print(f"Loaded shapes - features: {features.shape}, labels: {labels.shape}")
    # Ensure features is a column vector if 1D
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    # Ensure labels is a column vector if 1D
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    return features, labels

def evaluate_with_lazypredict(train_features, train_labels, test_features, test_labels, dataset_name, dim_index=None):
    print(f"\nEvaluating {dataset_name} dataset...")
    # If evaluating a specific dimension, use only the corresponding column
    if dim_index is not None:
        if train_labels.shape[1] > 1:
            train_labels = train_labels[:, dim_index:dim_index+1]
        if test_labels.shape[1] > 1:
            test_labels = test_labels[:, dim_index:dim_index+1]
    # Initialize LazyRegressor
    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    # Train models and get performance metrics
    models, _ = reg.fit(train_features, test_features, train_labels.ravel(), test_labels.ravel())
    # Create a DataFrame to store all predictions and extra evaluation metrics
    all_predictions = pd.DataFrame()
    all_predictions['Actual'] = test_labels.ravel()
    # Store max error for each model
    max_errors = {}
    print("\nGetting predictions and max error for each model...")
    # Iterate over all successfully trained models
    for model_name, model in reg.models.items():
        try:
            y_pred = model.predict(test_features)
            all_predictions[model_name] = y_pred
            model_max_error = max_error(test_labels.ravel(), y_pred)
            max_errors[model_name] = model_max_error
            print(f"\nModel: {model_name}")
            print(f"First 5 predictions: {y_pred[:5]}")
            print(f"Last 5 predictions: {y_pred[-5:]}")
            print(f"Max Error: {model_max_error:.4f}")
        except Exception as e:
            print(f"Could not get predictions for model {model_name}: {str(e)}")
            max_errors[model_name] = np.nan
    # Add max error to model evaluation metrics
    max_errors_series = pd.Series(max_errors)
    models['Max Error'] = max_errors_series
    print("\nUpdated model evaluation metrics (including Max Error):")
    print(models)
    # Save all predictions to CSV (optional)
    # predictions_file = f'{dataset_name.replace("/", "_")}{"_dim" + str(dim_index) if dim_index is not None else ""}_predictions.csv'
    # all_predictions.to_csv(predictions_file, index=False)
    # print(f"\nAll model predictions saved to {predictions_file}")
    # Save model evaluation metrics to CSV
    output_file = f'results/lazypredict/cartpole/{dataset_name.split("_")[0]}_dim{dim_index}.csv'
    models.to_csv(output_file)
    print(f"\nModel evaluation metrics (including Max Error) saved to {output_file}")

if __name__ == "__main__":
    datasets = [
        ('train_features', 'train_labels', 0),  # Original training data dim0
        ('train_features', 'train_labels', 1),  # Original training data dim1
        ('eds_features_dim0', 'eds_labels_dim0', 0),  # EDS dim0
        ('eds_features_dim1', 'eds_labels_dim1', 1),  # EDS dim1
        ('smogn_features_dim0', 'smogn_labels_dim0', 0),  # SMOGN dim0
        ('smogn_features_dim1', 'smogn_labels_dim1', 1),  # SMOGN dim1
        ('sampled_features_dim0', 'sampled_labels_dim0', 0),  # sampled dim0
        ('sampled_features_dim1', 'sampled_labels_dim1', 1),  # sampled dim1
    ]
    # Load test data
    test_features, test_labels = load_data('test_features.txt', 'test_labels.txt')
    for features_name, labels_name, dim_index in datasets:
        # Load training data
        train_features, train_labels = load_data(features_name + '.txt', labels_name + '.txt')
        # Evaluate with LazyPredict
        evaluate_with_lazypredict(
            train_features, 
            train_labels, 
            test_features, 
            test_labels,
            features_name,
            dim_index
        ) 