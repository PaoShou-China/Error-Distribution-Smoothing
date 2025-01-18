import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, max_error
import time
import matplotlib.pyplot as plt


def load_data(features_path, labels_path):
    """
    Load features and labels from specified file paths.
    """
    features = np.loadtxt('dataset/' + features_path)
    labels = np.loadtxt('dataset/' + labels_path)

    # Check if features is a vector and convert it to a column vector
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Check if labels is a vector and convert it to a column vector
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1).ravel()  # Ensure labels are 1D for sklearn compatibility

    return features, labels


def save_evaluation_results(file_path, epoch, rmse, max_err):
    """
    Save evaluation results to a specified file.
    """
    with open(file_path, 'a') as f:
        f.write(f"{epoch},{rmse:.5f},{max_err:.5f}\n")


def evaluate_mlp(train_features, train_labels, test_features, test_labels, max_epochs=2000, eval_interval=10,
                 results_file='evaluation_results.txt'):
    """
    Train and evaluate an MLP model, saving results to a file, supporting multi-output
    """
    # Clear file content if it exists
    open(results_file, 'w').close()

    # Check if the number of samples in features and labels match
    if train_features.shape[0] != train_labels.shape[0]:
        raise ValueError("Training features and labels must have the same number of samples.")
    if test_features.shape[0] != test_labels.shape[0]:
        raise ValueError("Test features and labels must have the same number of samples.")

    # Standardize feature data (fit on training set, transform on test set)
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)

    # Create and initialize the MLP model
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64), activation='tanh', solver='adam',
                       alpha=0.0001, batch_size=128, learning_rate='constant',
                       learning_rate_init=0.0005, power_t=0.5, max_iter=eval_interval,
                       shuffle=True, random_state=3704, tol=1e-6,
                       verbose=False, warm_start=True, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False,
                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-8, n_iter_no_change=10, max_fun=15000)

    for epoch in range(0, max_epochs, eval_interval):
        # Start timing
        start_time = time.time()

        # Train the model
        mlp.fit(scaled_train_features, train_labels)

        # End timing
        end_time = time.time()

        y_pred = mlp.predict(scaled_test_features)
        rmse = mean_squared_error(test_labels, y_pred, squared=False)
        max_err = max_error(test_labels, y_pred)

        # Save evaluation results to file
        save_evaluation_results(results_file, epoch + eval_interval, rmse, max_err)

    print(f"Training completed.")


def plot_evaluation_results(results_file, dataset_name):
    """
    Read evaluation results and plot them.
    """
    epochs, rmses, max_errors = [], [], []

    with open(results_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            epoch, rmse, max_err = map(float, line.strip().split(','))
            epochs.append(epoch)
            rmses.append(rmse)
            max_errors.append(max_err)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, rmses, label='RMSE')
    plt.plot(epochs, max_errors, label='Max Error')
    plt.title(f'{dataset_name} Dataset: Metrics vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    for head in ['train', 'sampled', 'managed']:
        print(f"Evaluating {head} dataset over iterations...")

        # Load all data
        train_features, train_labels = load_data(head + '_features.txt', head + '_labels.txt')
        test_features, test_labels = load_data('test_features.txt', 'test_labels.txt')

        # Define the results file name
        results_file = f'{head}_evaluation_results.txt'

        # Evaluate the model and get results for each iteration
        evaluate_mlp(train_features, train_labels, test_features, test_labels, results_file=results_file, max_epochs=3000)

        print(f"\nFinal Results for {head} dataset have been saved to {results_file}.\n")