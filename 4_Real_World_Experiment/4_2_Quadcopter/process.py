from manager import Manager
import numpy as np


# Function to load data from files
def load_data(features_path, labels_path):
    """
    Load features and labels from specified file paths.

    Parameters:
    - features_path (str): Path to the features file.
    - labels_path (str): Path to the labels file.

    Returns:
    - tuple: Features and labels as numpy arrays.
    """
    features = np.loadtxt(features_path)
    labels = np.loadtxt(labels_path)

    # Check if features is a vector and convert it to a column vector
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Check if labels is a vector and convert it to a column vector
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    return features, labels


# Function to process data in batches
def process_data(batch_size, initial_batch_size, folder_name, delta):
    """
    Process data in batches and update the DataManager.

    Parameters:
    - batch_size (int): Size of each batch.
    - initial_batch_size (int): Size of the initial batch.
    - folder_name (str): Folder containing the data files.
    - delta (float): Error threshold for adding points to the training set.

    Returns:
    - manager:  The updated Manager instance.
    """

    # Load data from txt files
    features_path = f'{folder_name}/train_features.txt'
    labels_path = f'{folder_name}/train_labels.txt'
    features, labels = load_data(features_path, labels_path)

    # Initialize Manager with the initial batch of data points
    manager = Manager(features[:initial_batch_size], labels[:initial_batch_size], delta)

    # Update with new data points in batches
    for i in range(initial_batch_size, features.shape[0], batch_size):
        end_idx = min(i + batch_size, features.shape[0])
        manager.update(features[i:end_idx], labels[i:end_idx])

    # Calculate the average error
    avg_error = manager.average_error()
    print(f'Average error: {avg_error}')

    return manager


if __name__ == "__main__":
    folder_name = 'dataset/'
    manager = process_data(batch_size=5, initial_batch_size=8, folder_name=folder_name, delta=0.01)

    # Save managed features and labels
    np.savetxt(folder_name+'/managed_features.txt', manager.buf_x_train.data)
    np.savetxt(folder_name+'/managed_labels.txt', manager.buf_y_train.data)

    # Sample data from total data
    A, B = manager.buf_x_total.data.shape[0], manager.buf_x_train.data.shape[0]
    indices = np.random.choice(A, size=B, replace=False)
    np.savetxt(folder_name+'/sampled_features.txt', manager.buf_x_total.data[indices])
    np.savetxt(folder_name+'/sampled_labels.txt', manager.buf_y_total.data[indices])


