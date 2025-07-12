import numpy as np
from scipy.spatial import Delaunay, cKDTree
from pathlib import Path


class Buffer:
    def __init__(self, init_data):
        """
        Initialize buffer to store data.

        Parameters:
        - init_data (np.ndarray): Initial data, shape (n_samples, dim)
        """
        self.data = init_data
        self.dim = init_data.shape[1]

    def append(self, data):
        """
        Append new data to the buffer.

        Parameters:
        - data (np.ndarray): New data, shape (n_samples, dim)
        """
        data = data.reshape(-1, self.dim)
        self.data = np.vstack([self.data, data])


class Manager:
    def __init__(self, x_init, y_init, delta):
        """
        Initialize manager for a single label dimension.

        Parameters:
        - x_init (np.ndarray): Initial input features, shape (n_samples, x_dim)
        - y_init (np.ndarray): Initial labels for one dimension, shape (n_samples, )
        - delta (float): Threshold for adding new data to training set
        """
        assert x_init.shape[0] == y_init.shape[0], "Mismatch between x and y samples"
        assert x_init.shape[0] >= x_init.shape[1] + 3, "Not enough data for Delaunay triangulation"
        assert delta > 0, "Delta must be positive"

        self.x_dim = x_init.shape[1]
        self.delta = delta

        # Initialize buffers
        self.buf_x_total = Buffer(x_init)
        self.buf_y_total = Buffer(y_init.reshape(-1, 1))

        # Train/test split
        self.buf_x_train = Buffer(x_init[1:])
        self.buf_y_train = Buffer(y_init[1:].reshape(-1, 1))
        self.buf_x_test = Buffer(x_init[:1])
        self.buf_y_test = Buffer(y_init[:1].reshape(-1, 1))

        # Build Delaunay triangulation and KDTree
        self.tri, self.tree = self.train()

    def train(self):
        """
        Build Delaunay triangulation and KDTree from training data.
        """
        tri = Delaunay(self.buf_x_train.data)
        tree = cKDTree(self.buf_x_train.data)
        return tri, tree

    def update(self, new_x, new_y):
        """
        Update data and decide whether to add to training or test set.

        Parameters:
        - new_x (np.ndarray): New input features, shape (n_samples, x_dim)
        - new_y (np.ndarray): New label values for this dimension, shape (n_samples, )
        """
        new_x = new_x.reshape(-1, self.x_dim)
        new_y = new_y.reshape(-1)

        self.buf_x_total.append(new_x)
        self.buf_y_total.append(new_y.reshape(-1, 1))

        errors = [self.error(x, y) for x, y in zip(new_x, new_y)]

        for x, y, error in zip(new_x, new_y, errors):
            if error > self.delta:
                self.buf_x_train.append(x)
                self.buf_y_train.append(y)
            else:
                self.buf_x_test.append(x)
                self.buf_y_test.append(y)

        self.tri, self.tree = self.train()

    def error(self, x, y):
        """
        Compute prediction error for this label dimension.
        """
        return abs(y - self.predict(x))

    def predict(self, x):
        """
        Predict using Delaunay barycentric interpolation or nearest neighbor.
        """
        simplex_idx = self.tri.find_simplex(x)

        if simplex_idx == -1:
            vertex_idx = self.find_nearest_vertex_idx(x)
            y_hat = self.buf_y_train.data[vertex_idx][0]
        else:
            t = self.tri.transform[simplex_idx]
            proj = t[:self.x_dim].dot(x - t[self.x_dim])
            bary_coords = np.append(proj, 1 - proj.sum())

            vertices = self.buf_y_train.data[self.tri.simplices[simplex_idx]]
            y_hat = self.barycentric_interpolation(vertices, bary_coords)

        return y_hat

    @staticmethod
    def barycentric_interpolation(vals, weights):
        """
        Perform barycentric interpolation.

        Parameters:
        - vals (np.ndarray): Values at the simplex vertices
        - weights (np.ndarray): Barycentric weights

        Returns:
        - float: Interpolated value
        """
        weights = weights.reshape(-1, 1)
        return np.sum(vals * weights, axis=0)[0]

    def find_nearest_vertex_idx(self, x):
        """
        Find the index of the nearest vertex in the training set.
        """
        _, idx = self.tree.query(x.reshape(1, -1))
        return idx[0]

    def average_error(self):
        """
        Compute average error on the test set.
        """
        x_test, y_test = self.buf_x_test.data, self.buf_y_test.data
        errors = [self.error(x, y[0]) for x, y in zip(x_test, y_test)]
        return sum(errors) / len(errors)


def load_data(features_path, labels_path):
    """
    Load feature and label data from text files.

    Parameters:
    - features_path (str): Path to features file
    - labels_path (str): Path to labels file

    Returns:
    - tuple: (features, labels) as numpy arrays
    """
    features = np.loadtxt(features_path)
    labels = np.loadtxt(labels_path)

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    return features, labels


def process_data(batch_size, initial_batch_size, folder_name, delta, target_dim):
    """
    Process data and generate training set for a specified label dimension.

    Parameters:
    - batch_size (int): Number of samples per batch
    - initial_batch_size (int): Initial number of training samples
    - folder_name (str): Folder containing data
    - delta (float): Threshold for adding new samples
    - target_dim (int): Index of the label dimension to process

    Returns:
    - Manager: Trained manager object
    """
    features_path = f'data/raw/{folder_name}/train_features.txt'
    labels_path = f'data/raw/{folder_name}/train_labels.txt'

    features, labels = load_data(features_path, labels_path)

    # Validate target dimension
    assert target_dim < labels.shape[1], "Target dimension out of range"

    # Extract label data for the specified dimension
    y_init_j = labels[:initial_batch_size, target_dim]
    manager_j = Manager(features[:initial_batch_size], y_init_j, delta)

    for i in range(initial_batch_size, features.shape[0], batch_size):
        end_idx = min(i + batch_size, features.shape[0])
        new_x = features[i:end_idx]
        new_y_j = labels[i:end_idx, target_dim]
        manager_j.update(new_x, new_y_j)

    # Save training data with dimension index in filename
    output_folder = f"data/processed/{folder_name}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    np.savetxt(f"{output_folder}/eds_features_dim{target_dim}.txt", manager_j.buf_x_train.data)
    np.savetxt(f"{output_folder}/eds_labels_dim{target_dim}.txt", manager_j.buf_y_train.data)

    return manager_j


if __name__ == "__main__":
    # Dataset 1 - Dimension 0
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='1',
        delta=0.005,
        target_dim=0
    )
    print(f"[Dataset: dataset/1, Dim: 0] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/1, Dim: 0] Average test error: {manager.average_error()}\n")

    # Dataset 2 - Dimension 0
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='2',
        delta=0.7,
        target_dim=0
    )
    print(f"[Dataset: dataset/2, Dim: 0] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/2, Dim: 0] Average test error: {manager.average_error()}\n")

    # Dataset 2 - Dimension 1
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='2',
        delta=0.7,
        target_dim=1
    )
    print(f"[Dataset: dataset/2, Dim: 1] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/2, Dim: 1] Average test error: {manager.average_error()}\n")

    # Dataset 2 - Dimension 2
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='2',
        delta=0.7,
        target_dim=2
    )
    print(f"[Dataset: dataset/2, Dim: 2] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/2, Dim: 2] Average test error: {manager.average_error()}\n")


    # Dataset 3 - Dimension 0
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='3',
        delta=0.001,
        target_dim=0
    )
    print(f"[Dataset: dataset/3, Dim: 0] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/3, Dim: 0] Average test error: {manager.average_error()}\n")

    # Dataset 3 - Dimension 1
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='3',
        delta=0.001,
        target_dim=1
    )
    print(f"[Dataset: dataset/3, Dim: 1] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/3, Dim: 1] Average test error: {manager.average_error()}\n")

    # Dataset 4 - Dimension 0
    manager = process_data(
        batch_size=10,
        initial_batch_size=10,
        folder_name='4',
        delta=0.001,
        target_dim=0
    )
    print(f"[Dataset: dataset/4, Dim: 0] Training set size: {manager.buf_x_train.data.shape[0]}")
    print(f"[Dataset: dataset/4, Dim: 0] Average test error: {manager.average_error()}\n")


