import numpy as np
from scipy.spatial import Delaunay, cKDTree


class Buffer:
    """
    A buffer to store and manage data points.
    """

    def __init__(self, init_data):
        """
        Initialize the buffer with initial data.

        Parameters:
        - init_data (numpy.ndarray): Initial data to store in the buffer.
        """
        self.data = init_data
        self.dim = init_data.shape[1]
        # Keep the public data array exact-sized while reserving geometric capacity.
        self._storage = init_data
        self._capacity = init_data.shape[0]

    def append(self, data):
        """
        Append new data to the buffer.

        Parameters:
        - data (numpy.ndarray): New data to append.
        """
        data = data.reshape(-1, self.dim)
        size = self.data.shape[0]
        required = size + data.shape[0]
        dtype = np.result_type(self._storage.dtype, data.dtype)

        if required > self._capacity or dtype != self._storage.dtype:
            capacity = max(required, max(1, self._capacity * 2))
            storage = np.empty((capacity, self.dim), dtype=dtype)
            storage[:size] = self.data
            self._storage = storage
            self._capacity = capacity

        self._storage[size:required] = data
        self.data = self._storage[:required]


class Manager:
    """
    A class to manage and update data points for training and testing.
    """
    def __init__(self, x_init, y_init, delta):
        """
        Initialize the Manager class.

        Parameters:
        - x_init (numpy.ndarray): Initial input data.
        - y_init (numpy.ndarray): Initial output data.
        - delta (float): Error threshold for adding points to the training set.
        """
        assert x_init.shape[0] == y_init.shape[0]
        assert x_init.shape[0] >= x_init.shape[1] + 3
        assert delta > 0

        self.x_dim, self.y_dim = x_init.shape[1], y_init.shape[1]
        self.delta = delta

        self.buf_x_total, self.buf_y_total = Buffer(x_init), Buffer(y_init)
        self.buf_x_train, self.buf_y_train = Buffer(x_init[1:]), Buffer(y_init[1:])
        self.buf_x_test, self.buf_y_test = Buffer(x_init[:1]), Buffer(y_init[:1])

        self.tri, self.tree = self.train()

    def train(self):
        """
        Train the linear model and build Delaunay triangulation and cKDTree.
        """
        tri = Delaunay(self.buf_x_train.data)
        tree = cKDTree(self.buf_x_train.data)
        return tri, tree

    def add_points(self, new_points):
        """
        Add new points to the Delaunay triangulation and cKDTree.

        Parameters:
        - new_points (numpy.ndarray): New points to add.
        """
        self.tri.add_points(new_points)
        self.tree = cKDTree(np.vstack((self.tree.data, new_points)))

    def update(self, new_x, new_y):
        """
        Batch update data points.

        Parameters:
        - new_x (numpy.ndarray): New input data points.
        - new_y (numpy.ndarray): New output data points.
        """
        new_x = np.asarray(new_x)
        new_y = np.asarray(new_y)
        if new_x.ndim != 2 or new_x.shape[1] != self.x_dim:
            raise ValueError(f"new_x must have shape (n_samples, {self.x_dim})")
        if new_y.ndim == 1:
            new_y = new_y.reshape(-1, 1)
        if new_y.ndim != 2 or new_y.shape[1] != self.y_dim:
            raise ValueError(f"new_y must have shape (n_samples, {self.y_dim})")
        if new_x.shape[0] != new_y.shape[0]:
            raise ValueError("new_x and new_y must contain the same number of samples")
        if not new_x.shape[0]:
            return

        self.buf_x_total.append(new_x)
        self.buf_y_total.append(new_y)
        errors = np.asarray([self.error(x, y) for x, y in zip(new_x, new_y)])
        train_mask = errors > self.delta
        test_mask = ~train_mask

        if np.any(train_mask):
            self.buf_x_train.append(new_x[train_mask])
            self.buf_y_train.append(new_y[train_mask])
        if np.any(test_mask):
            self.buf_x_test.append(new_x[test_mask])
            self.buf_y_test.append(new_y[test_mask])

        self.tri, self.tree = self.train()
import numpy as np
from scipy.spatial import Delaunay, cKDTree


class Buffer:
    """
    A buffer to store and manage data points.
    """

    def __init__(self, init_data):
        """
        Initialize the buffer with initial data.

        Parameters:
        - init_data (numpy.ndarray): Initial data to store in the buffer.
        """
        self.data = init_data
        self.dim = init_data.shape[1]
        # Keep the public data array exact-sized while reserving geometric capacity.
        self._storage = init_data
        self._capacity = init_data.shape[0]

    def append(self, data):
        """
        Append new data to the buffer.

        Parameters:
        - data (numpy.ndarray): New data to append.
        """
        data = data.reshape(-1, self.dim)
        size = self.data.shape[0]
        required = size + data.shape[0]
        dtype = np.result_type(self._storage.dtype, data.dtype)

        if required > self._capacity or dtype != self._storage.dtype:
            capacity = max(required, max(1, self._capacity * 2))
            storage = np.empty((capacity, self.dim), dtype=dtype)
            storage[:size] = self.data
            self._storage = storage
            self._capacity = capacity

        self._storage[size:required] = data
        self.data = self._storage[:required]

    def error(self, x, y):
        """Return the Euclidean output prediction error for one sample."""
        return np.linalg.norm(y - self.predict(x))

    def predict(self, x):
        """Predict by simplex interpolation, falling back to nearest neighbor."""
        x = np.asarray(x)
        simplex_idx = self.tri.find_simplex(x)
        if simplex_idx == -1:
            vertex_idx = self.find_nearest_vertex_idx(x)
            return self.buf_y_train.data[vertex_idx]

        transform = self.tri.transform[simplex_idx]
        barycentric_pre = transform[:self.x_dim].dot(x - transform[self.x_dim])
        weights = np.append(barycentric_pre, 1 - barycentric_pre.sum())
        vertices = self.buf_y_train.data[self.tri.simplices[simplex_idx]]
        return self.barycentric_interpolation(vertices, weights)

    @staticmethod
    def barycentric_interpolation(vals, weights):
        """Return the weighted output value at the simplex vertices."""
        weights = np.asarray(weights).reshape(-1, 1)
        return np.sum(vals * weights, axis=0)

    def find_nearest_vertex_idx(self, x):
        """Return the index of the closest training input."""
        _, vertex_idx = self.tree.query(np.asarray(x).reshape(1, -1))
        return int(vertex_idx[0])

    def average_error(self):
        """Return mean prediction error on the test buffer."""
        x_test, y_test = self.buf_x_test.data, self.buf_y_test.data
        if not len(x_test):
            return float("nan")
        errors = [self.error(x, y) for x, y in zip(x_test, y_test)]
        return float(np.mean(errors))
