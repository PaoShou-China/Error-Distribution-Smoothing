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

    def append(self, data):
        """
        Append new data to the buffer.

        Parameters:
        - data (numpy.ndarray): New data to append.
        """
        data = data.reshape(-1, self.dim)
        self.data = np.concatenate((self.data, data))


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
        self.buf_x_total.append(new_x)
        self.buf_y_total.append(new_y)
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
        Calculate the prediction error.

        Parameters:
        - x (numpy.ndarray): Input data point.
        - y (numpy.ndarray): Output data point.

        Returns:
        - float: Prediction error.
        """
        error = np.linalg.norm(y - self.predict(x))
        return error

    def predict(self, x):
        """
        Predict using the linear model.

        Parameters:
        - x (numpy.ndarray): Input data point.

        Returns:
        - numpy.ndarray: Predicted output.
        """
        simplex_idx = self.tri.find_simplex(x)
        if simplex_idx == -1:
            vertex_idx = self.find_nearest_vertex_idx(x)
            y_hat = self.buf_y_train.data[vertex_idx]
        else:
            barycentric_coords_pre = self.tri.transform[simplex_idx, :self.x_dim].dot(
                (x - self.tri.transform[simplex_idx, self.x_dim]).T
            )
            barycentric_coords = np.append(
                barycentric_coords_pre.T, 1 - barycentric_coords_pre.sum(axis=0)
            )
            y_hat = self.barycentric_interpolation(
                self.buf_y_train.data[self.tri.simplices[simplex_idx]],
                barycentric_coords
            )
        return y_hat

    @staticmethod
    def barycentric_interpolation(vals, weights):
        """
        Perform barycentric interpolation.

        Parameters:
        - vals (numpy.ndarray): Values at the vertices.
        - weights (numpy.ndarray): Barycentric coordinates.

        Returns:
        - numpy.ndarray: Interpolated value.
        """
        weights = weights.reshape(-1, 1)
        y_hat = np.sum(vals * weights, axis=0)
        return y_hat

    def find_nearest_vertex_idx(self, x):
        """
        Find the nearest vertex to the given point.

        Parameters:
        - x (numpy.ndarray): Input data point.

        Returns:
        - int: Index of the nearest vertex.
        """
        _, vertex_idx = self.tree.query(x.reshape(1, -1))
        return vertex_idx

    def average_error(self):
        """
        Calculate the average error on the test set.

        Returns:
        - float: Average error.
        """
        x_test, y_test = self.buf_x_test.data, self.buf_y_test.data
        errors = [self.error(x, y) for x, y in zip(x_test, y_test)]
        return sum(errors) / len(errors)
