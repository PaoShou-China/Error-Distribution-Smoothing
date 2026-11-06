import unittest

import numpy as np

from manager import Buffer, Manager


class BufferTests(unittest.TestCase):
    def test_append_preserves_shape_and_promotes_dtype_for_empty_batch(self):
        buffer = Buffer(np.zeros((1, 2), dtype=np.float32))

        buffer.append(np.ones((0, 2), dtype=np.float64))

        self.assertEqual(buffer.data.shape, (1, 2))
        self.assertEqual(buffer.data.dtype, np.dtype(np.float64))

    def test_append_keeps_rows_in_order(self):
        buffer = Buffer(np.zeros((1, 2), dtype=np.float32))
        rows = np.ones((8, 2), dtype=np.float32)

        buffer.append(rows)

        self.assertEqual(buffer.data.shape, (9, 2))
        np.testing.assert_array_equal(buffer.data[1:], rows)


class ManagerBatchTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.4, 0.4],
        ])
        self.y = np.arange(5.0).reshape(-1, 1)
        self.manager = Manager(self.x, self.y, delta=1e6)

    def test_batch_update_routes_all_low_error_rows_to_test(self):
        new_x = np.array([[0.2, 0.2], [0.3, 0.3]])
        new_y = np.array([[1.0], [2.0]])

        self.manager.update(new_x, new_y)

        self.assertEqual(self.manager.buf_x_total.data.shape[0], 7)
        self.assertEqual(self.manager.buf_x_test.data.shape[0], 3)
        self.assertEqual(self.manager.buf_x_train.data.shape[0], 4)

    def test_batch_update_rejects_mismatched_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "same number of samples"):
            self.manager.update(np.array([[0.2, 0.2]]), np.array([[1.0], [2.0]]))

    def test_empty_batch_still_checks_feature_dimension(self):
        with self.assertRaisesRegex(ValueError, "new_x must have shape"):
            self.manager.update(np.empty((0, 3)), np.empty((0, 1)))


if __name__ == "__main__":
    unittest.main()
