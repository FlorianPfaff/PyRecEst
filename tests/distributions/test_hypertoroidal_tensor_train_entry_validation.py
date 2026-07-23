import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.distributions.hypertorus._tensor_train import TensorTrain


class TestHypertoroidalTensorTrainEntryValidation(unittest.TestCase):
    def setUp(self):
        self.tensor = np.arange(9, dtype=float).reshape(3, 3)
        self.tt = TensorTrain.from_dense(self.tensor)

    def test_entry_accepts_numpy_integer_and_negative_indices(self):
        npt.assert_allclose(
            self.tt.entry((np.int64(1), -1)), self.tensor[1, -1], atol=1e-12
        )

    def test_entry_rejects_non_integer_indices_without_coercion(self):
        invalid_indices = [
            (0.0, 1),
            (1.5, 1),
            ("1", 1),
            (True, 1),
            (np.bool_(False), 1),
        ]
        for multi_index in invalid_indices:
            with self.subTest(multi_index=multi_index):
                with self.assertRaises(TypeError):
                    self.tt.entry(multi_index)

    def test_entry_reports_out_of_bounds_axis(self):
        with self.assertRaisesRegex(IndexError, "axis 1"):
            self.tt.entry((0, 3))


if __name__ == "__main__":
    unittest.main()
