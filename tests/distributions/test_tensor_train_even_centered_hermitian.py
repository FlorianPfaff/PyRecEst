import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.distributions.hypertorus._tensor_train import TensorTrain


class TestTensorTrainEvenCenteredHermitian(unittest.TestCase):
    def test_even_grid_detection_preserves_self_conjugate_bins(self):
        coefficients = np.array(
            [0.4, 0.2 + 0.1j, 1.0, 0.2 - 0.1j],
            dtype=np.complex128,
        )
        tensor_train = TensorTrain.from_dense(coefficients)

        self.assertTrue(tensor_train.is_centered_hermitian())
        npt.assert_allclose(
            tensor_train.centered_hermitian_deviation(),
            0.0,
            atol=1e-12,
        )

        for self_conjugate_index in (0, 2):
            with self.subTest(index=self_conjugate_index):
                broken = coefficients.copy()
                broken[self_conjugate_index] += 0.3j
                self.assertFalse(
                    TensorTrain.from_dense(broken).is_centered_hermitian(
                        atol=1e-12
                    )
                )

    def test_even_grid_repair_averages_true_negative_frequencies(self):
        coefficients = np.array(
            [0.4 + 0.3j, 0.2 + 0.1j, 1.0 + 0.4j, 0.5 + 0.2j],
            dtype=np.complex128,
        )
        negative_frequency_indices = np.array([0, 3, 2, 1])
        expected = 0.5 * (
            coefficients
            + np.conjugate(coefficients[negative_frequency_indices])
        )

        repaired = TensorTrain.from_dense(coefficients).centered_hermitianized()

        self.assertTrue(repaired.is_centered_hermitian())
        npt.assert_allclose(repaired.to_dense(), expected, atol=1e-12)
        npt.assert_allclose(repaired.to_dense()[[0, 2]].imag, 0.0, atol=1e-12)

    def test_mixed_even_and_odd_axes_use_axis_specific_reflections(self):
        rng = np.random.default_rng(3)
        coefficients = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
        even_negative_indices = np.array([0, 3, 2, 1])
        odd_negative_indices = np.array([2, 1, 0])
        reflected = np.conjugate(
            coefficients[np.ix_(even_negative_indices, odd_negative_indices)]
        )
        hermitian_coefficients = 0.5 * (coefficients + reflected)

        tensor_train = TensorTrain.from_dense(hermitian_coefficients)

        self.assertTrue(tensor_train.is_centered_hermitian())
        npt.assert_allclose(
            tensor_train.centered_hermitian_deviation(),
            0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
