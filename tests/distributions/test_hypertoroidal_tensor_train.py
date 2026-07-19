import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.distributions.hypertorus._tensor_train import TensorTrain


class TestHypertoroidalTensorTrain(unittest.TestCase):
    def test_dense_roundtrip_and_entry_access(self):
        tensor = np.arange(27, dtype=float).reshape(3, 3, 3) + 1j
        tt = TensorTrain.from_dense(tensor)
        npt.assert_allclose(tt.to_dense(), tensor, atol=1e-12)
        self.assertEqual(tt.shape, (3, 3, 3))
        npt.assert_allclose(tt.entry((1, 2, 0)), tensor[1, 2, 0], atol=1e-12)

    def test_frobenius_norm(self):
        tensor = np.arange(9, dtype=float).reshape(3, 3) - 2.0
        tt = TensorTrain.from_dense(tensor)
        npt.assert_allclose(tt.norm_squared(), np.vdot(tensor, tensor).real, atol=1e-12)

    def test_hadamard_product_matches_dense(self):
        left = np.arange(9, dtype=float).reshape(3, 3)
        right = np.arange(9, dtype=float).reshape(3, 3) + 1.0
        product = TensorTrain.from_dense(left).hadamard_product(
            TensorTrain.from_dense(right)
        )
        npt.assert_allclose(product.to_dense(), left * right, atol=1e-12)

    def test_centered_coefficient_convolution_matches_dense_1d(self):
        left = np.array([1.0, 2.0, 3.0])
        right = np.array([0.5, 1.0, -0.5])
        result = TensorTrain.from_dense(left).coefficient_convolution(
            TensorTrain.from_dense(right), target_shape=(3,)
        )
        expected = np.convolve(left, right, mode="same")
        npt.assert_allclose(result.to_dense(), expected, atol=1e-12)

    def test_round_without_truncation_preserves_tensor(self):
        rng = np.random.default_rng(0)
        tensor = rng.normal(size=(3, 4, 2)) + 1j * rng.normal(size=(3, 4, 2))
        tt = TensorTrain.from_dense(tensor)
        rounded = tt.round()
        npt.assert_allclose(rounded.to_dense(), tensor, atol=1e-12)
        npt.assert_allclose(rounded.norm_squared(), tt.norm_squared(), atol=1e-12)

    def test_round_truncates_ranks_without_dense_guard(self):
        rng = np.random.default_rng(1)
        tensor = rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3))
        tt = TensorTrain.from_dense(tensor)
        rounded = tt.round(max_rank=1, max_dense_entries=1)
        self.assertEqual(rounded.ranks, (1, 1, 1, 1))
        self.assertEqual(rounded.shape, tt.shape)
        self.assertTrue(np.isfinite(rounded.norm_squared()))

    def test_centered_hermitian_symmetry_detection(self):
        coeff = np.zeros((3, 3), dtype=np.complex128)
        coeff[1, 1] = 1.0
        coeff[0, 1] = 0.2 + 0.1j
        coeff[2, 1] = np.conjugate(coeff[0, 1])
        coeff[1, 0] = -0.3 + 0.05j
        coeff[1, 2] = np.conjugate(coeff[1, 0])
        tt = TensorTrain.from_dense(coeff)
        self.assertTrue(tt.is_centered_hermitian())
        npt.assert_allclose(tt.centered_hermitian_deviation(), 0.0, atol=1e-12)

        broken = coeff.copy()
        broken[2, 1] += 0.1
        broken_tt = TensorTrain.from_dense(broken)
        self.assertFalse(broken_tt.is_centered_hermitian(atol=1e-12))
        self.assertGreater(broken_tt.centered_hermitian_deviation(), 0.0)

    def test_centered_hermitianized_repairs_dense_average(self):
        coeff = np.zeros(5, dtype=np.complex128)
        coeff[2] = 1.0
        coeff[1] = 0.2 + 0.1j
        coeff[3] = 0.3 - 0.2j
        repaired = TensorTrain.from_dense(coeff).centered_hermitianized()
        self.assertTrue(repaired.is_centered_hermitian())
        dense = repaired.to_dense()
        npt.assert_allclose(dense[1], np.conjugate(dense[3]), atol=1e-12)
        npt.assert_allclose(dense[2].imag, 0.0, atol=1e-12)

    def test_centered_hermitian_validation_obeys_max_entries_guard(self):
        tt = TensorTrain.from_dense(np.zeros((3, 3), dtype=np.complex128))
        with self.assertRaises(ValueError):
            tt.centered_hermitian_deviation(max_entries=1)
        with self.assertRaises(ValueError):
            tt.centered_hermitianized(max_entries=1)

    def test_max_rank_requires_positive_integer(self):
        tensor = np.arange(8, dtype=float).reshape(2, 2, 2)
        tt = TensorTrain.from_dense(tensor)
        for invalid in (True, np.bool_(False), 1.5, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    TensorTrain.from_dense(tensor, max_rank=invalid)
                with self.assertRaises(TypeError):
                    tt.round(max_rank=invalid)
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    TensorTrain.from_dense(tensor, max_rank=invalid)
                with self.assertRaises(ValueError):
                    tt.round(max_rank=invalid)

    def test_tolerances_require_nonnegative_finite_scalars(self):
        tensor = np.arange(8, dtype=float).reshape(2, 2, 2)
        tt = TensorTrain.from_dense(tensor)

        for keyword in ("rtol", "atol"):
            for invalid in (True, np.bool_(False), "0.1", [0.1], 1.0 + 0.0j):
                with self.subTest(keyword=keyword, invalid=invalid):
                    with self.assertRaises(TypeError):
                        TensorTrain.from_dense(tensor, **{keyword: invalid})
                    with self.assertRaises(TypeError):
                        tt.round(**{keyword: invalid})

            for invalid in (-1.0, np.inf, np.nan):
                with self.subTest(keyword=keyword, invalid=invalid):
                    with self.assertRaises(ValueError):
                        TensorTrain.from_dense(tensor, **{keyword: invalid})
                    with self.assertRaises(ValueError):
                        tt.round(**{keyword: invalid})


if __name__ == "__main__":
    unittest.main()
