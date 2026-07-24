import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.distributions.hypertorus._tensor_train import TensorTrain
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)


def _identity_coefficients_1d():
    coeff = np.zeros(5, dtype=np.complex128)
    coeff[2] = 1.0 / (2.0 * np.pi)
    coeff[1] = 0.01 + 0.02j
    coeff[3] = np.conjugate(coeff[1])
    return coeff


def _identity_coefficients_2d():
    coeff = np.zeros((3, 3), dtype=np.complex128)
    coeff[1, 1] = 1.0 / (2.0 * np.pi) ** 2
    coeff[0, 1] = 0.005 + 0.002j
    coeff[2, 1] = np.conjugate(coeff[0, 1])
    coeff[1, 0] = -0.004 + 0.003j
    coeff[1, 2] = np.conjugate(coeff[1, 0])
    return coeff


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankHypertoroidalFourierDistribution(unittest.TestCase):
    def test_uniform_identity_normalization(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform((3, 3, 3))
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-12)
        npt.assert_allclose(
            dist.coefficient_at_zero(), 1.0 / (2.0 * np.pi) ** 3, atol=1e-12
        )

    def test_accepts_numpy_integer_shape(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform(np.int64(3))
        self.assertEqual(dist.coeff_shape, (3,))

    def test_rejects_non_integral_shape_entries(self):
        invalid_shapes = [
            True,
            np.bool_(True),
            3.0,
            np.float64(3.0),
            "3",
            (True,),
            (np.bool_(True),),
            (3.0,),
            (np.float64(3.0),),
            ("3",),
        ]
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaises((TypeError, ValueError)):
                    LowRankHypertoroidalFourierDistribution.uniform(shape)

    def test_value_and_pdf_match_dense_1d(self):
        dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        low_rank = LowRankHypertoroidalFourierDistribution.from_dense(dense)
        xs = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
        npt.assert_allclose(low_rank.value(xs), dense.value(xs), atol=1e-10)
        npt.assert_allclose(low_rank.pdf(xs), dense.pdf(xs), atol=1e-10)

    def test_value_rejects_invalid_point_rank(self):
        low_rank_1d = LowRankHypertoroidalFourierDistribution.from_dense(
            HypertoroidalFourierDistribution(_identity_coefficients_1d(), "identity")
        )
        with self.assertRaisesRegex(ValueError, "scalar, vector, or column"):
            low_rank_1d.value(np.zeros((2, 2, 1)))

        low_rank_2d = LowRankHypertoroidalFourierDistribution.from_dense(
            HypertoroidalFourierDistribution(_identity_coefficients_2d(), "identity")
        )
        for points in (0.0, np.zeros((2, 2, 2))):
            with self.subTest(points_shape=np.shape(points)):
                with self.assertRaisesRegex(ValueError, "shape \\(2,\\) or \\(n, 2\\)"):
                    low_rank_2d.value(points)

    def test_shift_matches_dense_2d(self):
        dense = HypertoroidalFourierDistribution(
            _identity_coefficients_2d(), "identity"
        )
        low_rank = LowRankHypertoroidalFourierDistribution.from_dense(dense)
        shift = np.array([0.2, -0.5])
        npt.assert_allclose(
            low_rank.shift(shift).to_dense(), dense.shift(shift).coeff_mat, atol=1e-10
        )

    def test_predict_additive_noise_matches_dense_2d(self):
        prior_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_2d(), "identity"
        )
        noise_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_2d(), "identity"
        )
        predicted_dense = prior_dense.convolve(noise_dense)
        predicted_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).convolve(LowRankHypertoroidalFourierDistribution.from_dense(noise_dense))
        npt.assert_allclose(
            predicted_low_rank.to_dense(), predicted_dense.coeff_mat, atol=1e-10
        )

    def test_update_multiply_matches_dense_1d(self):
        prior_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        likelihood_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        ).shift(np.array([1.5]))
        updated_dense = prior_dense.multiply(likelihood_dense)
        updated_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).multiply(LowRankHypertoroidalFourierDistribution.from_dense(likelihood_dense))
        npt.assert_allclose(
            updated_low_rank.to_dense(), updated_dense.coeff_mat, atol=1e-10
        )

    def test_high_dimensional_uniform_smoke(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform((3,) * 8)
        self.assertEqual(dist.coeff_shape, (3,) * 8)
        self.assertEqual(dist.tt_ranks, (1,) * 9)
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-12)
        self.assertTrue(np.isfinite(dist.pdf(np.zeros(8))))

    def test_hermitian_helpers_delegate_to_coefficients(self):
        dist = LowRankHypertoroidalFourierDistribution.from_dense(
            HypertoroidalFourierDistribution(_identity_coefficients_2d(), "identity")
        )
        self.assertTrue(dist.is_centered_hermitian())
        npt.assert_allclose(dist.centered_hermitian_deviation(), 0.0, atol=1e-12)

    def test_centered_hermitianized_repairs_and_normalizes_distribution(self):
        coeff = _identity_coefficients_1d()
        coeff[3] += 0.1
        dist = LowRankHypertoroidalFourierDistribution.from_dense(
            HypertoroidalFourierDistribution(coeff, "identity")
        )
        self.assertFalse(dist.is_centered_hermitian(atol=1e-12))

        repaired = dist.centered_hermitianized()

        self.assertTrue(repaired.is_centered_hermitian())
        npt.assert_allclose(repaired.integrate(), 1.0, atol=1e-12)

    def test_hermitian_helpers_reject_sqrt_transformation(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform(
            (3,), transformation="sqrt"
        )
        with self.assertRaises(NotImplementedError):
            dist.centered_hermitian_deviation()
        with self.assertRaises(NotImplementedError):
            dist.is_centered_hermitian()
        with self.assertRaises(NotImplementedError):
            dist.centered_hermitianized()

    def test_tensor_train_from_dense_validates_max_rank_before_one_dimensional_return(
        self,
    ):
        for max_rank in (0, -1, np.array(0)):
            with self.subTest(max_rank=repr(max_rank)):
                with self.assertRaises(ValueError):
                    TensorTrain.from_dense(np.ones(5), max_rank=max_rank)

        for max_rank in (True, False, np.bool_(True), np.array(True), 1.5, "1"):
            with self.subTest(max_rank=repr(max_rank)):
                with self.assertRaises(TypeError):
                    TensorTrain.from_dense(np.ones(5), max_rank=max_rank)

    def test_tensor_train_round_validates_max_rank_before_one_dimensional_return(self):
        tensor_train = TensorTrain.from_dense(np.ones(5))
        for max_rank in (0, -1, np.array(0)):
            with self.subTest(max_rank=repr(max_rank)):
                with self.assertRaises(ValueError):
                    tensor_train.round(max_rank=max_rank)

        for max_rank in (True, False, np.bool_(True), np.array(True), 1.5, "1"):
            with self.subTest(max_rank=repr(max_rank)):
                with self.assertRaises(TypeError):
                    tensor_train.round(max_rank=max_rank)

    def test_tensor_train_accepts_integer_scalar_array_max_rank(self):
        tensor_train = TensorTrain.from_dense(np.eye(3), max_rank=np.array(1))

        self.assertEqual(tensor_train.ranks, (1, 1, 1))

    def test_update_multiply_accepts_scalar_1d_coefficient_count(self):
        prior_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        likelihood_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        ).shift(np.array([1.5]))
        updated_dense = prior_dense.multiply(likelihood_dense, n_coefficients=5)
        updated_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).multiply(
            LowRankHypertoroidalFourierDistribution.from_dense(likelihood_dense),
            n_coefficients=5,
        )
        npt.assert_allclose(
            updated_low_rank.to_dense(), updated_dense.coeff_mat, atol=1e-10
        )

    def test_predict_convolve_accepts_scalar_1d_coefficient_count(self):
        prior_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        noise_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        predicted_dense = prior_dense.convolve(noise_dense)
        predicted_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).convolve(
            LowRankHypertoroidalFourierDistribution.from_dense(noise_dense),
            n_coefficients=5,
        )
        npt.assert_allclose(
            predicted_low_rank.to_dense(), predicted_dense.coeff_mat, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
