import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
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

    def test_value_and_pdf_match_dense_1d(self):
        dense = HypertoroidalFourierDistribution(_identity_coefficients_1d(), "identity")
        low_rank = LowRankHypertoroidalFourierDistribution.from_dense(dense)
        xs = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
        npt.assert_allclose(low_rank.value(xs), dense.value(xs), atol=1e-10)
        npt.assert_allclose(low_rank.pdf(xs), dense.pdf(xs), atol=1e-10)

    def test_shift_matches_dense_2d(self):
        dense = HypertoroidalFourierDistribution(_identity_coefficients_2d(), "identity")
        low_rank = LowRankHypertoroidalFourierDistribution.from_dense(dense)
        shift = np.array([0.2, -0.5])
        npt.assert_allclose(
            low_rank.shift(shift).to_dense(), dense.shift(shift).coeff_mat, atol=1e-10
        )

    def test_predict_additive_noise_matches_dense_2d(self):
        prior_dense = HypertoroidalFourierDistribution(_identity_coefficients_2d(), "identity")
        noise_dense = HypertoroidalFourierDistribution(_identity_coefficients_2d(), "identity")
        predicted_dense = prior_dense.convolve(noise_dense)
        predicted_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).convolve(LowRankHypertoroidalFourierDistribution.from_dense(noise_dense))
        npt.assert_allclose(predicted_low_rank.to_dense(), predicted_dense.coeff_mat, atol=1e-10)

    def test_update_multiply_matches_dense_1d(self):
        prior_dense = HypertoroidalFourierDistribution(_identity_coefficients_1d(), "identity")
        likelihood_dense = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        ).shift(np.array([1.5]))
        updated_dense = prior_dense.multiply(likelihood_dense)
        updated_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).multiply(LowRankHypertoroidalFourierDistribution.from_dense(likelihood_dense))
        npt.assert_allclose(updated_low_rank.to_dense(), updated_dense.coeff_mat, atol=1e-10)

    def test_high_dimensional_uniform_smoke(self):
        dist = LowRankHypertoroidalFourierDistribution.uniform((3,) * 8)
        self.assertEqual(dist.coeff_shape, (3,) * 8)
        self.assertEqual(dist.tt_ranks, (1,) * 9)
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-12)
        self.assertTrue(np.isfinite(dist.pdf(np.zeros(8))))


if __name__ == "__main__":
    unittest.main()
