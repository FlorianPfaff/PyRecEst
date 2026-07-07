import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


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
        dense = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(1.0), np.array(0.6)),
            (9,),
            "identity",
        )
        low_rank = LowRankHypertoroidalFourierDistribution.from_dense(dense)
        xs = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
        npt.assert_allclose(low_rank.value(xs), dense.value(xs), atol=1e-10)
        npt.assert_allclose(low_rank.pdf(xs), dense.pdf(xs), atol=1e-10)

    def test_shift_matches_dense_2d(self):
        dense = HypertoroidalFourierDistribution.from_distribution(
            ToroidalWrappedNormalDistribution(np.array([1.0, 2.0]), 0.4 * np.eye(2)),
            (7, 7),
            "identity",
        )
        low_rank = LowRankHypertoroidalFourierDistribution.from_dense(dense)
        shift = np.array([0.2, -0.5])
        npt.assert_allclose(
            low_rank.shift(shift).to_dense(), dense.shift(shift).coeff_mat, atol=1e-10
        )

    def test_predict_additive_noise_matches_dense_2d(self):
        prior_dense = HypertoroidalFourierDistribution.from_distribution(
            ToroidalWrappedNormalDistribution(np.array([1.0, 2.0]), 0.4 * np.eye(2)),
            (7, 7),
            "identity",
        )
        noise_dense = HypertoroidalFourierDistribution.from_distribution(
            ToroidalWrappedNormalDistribution(np.zeros(2), 0.2 * np.eye(2)),
            (7, 7),
            "identity",
        )
        predicted_dense = prior_dense.convolve(noise_dense)
        predicted_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(
            prior_dense
        ).convolve(LowRankHypertoroidalFourierDistribution.from_dense(noise_dense))
        npt.assert_allclose(predicted_low_rank.to_dense(), predicted_dense.coeff_mat, atol=1e-10)

    def test_update_multiply_matches_dense_1d(self):
        prior_dense = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(1.0), np.array(0.5)),
            (9,),
            "identity",
        )
        likelihood_dense = HypertoroidalFourierDistribution.from_distribution(
            WrappedNormalDistribution(np.array(0.0), np.array(1.0)),
            (9,),
            "identity",
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
