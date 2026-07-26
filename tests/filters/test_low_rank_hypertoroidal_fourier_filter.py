import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.filters.hypertoroidal_fourier_filter import HypertoroidalFourierFilter
from pyrecest.filters.low_rank_hypertoroidal_fourier_filter import (
    LowRankHypertoroidalFourierFilter,
)


def _identity_coefficients_1d(scale=1.0):
    coeff = np.zeros(5, dtype=np.complex128)
    coeff[2] = 1.0 / (2.0 * np.pi)
    coeff[1] = scale * (0.01 + 0.02j)
    coeff[3] = np.conjugate(coeff[1])
    return coeff


def _separable_identity_coefficients_2d(scale=1.0):
    coeff_1d = np.zeros(3, dtype=np.complex128)
    coeff_1d[1] = 1.0 / (2.0 * np.pi)
    coeff_1d[0] = scale * (0.005 + 0.002j)
    coeff_1d[2] = np.conjugate(coeff_1d[0])
    return np.outer(coeff_1d, coeff_1d)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankHypertoroidalFourierFilter(unittest.TestCase):
    def test_rejects_sqrt_transform(self):
        with self.assertRaises(NotImplementedError):
            LowRankHypertoroidalFourierFilter((5,), "sqrt")

    def test_rejects_non_integral_coefficient_counts(self):
        invalid_counts = [
            True,
            np.bool_(True),
            5.0,
            np.float64(5.0),
            "5",
            (True,),
            (np.bool_(True),),
            (5.0,),
            (np.float64(5.0),),
            ("5",),
        ]
        for n_coefficients in invalid_counts:
            with self.subTest(n_coefficients=n_coefficients):
                with self.assertRaises((TypeError, ValueError)):
                    LowRankHypertoroidalFourierFilter(n_coefficients, "identity")

    def test_stores_truncation_controls(self):
        low_rank_filter = LowRankHypertoroidalFourierFilter(
            (5,), "identity", max_rank=2, rtol=1e-8, atol=1e-12
        )
        self.assertEqual(low_rank_filter.max_rank, 2)
        self.assertEqual(low_rank_filter.rtol, 1e-8)
        self.assertEqual(low_rank_filter.atol, 1e-12)

    def test_predict_identity_matches_dense_1d(self):
        dense_filter = HypertoroidalFourierFilter((5,), "identity")
        low_rank_filter = LowRankHypertoroidalFourierFilter((5,), "identity")
        prior = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        noise = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(0.5), "identity"
        )
        dense_filter.filter_state = prior
        low_rank_filter.filter_state = prior
        dense_filter.predict_identity(noise)
        low_rank_filter.predict_identity(noise)
        npt.assert_allclose(
            low_rank_filter.filter_state.to_dense(),
            dense_filter.filter_state.coeff_mat,
            atol=1e-10,
        )

    def test_update_identity_matches_dense_1d(self):
        dense_filter = HypertoroidalFourierFilter((5,), "identity")
        low_rank_filter = LowRankHypertoroidalFourierFilter((5,), "identity")
        prior = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        noise = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(0.5), "identity"
        )
        dense_filter.filter_state = prior
        low_rank_filter.filter_state = prior
        dense_filter.update_identity(noise, np.array([1.5]))
        low_rank_filter.update_identity(noise, np.array([1.5]))
        npt.assert_allclose(
            low_rank_filter.filter_state.to_dense(),
            dense_filter.filter_state.coeff_mat,
            atol=1e-10,
        )

    def test_update_identity_accepts_scalar_1d_measurement(self):
        vector_filter = LowRankHypertoroidalFourierFilter((5,), "identity")
        scalar_filter = LowRankHypertoroidalFourierFilter((5,), "identity")
        prior = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(), "identity"
        )
        noise = HypertoroidalFourierDistribution(
            _identity_coefficients_1d(0.5), "identity"
        )
        vector_filter.filter_state = prior
        scalar_filter.filter_state = prior

        vector_filter.update_identity(noise, np.array([1.5]))
        scalar_filter.update_identity(noise, 1.5)

        npt.assert_allclose(
            scalar_filter.filter_state.to_dense(),
            vector_filter.filter_state.to_dense(),
            atol=1e-10,
        )

    def test_predict_identity_respects_max_rank_and_normalization(self):
        low_rank_filter = LowRankHypertoroidalFourierFilter(
            (3, 3), "identity", max_rank=1, rtol=0.0, atol=0.0
        )
        low_rank_filter.filter_state = HypertoroidalFourierDistribution(
            _separable_identity_coefficients_2d(), "identity"
        )
        noise = HypertoroidalFourierDistribution(
            _separable_identity_coefficients_2d(0.5), "identity"
        )

        low_rank_filter.predict_identity(noise)

        self.assertLessEqual(max(low_rank_filter.filter_state.tt_ranks), 1)
        npt.assert_allclose(low_rank_filter.filter_state.integrate(), 1.0, atol=1e-10)

    def test_update_identity_respects_max_rank_and_normalization(self):
        low_rank_filter = LowRankHypertoroidalFourierFilter(
            (3, 3), "identity", max_rank=1, rtol=0.0, atol=0.0
        )
        low_rank_filter.filter_state = HypertoroidalFourierDistribution(
            _separable_identity_coefficients_2d(), "identity"
        )
        noise = HypertoroidalFourierDistribution(
            _separable_identity_coefficients_2d(0.5), "identity"
        )

        low_rank_filter.update_identity(noise, np.array([0.3, -0.2]))

        self.assertLessEqual(max(low_rank_filter.filter_state.tt_ranks), 1)
        npt.assert_allclose(low_rank_filter.filter_state.integrate(), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
