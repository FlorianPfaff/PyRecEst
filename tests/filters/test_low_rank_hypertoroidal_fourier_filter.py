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


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankHypertoroidalFourierFilter(unittest.TestCase):
    def test_rejects_sqrt_transform(self):
        with self.assertRaises(NotImplementedError):
            LowRankHypertoroidalFourierFilter((5,), "sqrt")

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


if __name__ == "__main__":
    unittest.main()
