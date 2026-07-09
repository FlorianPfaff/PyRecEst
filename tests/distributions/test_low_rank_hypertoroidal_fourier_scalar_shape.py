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


def _coefficients_1d():
    coeff = np.zeros(5, dtype=np.complex128)
    coeff[2] = 1.0 / (2.0 * np.pi)
    coeff[1] = 0.01 + 0.02j
    coeff[3] = np.conjugate(coeff[1])
    return coeff


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="Low-rank Fourier prototype is NumPy-only",
)
class TestLowRankHypertoroidalFourierScalarShape(unittest.TestCase):
    def test_scalar_target_shape_matches_dense_multiply_and_convolve(self):
        prior_dense = HypertoroidalFourierDistribution(_coefficients_1d(), "identity")
        other_dense = HypertoroidalFourierDistribution(
            _coefficients_1d(), "identity"
        ).shift(np.array([0.25]))
        prior_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(prior_dense)
        other_low_rank = LowRankHypertoroidalFourierDistribution.from_dense(other_dense)

        updated_dense = prior_dense.multiply(other_dense, n_coefficients=3)
        updated_low_rank = prior_low_rank.multiply(other_low_rank, n_coefficients=3)
        self.assertEqual(updated_low_rank.coeff_shape, (3,))
        npt.assert_allclose(
            updated_low_rank.to_dense(), updated_dense.coeff_mat, atol=1e-10
        )

        predicted_dense = prior_dense.convolve(other_dense, n_coefficients=5)
        predicted_low_rank = prior_low_rank.convolve(other_low_rank, n_coefficients=5)
        npt.assert_allclose(
            predicted_low_rank.to_dense(), predicted_dense.coeff_mat, atol=1e-10
        )
