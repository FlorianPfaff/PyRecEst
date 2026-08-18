import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.filters.hypertoroidal_fourier_filter import HypertoroidalFourierFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",  # pylint: disable=no-member
    reason="HypertoroidalFourierFilter is NumPy-only",
)
class TestHypertoroidalFourierFilterStateOwnership(unittest.TestCase):
    def test_filter_state_assignment_does_not_alias_input(self):
        fourier_filter = HypertoroidalFourierFilter((3,), "identity")
        coefficients = np.zeros(3, dtype=np.complex128)
        coefficients[1] = 1.0 / (2.0 * np.pi)
        coefficients[0] = 0.01 + 0.02j
        coefficients[2] = np.conjugate(coefficients[0])
        state = HypertoroidalFourierDistribution(coefficients, "identity")
        expected = state.coeff_mat.copy()

        fourier_filter.filter_state = state

        self.assertIsNot(fourier_filter.filter_state, state)
        state.coeff_mat[...] = 0.0
        npt.assert_allclose(fourier_filter.filter_state.coeff_mat, expected)


if __name__ == "__main__":
    unittest.main()
