import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import JulierSigmaPoints, MerweScaledSigmaPoints


@unittest.skipUnless(
    __backend_name__ == "numpy",
    reason="Regression targets float64 sigma-point rounding symmetry",
)
class TestSigmaPointPairSymmetry(unittest.TestCase):
    def test_small_spread_pairs_are_exact_reflections_of_the_mean(self):
        mean = asarray(np.array([0.5]))
        covariance = asarray(np.array([[0.7]]))
        generators = (
            MerweScaledSigmaPoints(n=1, alpha=1.0e-3, beta=2.0, kappa=0.0),
            JulierSigmaPoints(n=1, kappa=-0.999999),
        )

        for generator in generators:
            with self.subTest(generator=type(generator).__name__):
                sigmas = to_numpy(generator.sigma_points(mean, covariance))
                positive_offset = sigmas[1] - sigmas[0]
                negative_offset = sigmas[0] - sigmas[2]
                npt.assert_array_equal(positive_offset, negative_offset)


if __name__ == "__main__":
    unittest.main()
