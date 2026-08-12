import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import MerweScaledSigmaPoints


@unittest.skipUnless(
    __backend_name__ == "numpy",
    reason="Regression targets NumPy float64 overflow handling",
)
class TestMerweNegativeSigmaOverflow(unittest.TestCase):
    def test_large_finite_mean_keeps_negative_sigma_points_finite(self):
        max_float = np.finfo(np.float64).max
        mean = np.array([0.75 * max_float])
        covariance = np.array([[max_float]])
        points = MerweScaledSigmaPoints(
            n=1,
            alpha=1.0,
            beta=2.0,
            kappa=0.0,
        )

        with np.errstate(over="raise", invalid="raise"):
            sigma_points = to_numpy(
                points.sigma_points(asarray(mean), asarray(covariance))
            )

        expected_offset = np.sqrt(max_float)
        expected = np.stack(
            [
                mean,
                mean + expected_offset,
                mean - expected_offset,
            ]
        )
        self.assertTrue(np.all(np.isfinite(sigma_points)))
        npt.assert_array_equal(sigma_points, expected)


if __name__ == "__main__":
    unittest.main()
