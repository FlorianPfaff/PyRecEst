import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import MerweScaledSigmaPoints


@unittest.skipUnless(
    __backend_name__ == "numpy",
    reason="Regression targets NumPy float64 rounding of tightly spaced sigma points",
)
class TestMerweSigmaPointRoundingSymmetry(unittest.TestCase):
    def test_small_alpha_preserves_representable_pair_symmetry(self):
        mean = np.array([0.5])
        covariance = np.array([[0.7]])
        points = MerweScaledSigmaPoints(
            n=1,
            alpha=1.0e-3,
            beta=2.0,
            kappa=0.0,
        )

        sigmas = to_numpy(points.sigma_points(asarray(mean), asarray(covariance)))
        positive_offset = sigmas[1] - sigmas[0]
        negative_offset = sigmas[2] - sigmas[0]

        npt.assert_array_equal(negative_offset, -positive_offset)
        npt.assert_array_equal(to_numpy(points.Wm) @ sigmas, mean)


if __name__ == "__main__":
    unittest.main()
