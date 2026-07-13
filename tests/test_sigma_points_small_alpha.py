import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import MerweScaledSigmaPoints


@unittest.skipIf(
    __backend_name__ == "pytorch",
    reason="Sigma-point tests use NumPy assertions and the PyTorch backend is unsupported",
)
class TestMerweSmallAlpha(unittest.TestCase):
    def test_small_positive_alpha_does_not_cancel_scale_to_zero(self):
        points = MerweScaledSigmaPoints(n=1, alpha=1.0e-9, beta=2.0, kappa=0.0)

        sigmas = points.sigma_points(asarray([0.0]), asarray([[1.0]]))

        self.assertTrue(np.all(np.isfinite(to_numpy(points.Wm))))
        self.assertTrue(np.all(np.isfinite(to_numpy(points.Wc))))
        npt.assert_allclose(
            to_numpy(sigmas),
            np.array([[0.0], [1.0e-9], [-1.0e-9]]),
            rtol=1.0e-12,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
