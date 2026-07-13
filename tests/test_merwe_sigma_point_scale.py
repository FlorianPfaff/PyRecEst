import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, asarray, to_numpy
from pyrecest.sampling import MerweScaledSigmaPoints


@unittest.skipIf(
    __backend_name__ == "pytorch",
    reason="Sigma-point tests use NumPy assertions and the PyTorch backend is unsupported",
)
class TestMerweSigmaPointScale(unittest.TestCase):
    def test_small_positive_alpha_preserves_nonzero_sigma_spread(self):
        alpha = 1.0e-9
        points = MerweScaledSigmaPoints(n=2, alpha=alpha, beta=2.0, kappa=0.0)

        sigmas = to_numpy(
            points.sigma_points(asarray(np.zeros(2)), asarray(np.eye(2)))
        )
        offsets = sigmas[1:] - sigmas[0]
        expected_radius = np.sqrt(alpha**2 * 2.0)

        npt.assert_allclose(
            np.linalg.norm(offsets, axis=1),
            expected_radius,
            rtol=1.0e-6,
            atol=0.0,
        )
        self.assertTrue(np.all(np.isfinite(to_numpy(points.Wm))))
        self.assertTrue(np.all(np.isfinite(to_numpy(points.Wc))))


if __name__ == "__main__":
    unittest.main()
