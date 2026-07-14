import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, float64
from pyrecest.utils.point_set_registration import estimate_transform


class TestEstimateTransformExtremeWeights(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_finite_extreme_weights_preserve_relative_mass(self):
        source = array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float64,
        )
        true_offset = array([2.0, -3.0], dtype=float64)
        target = source + true_offset
        reference_weights = array([4.0, 2.0, 1.0], dtype=float64)
        extreme_weights = reference_weights * (np.finfo(np.float64).max / 4.0)

        reference = estimate_transform(
            source,
            target,
            model="translation",
            weights=reference_weights,
        )
        estimated = estimate_transform(
            source,
            target,
            model="translation",
            weights=extreme_weights,
        )

        npt.assert_allclose(estimated.matrix, reference.matrix, atol=1e-12)
        npt.assert_allclose(estimated.offset, true_offset, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
