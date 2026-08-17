"""Regression tests for scale-stable SO(3) exponential maps."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import so3_helpers


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme float64 regression is specific to the NumPy backend",
)
class SO3ExtremeTangentExpMapTest(unittest.TestCase):
    def test_exp_map_preserves_extreme_finite_tangent_direction(self):
        tangent_vector = array([[1.0e200, 1.0e200, 0.0]])

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            quaternion = so3_helpers.exp_map(tangent_vector)

        quaternion = np.asarray(to_numpy(quaternion), dtype=float)
        self.assertTrue(np.all(np.isfinite(quaternion)))
        npt.assert_allclose(
            np.linalg.norm(quaternion, axis=-1),
            np.ones(1),
            rtol=1.0e-15,
            atol=0.0,
        )
        npt.assert_allclose(
            quaternion[:, 0],
            quaternion[:, 1],
            rtol=1.0e-15,
            atol=0.0,
        )
        npt.assert_array_equal(quaternion[:, 2], np.zeros(1))
        self.assertGreaterEqual(quaternion[0, 3], 0.0)


if __name__ == "__main__":
    unittest.main()
