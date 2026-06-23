import unittest

import numpy as np
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import estimate_transform


class TestPointSetRegistrationFinitePoints(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_nonfinite_source_and_target_points(self):
        finite_points = array([[0.0, 0.0], [1.0, 1.0]])
        nonfinite_points = array([[0.0, 0.0], [1.0, np.nan]])

        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_transform(nonfinite_points, finite_points, model="translation")
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_transform(finite_points, nonfinite_points, model="translation")


if __name__ == "__main__":
    unittest.main()
