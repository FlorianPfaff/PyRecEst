import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import estimate_transform


class TestPointSetRegistrationRankDeficiency(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_affine_fit_rejects_collinear_source_points(self):
        source = array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        target = array([[3.0, -1.0], [5.0, -2.0], [7.0, -3.0]])

        with self.assertRaisesRegex(ValueError, "affine.*underdetermined"):
            estimate_transform(source, target, model="affine")

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_affine_fit_checks_positive_weight_geometry(self):
        source = array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
        )
        target = source + array([4.0, -2.0])
        weights = array([1.0, 1.0, 1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "affine.*underdetermined"):
            estimate_transform(
                source,
                target,
                model="affine",
                weights=weights,
            )


if __name__ == "__main__":
    unittest.main()
