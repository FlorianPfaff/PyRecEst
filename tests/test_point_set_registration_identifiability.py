import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import estimate_transform


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestPointSetRegistrationIdentifiability(unittest.TestCase):
    def test_affine_fit_rejects_collinear_2d_correspondences(self):
        source = array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        target = array([[3.0, -1.0], [4.0, -1.0], [5.0, -1.0]])

        with self.assertRaisesRegex(ValueError, "does not uniquely determine"):
            estimate_transform(source, target, model="affine")

    def test_rigid_fit_rejects_collinear_3d_correspondences(self):
        source = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        target = source + array([2.0, -1.0, 0.5])

        with self.assertRaisesRegex(ValueError, "does not uniquely determine"):
            estimate_transform(source, target, model="rigid")

    def test_reflection_enabled_rigid_fit_requires_full_rank_geometry(self):
        source = array([[0.0, 0.0], [1.0, 0.0]])
        target = array([[0.0, 0.0], [-1.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "does not uniquely determine"):
            estimate_transform(
                source,
                target,
                model="rigid",
                allow_reflection=True,
            )

    def test_proper_3d_rigid_fit_accepts_noncollinear_planar_points(self):
        source = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        rotation = array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        offset = array([2.0, -3.0, 0.5])
        target = (rotation @ source.T).T + offset

        estimated = estimate_transform(source, target, model="rigid")

        npt.assert_allclose(estimated.matrix, rotation, atol=1e-10)
        npt.assert_allclose(estimated.offset, offset, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
