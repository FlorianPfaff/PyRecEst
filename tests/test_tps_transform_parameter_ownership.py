import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.nonrigid_point_set_registration import ThinPlateSplineTransform


class TestThinPlateSplineTransformOwnership(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="JAX arrays are immutable and cannot expose caller-side aliasing.",
    )
    def test_constructor_copies_parameter_arrays(self):
        control_points = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        weights = array([[0.1, 0.0], [-0.05, 0.02], [-0.05, -0.02]])
        affine_coefficients = array([[1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        query = array([[0.25, 0.5], [0.75, 0.1]])

        transform = ThinPlateSplineTransform(
            control_points=control_points,
            weights=weights,
            affine_coefficients=affine_coefficients,
        )
        expected_control_points = np.asarray(control_points).copy()
        expected_weights = np.asarray(weights).copy()
        expected_affine_coefficients = np.asarray(affine_coefficients).copy()
        expected_output = np.asarray(transform.apply(query)).copy()

        control_points[...] = 10.0
        weights[...] = -20.0
        affine_coefficients[...] = 30.0

        npt.assert_allclose(transform.control_points, expected_control_points)
        npt.assert_allclose(transform.weights, expected_weights)
        npt.assert_allclose(transform.affine_coefficients, expected_affine_coefficients)
        npt.assert_allclose(transform.apply(query), expected_output)


if __name__ == "__main__":
    unittest.main()
