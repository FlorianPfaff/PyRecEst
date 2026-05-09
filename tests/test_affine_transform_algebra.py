import unittest

import numpy.testing as npt
from pyrecest.backend import __backend_name__, array
from pyrecest.utils.point_set_registration import AffineTransform

_ATOL = 1e-6 if __backend_name__ == "jax" else 1e-10


class TestAffineTransformAlgebra(unittest.TestCase):
    def test_inverse_round_trips_points(self):
        points = array([[0.0, 0.0], [1.0, -2.0], [3.0, 4.0]])
        transform = AffineTransform(
            array([[1.2, 0.3], [-0.4, 0.8]]),
            array([2.0, -1.5]),
        )

        recovered = transform.inverse().apply(transform.apply(points))

        npt.assert_allclose(recovered, points, atol=_ATOL)

    def test_compose_matches_sequential_application(self):
        points = array([[0.0, 0.0], [1.0, -2.0], [3.0, 4.0]])
        first = AffineTransform(
            array([[0.8, -0.1], [0.2, 1.1]]),
            array([1.0, 0.5]),
        )
        second = AffineTransform(
            array([[1.3, 0.4], [-0.2, 0.7]]),
            array([-2.0, 1.5]),
        )

        composed = second.compose(first)

        npt.assert_allclose(
            composed.apply(points),
            second.apply(first.apply(points)),
            atol=_ATOL,
        )

    def test_compose_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            AffineTransform.identity(2).compose(AffineTransform.identity(3))

    def test_compose_rejects_non_transform(self):
        with self.assertRaises(TypeError):
            AffineTransform.identity(2).compose(object())


if __name__ == "__main__":
    unittest.main()
