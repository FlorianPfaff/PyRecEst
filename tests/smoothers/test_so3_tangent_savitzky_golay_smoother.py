import math
import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, cos, eye, sin, stack
from pyrecest.smoothers import SO3TangentSavitzkyGolaySmoother, SO3TSGSmoother

ATOL = 1e-6


def z_rotation(angle):
    return array(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def geodesic_distance(rotation_a, rotation_b):
    relative = rotation_a @ rotation_b.T
    trace = relative[0, 0] + relative[1, 1] + relative[2, 2]
    cosine = min(max(float(0.5 * (trace - 1.0)), -1.0), 1.0)
    return math.acos(cosine)


class SO3TangentSavitzkyGolaySmootherTest(unittest.TestCase):
    def test_bridges_single_frame_occlusion_on_linear_z_sequence(self):
        rotations = [z_rotation(0.1 * idx) for idx in range(5)]
        mask = [True, True, False, True, True]
        smoother = SO3TangentSavitzkyGolaySmoother(window_size=5, polynomial_degree=1)

        smoothed = smoother.smooth(rotations, mask=mask)

        self.assertLess(geodesic_distance(smoothed[2], z_rotation(0.2)), ATOL)

    def test_product_smoother_applies_component_masks(self):
        first_component = [z_rotation(0.1 * idx) for idx in range(5)]
        second_component = [eye(3) for _ in range(5)]
        rotations = stack(
            [
                stack([first_component[idx], second_component[idx]], axis=0)
                for idx in range(5)
            ],
            axis=0,
        )
        mask = array(
            [
                [True, True],
                [True, True],
                [False, True],
                [True, True],
                [True, True],
            ]
        )
        smoother = SO3TangentSavitzkyGolaySmoother(window_size=5, polynomial_degree=1)

        smoothed = smoother.smooth_product(rotations, mask=mask)

        self.assertEqual(smoothed.shape, rotations.shape)
        self.assertLess(geodesic_distance(smoothed[2, 0], z_rotation(0.2)), ATOL)
        npt.assert_allclose(smoothed[:, 1], rotations[:, 1], atol=ATOL)

    def test_validates_mask_and_parameters(self):
        smoother = SO3TangentSavitzkyGolaySmoother()

        with self.assertRaises(ValueError):
            SO3TangentSavitzkyGolaySmoother(window_size=4)
        for invalid_window in (
            True,
            np.bool_(True),
            5.5,
            np.array(5.5),
            np.array([5]),
            0,
        ):
            with self.subTest(window_size=invalid_window):
                with self.assertRaisesRegex(ValueError, "window_size"):
                    SO3TangentSavitzkyGolaySmoother(window_size=invalid_window)

        with self.assertRaises(ValueError):
            SO3TangentSavitzkyGolaySmoother(polynomial_degree=-1)
        for invalid_degree in (
            True,
            np.bool_(True),
            1.5,
            np.array(1.5),
            np.array([1]),
            -1,
        ):
            with self.subTest(polynomial_degree=invalid_degree):
                with self.assertRaisesRegex(ValueError, "polynomial_degree"):
                    SO3TangentSavitzkyGolaySmoother(polynomial_degree=invalid_degree)

        valid = SO3TangentSavitzkyGolaySmoother(
            window_size=np.array(5),
            polynomial_degree=np.int64(1),
        )
        self.assertEqual(valid.window_size, 5)
        self.assertEqual(valid.polynomial_degree, 1)

        with self.assertRaisesRegex(ValueError, "window_size"):
            smoother.smooth([eye(3), eye(3)], window_size=1.5)
        with self.assertRaisesRegex(ValueError, "polynomial_degree"):
            smoother.smooth([eye(3), eye(3)], polynomial_degree=True)

        with self.assertRaises(ValueError):
            smoother.smooth([eye(3), eye(3)], mask=[False, False])
        with self.assertRaises(ValueError):
            smoother.smooth([eye(3), eye(3)], mask=[True])

    def test_alias_is_exported(self):
        self.assertIs(SO3TSGSmoother, SO3TangentSavitzkyGolaySmoother)


if __name__ == "__main__":
    unittest.main()
