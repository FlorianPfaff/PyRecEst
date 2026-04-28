import unittest

import numpy.testing as npt
from pyrecest.backend import array, cos, eye, linalg, sin, stack
from pyrecest.smoothers import SO3ChordalMeanSmoother, SO3CMSmoother

ATOL = 1e-6


def z_rotation(angle):
    return array(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


class SO3ChordalMeanSmootherTest(unittest.TestCase):
    def test_chordal_mean_between_two_z_rotations(self):
        identity = eye(3)
        quarter_turn = z_rotation(0.5 * 3.141592653589793)

        mean_rotation = SO3ChordalMeanSmoother.chordal_mean([identity, quarter_turn])

        npt.assert_allclose(
            mean_rotation,
            z_rotation(0.25 * 3.141592653589793),
            atol=ATOL,
        )
        npt.assert_allclose(mean_rotation.T @ mean_rotation, eye(3), atol=ATOL)

    def test_weighted_chordal_mean_biases_toward_larger_weight(self):
        identity = eye(3)
        quarter_turn = z_rotation(0.5 * 3.141592653589793)

        weighted_mean = SO3ChordalMeanSmoother.chordal_mean(
            stack([identity, quarter_turn], axis=0),
            weights=array([3.0, 1.0]),
        )

        expected_angle = 0.3217505543966422
        npt.assert_allclose(weighted_mean, z_rotation(expected_angle), atol=ATOL)

    def test_accepts_matlab_style_rotation_stack(self):
        identity = eye(3)
        quarter_turn = z_rotation(0.5 * 3.141592653589793)

        mean_rotation = SO3ChordalMeanSmoother.chordal_mean(
            stack([identity, quarter_turn], axis=2)
        )

        npt.assert_allclose(
            mean_rotation,
            z_rotation(0.25 * 3.141592653589793),
            atol=ATOL,
        )

    def test_smooth_uses_local_windows(self):
        smoother = SO3ChordalMeanSmoother(window_size=3)
        rotations = [
            eye(3),
            z_rotation(0.5 * 3.141592653589793),
            z_rotation(0.5 * 3.141592653589793),
        ]

        smoothed = smoother.smooth(rotations)

        self.assertEqual(len(smoothed), 3)
        npt.assert_allclose(
            smoothed[0],
            z_rotation(0.25 * 3.141592653589793),
            atol=ATOL,
        )
        npt.assert_allclose(
            smoothed[1],
            z_rotation(1.1071487177940904),
            atol=ATOL,
        )
        npt.assert_allclose(smoothed[2], rotations[2], atol=ATOL)

    def test_project_to_so3_repairs_reflection(self):
        reflected = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )

        projected = SO3ChordalMeanSmoother.project_to_so3(reflected)

        npt.assert_allclose(projected.T @ projected, eye(3), atol=ATOL)
        npt.assert_allclose(linalg.det(projected), 1.0, atol=ATOL)

    def test_invalid_weights_raise_value_error(self):
        smoother = SO3ChordalMeanSmoother()

        with self.assertRaises(ValueError):
            SO3ChordalMeanSmoother.chordal_mean([eye(3), eye(3)], weights=[1.0])

        with self.assertRaises(ValueError):
            smoother.smooth([eye(3), eye(3)], weights=[1.0, -1.0])

    def test_alias_is_exported(self):
        self.assertIs(SO3CMSmoother, SO3ChordalMeanSmoother)


if __name__ == "__main__":
    unittest.main()
