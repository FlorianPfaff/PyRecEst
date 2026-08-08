import unittest

from pyrecest.backend import array, zeros
from pyrecest.utils.nonrigid_point_set_registration import ThinPlateSplineTransform


class TestThinPlateSplineTransformFiniteValidation(unittest.TestCase):
    def test_rejects_nonfinite_transform_parameters(self):
        valid_control_points = array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        valid_weights = zeros((3, 2))
        valid_affine_coefficients = array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )

        invalid_parameters = (
            (
                "control_points",
                array([[0.0, 0.0], [float("nan"), 0.0], [0.0, 1.0]]),
            ),
            (
                "control_points",
                array([[0.0, 0.0], [float("inf"), 0.0], [0.0, 1.0]]),
            ),
            (
                "weights",
                array([[0.0, 0.0], [0.0, -float("inf")], [0.0, 0.0]]),
            ),
            (
                "weights",
                array([[0.0, 0.0], [0.0, float("nan")], [0.0, 0.0]]),
            ),
            (
                "affine_coefficients",
                array([[0.0, 0.0], [1.0, 0.0], [0.0, float("inf")]]),
            ),
            (
                "affine_coefficients",
                array([[0.0, 0.0], [1.0, 0.0], [0.0, float("nan")]]),
            ),
        )

        for parameter_name, invalid_value in invalid_parameters:
            with self.subTest(parameter_name=parameter_name, invalid_value=invalid_value):
                parameters = {
                    "control_points": valid_control_points,
                    "weights": valid_weights,
                    "affine_coefficients": valid_affine_coefficients,
                }
                parameters[parameter_name] = invalid_value
                with self.assertRaisesRegex(ValueError, parameter_name):
                    ThinPlateSplineTransform(**parameters)

    def test_from_translation_rejects_nonfinite_offsets(self):
        for invalid_offset in (
            array([float("nan"), 0.0]),
            array([0.0, float("inf")]),
            array([-float("inf"), 0.0]),
        ):
            with self.subTest(invalid_offset=invalid_offset):
                with self.assertRaisesRegex(ValueError, "affine_coefficients"):
                    ThinPlateSplineTransform.from_translation(invalid_offset)


if __name__ == "__main__":
    unittest.main()
