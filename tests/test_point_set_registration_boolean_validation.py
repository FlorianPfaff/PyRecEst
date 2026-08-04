import unittest

import numpy as np
import pyrecest.backend
from pyrecest.backend import array, linalg
from pyrecest.utils.point_set_registration import (
    estimate_transform,
    joint_registration_assignment,
)


class TestPointSetRegistrationBooleanValidation(unittest.TestCase):
    @staticmethod
    def _reflected_points():
        source = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        return source, target

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_non_boolean_reflection_flags(self):
        source, target = self._reflected_points()
        invalid_flags = (
            "false",
            1,
            0.0,
            np.array([False]),
            np.ma.array(False, mask=True),
        )

        for invalid_flag in invalid_flags:
            with self.subTest(invalid_flag=invalid_flag):
                with self.assertRaisesRegex(ValueError, "allow_reflection"):
                    estimate_transform(
                        source,
                        target,
                        model="rigid",
                        allow_reflection=invalid_flag,
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_rejects_non_boolean_reflection_flags_eagerly(self):
        source, target = self._reflected_points()

        for invalid_flag in ("false", 1, np.array([True])):
            with self.subTest(invalid_flag=invalid_flag):
                with self.assertRaisesRegex(ValueError, "allow_reflection"):
                    joint_registration_assignment(
                        source,
                        target,
                        model="rigid",
                        allow_reflection=invalid_flag,
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_accepts_scalar_boolean_array(self):
        source, target = self._reflected_points()

        transform = estimate_transform(
            source,
            target,
            model="rigid",
            allow_reflection=np.array(True),
        )

        self.assertLess(float(linalg.det(transform.matrix)), 0.0)


if __name__ == "__main__":
    unittest.main()
