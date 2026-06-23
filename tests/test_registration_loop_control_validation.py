import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.nonrigid_point_set_registration import joint_tps_registration_assignment
from pyrecest.utils.point_set_registration import joint_registration_assignment


class TestRegistrationLoopControlValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_assignment_rejects_invalid_max_iterations(self):
        reference = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        moving = reference + array([1.0, -1.0])

        for bad_value in (True, 1.5, float("nan"), array([2])):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "max_iterations"):
                    joint_registration_assignment(
                        reference,
                        moving,
                        model="translation",
                        max_iterations=bad_value,
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_assignment_rejects_invalid_min_matches(self):
        reference = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        moving = reference + array([1.0, -1.0])

        for bad_value in (True, 1.5, float("inf"), array([1])):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "min_matches"):
                    joint_registration_assignment(
                        reference,
                        moving,
                        model="translation",
                        min_matches=bad_value,
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_tps_registration_assignment_rejects_invalid_max_iterations(self):
        reference = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        moving = reference + array([0.5, -0.25])

        for bad_value in (False, 2.5, float("nan"), array([2])):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "max_iterations"):
                    joint_tps_registration_assignment(
                        reference,
                        moving,
                        max_iterations=bad_value,
                    )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_tps_registration_assignment_rejects_invalid_min_matches(self):
        reference = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        moving = reference + array([0.5, -0.25])

        for bad_value in (True, 2, 2.5, float("inf"), array([3])):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "min_matches"):
                    joint_tps_registration_assignment(
                        reference,
                        moving,
                        min_matches=bad_value,
                    )


if __name__ == "__main__":
    unittest.main()
