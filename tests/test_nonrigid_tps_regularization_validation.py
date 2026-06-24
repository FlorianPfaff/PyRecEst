import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.nonrigid_point_set_registration import (
    ThinPlateSplineTransform,
    joint_tps_registration_assignment,
)


class TestJointThinPlateSplineRegularizationValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_tps_registration_rejects_invalid_regularization_before_matching(
        self,
    ):
        reference = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        moving = reference + array([100.0, 100.0])

        for bad_regularization in (
            float("nan"),
            float("inf"),
            -float("inf"),
            array([1e-3]),
        ):
            with self.subTest(bad_regularization=bad_regularization):
                with self.assertRaisesRegex(ValueError, "regularization"):
                    joint_tps_registration_assignment(
                        reference,
                        moving,
                        initial_transform=ThinPlateSplineTransform.identity(),
                        max_cost=0.0,
                        regularization=bad_regularization,
                    )


if __name__ == "__main__":
    unittest.main()
