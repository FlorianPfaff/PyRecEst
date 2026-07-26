import unittest

import numpy as np
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.utils.point_set_registration import (
    estimate_transform,
    joint_registration_assignment,
    solve_gated_assignment,
)


class TestPointSetRegistrationNumericValidation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_estimate_transform_rejects_boolean_point_coordinates(self):
        source = np.array([[False, False], [True, False]], dtype=bool)
        target = array([[0.0, 0.0], [1.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "real numeric"):
            estimate_transform(source, target, model="translation")

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_solve_gated_assignment_rejects_temporal_cost_matrix(self):
        cost_matrix = np.array([[np.timedelta64(1, "ns")]], dtype="timedelta64[ns]")

        with self.assertRaisesRegex(ValueError, "real numeric"):
            solve_gated_assignment(cost_matrix)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_solve_gated_assignment_rejects_temporal_max_cost(self):
        cost_matrix = array([[0.5], [2.0]])
        invalid_max_costs = (
            np.timedelta64(1, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000001"),
            np.array(np.timedelta64(1, "ns"), dtype=object),
        )

        for invalid_max_cost in invalid_max_costs:
            with self.subTest(invalid_max_cost=invalid_max_cost):
                with self.assertRaisesRegex(ValueError, "max_cost"):
                    solve_gated_assignment(cost_matrix, max_cost=invalid_max_cost)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_joint_registration_assignment_rejects_temporal_tolerance(self):
        points = array([[0.0, 0.0], [1.0, 0.0]])
        invalid_tolerances = (
            np.timedelta64(0, "ns"),
            np.datetime64("1970-01-01T00:00:00.000000000"),
            np.array(np.timedelta64(0, "ns"), dtype=object),
        )

        for invalid_tolerance in invalid_tolerances:
            with self.subTest(invalid_tolerance=invalid_tolerance):
                with self.assertRaisesRegex(ValueError, "tolerance"):
                    joint_registration_assignment(
                        points,
                        points,
                        model="translation",
                        tolerance=invalid_tolerance,
                    )


if __name__ == "__main__":
    unittest.main()
