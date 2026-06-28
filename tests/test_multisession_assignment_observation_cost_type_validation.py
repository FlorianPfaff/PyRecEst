"""Regression tests for observation-specific cost type validation."""

import unittest

import numpy as np
from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
)
from pyrecest.utils import solve_multisession_assignment_with_observation_costs


@unittest.skipIf(
    __backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestObservationCostTypeValidation(unittest.TestCase):
    def test_rejects_text_scalar_cost_parameters(self):
        invalid_cost_kwargs = (
            {"start_cost": "1.0"},
            {"end_cost": b"2.0"},
            {"gap_penalty": np.str_("0.5")},
            {"cost_threshold": "3.0"},
        )

        for cost_kwargs in invalid_cost_kwargs:
            with self.subTest(cost_kwargs=cost_kwargs):
                with self.assertRaisesRegex(ValueError, "must be a finite scalar"):
                    solve_multisession_assignment_with_observation_costs(
                        [array([[1.0]], dtype=float)],
                        **cost_kwargs,
                    )

    def test_rejects_text_observation_cost_entries(self):
        invalid_cost_kwargs = (
            {"start_costs": {0: "1.0"}},
            {"end_costs": {1: np.array(["2.0"], dtype=object)}},
        )

        for cost_kwargs in invalid_cost_kwargs:
            with self.subTest(cost_kwargs=cost_kwargs):
                with self.assertRaisesRegex(ValueError, "real numeric values"):
                    solve_multisession_assignment_with_observation_costs(
                        [array([[1.0]], dtype=float)],
                        **cost_kwargs,
                    )

    def test_rejects_object_boolean_observation_cost_entries(self):
        invalid_cost_kwargs = (
            {"start_costs": {0: np.array(True, dtype=object)}},
            {"end_costs": {1: np.array([True], dtype=object)}},
        )

        for cost_kwargs in invalid_cost_kwargs:
            with self.subTest(cost_kwargs=cost_kwargs):
                with self.assertRaisesRegex(ValueError, "not boolean"):
                    solve_multisession_assignment_with_observation_costs(
                        [array([[1.0]], dtype=float)],
                        **cost_kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
