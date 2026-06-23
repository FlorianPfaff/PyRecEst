"""Regression tests for boolean per-observation cost validation."""

import unittest

from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
)
from pyrecest.utils import solve_multisession_assignment_with_observation_costs


@unittest.skipIf(
    __backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestObservationCostBooleanEntries(unittest.TestCase):
    def test_rejects_boolean_observation_cost_entries(self):
        invalid_cost_kwargs = (
            {"start_costs": {0: True}},
            {"end_costs": {1: array([True])}},
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
