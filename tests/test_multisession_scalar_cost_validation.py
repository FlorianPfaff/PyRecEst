"""Regression tests for multi-session scalar cost validation."""

import unittest

import numpy as np
from pyrecest.backend import __backend_name__
from pyrecest.utils import multisession_assignment as multisession_assignment_module
from pyrecest.utils import solve_multisession_assignment


class TestMultiSessionScalarCostValidation(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_boolean_scalar_costs_are_rejected(self):
        invalid_costs = (
            ("start_cost", True),
            ("end_cost", np.bool_(True)),
            ("gap_penalty", np.array(False, dtype=object)),
            ("cost_threshold", np.array(True)),
        )

        for name, value in invalid_costs:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, f"{name} must be a finite scalar"
                ):
                    solve_multisession_assignment(
                        {},
                        session_sizes=[1],
                        **{name: value},
                    )
                with self.assertRaisesRegex(
                    ValueError, f"{name} must be a finite scalar"
                ):
                    multisession_assignment_module.solve_multisession_assignment(
                        {},
                        session_sizes=[1],
                        **{name: value},
                    )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_text_scalar_costs_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "start_cost must be a finite scalar"):
            solve_multisession_assignment({}, session_sizes=[1], start_cost="1.0")


if __name__ == "__main__":
    unittest.main()
