# pylint: disable=no-member
import unittest

import pyrecest.backend
from tests.test_evaluation_basic import TestEvalationBase


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Not supported on this backend",
)
class TestEvalationEOT(TestEvalationBase):
    scenario_name = "R2randomWalkEOT"

    def test_evaluate_for_simulation_config_R2_random_walk(self):
        filters_configs_input = [
            {"name": "random_matrix", "parameter": None},
        ]

        self._evaluate_for_simulation_config(filters_configs_input)
