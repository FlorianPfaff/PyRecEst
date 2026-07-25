"""Regression tests for RNG isolation during scenario generation."""

from __future__ import annotations

import copy
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import random

scenario_generation = importlib.import_module(
    "pyrecest.evaluation.generate_simulated_scenarios"
)


class TestScenarioGenerationRandomState(unittest.TestCase):
    @staticmethod
    def _run_stubbed_generation():
        simulation_params = {"all_seeds": [7, 11], "n_timesteps": 1}
        groundtruth = [np.array([0.0])]
        measurements = [np.array([[0.0]])]

        with (
            patch.object(
                scenario_generation,
                "check_and_fix_config",
                side_effect=lambda config: config,
            ),
            patch.object(
                scenario_generation,
                "generate_groundtruth",
                return_value=groundtruth,
            ),
            patch.object(
                scenario_generation,
                "generate_measurements",
                return_value=measurements,
            ),
        ):
            scenario_generation.generate_simulated_scenarios(simulation_params)

    def test_generation_preserves_caller_rng_state(self):
        original_backend_state = copy.deepcopy(random.get_state())
        original_numpy_state = np.random.get_state()
        try:
            if pyrecest.backend.__backend_name__ in ("numpy", "autograd"):
                np.random.seed(314159)
                expected_numpy = np.random.random(4)
                np.random.seed(314159)

                self._run_stubbed_generation()

                npt.assert_allclose(np.random.random(4), expected_numpy)
                return

            random.seed(314159)
            expected_backend = pyrecest.backend.to_numpy(random.rand(size=(4,))).copy()
            random.seed(314159)

            np.random.seed(271828)
            expected_numpy = np.random.random(4)
            np.random.seed(271828)

            self._run_stubbed_generation()

            actual_backend = pyrecest.backend.to_numpy(random.rand(size=(4,)))
            actual_numpy = np.random.random(4)
            npt.assert_allclose(actual_backend, expected_backend)
            npt.assert_allclose(actual_numpy, expected_numpy)
        finally:
            random.set_state(original_backend_state)
            np.random.set_state(original_numpy_state)


if __name__ == "__main__":
    unittest.main()
