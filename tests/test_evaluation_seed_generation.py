"""Regression tests for simulation seed generation."""

from __future__ import annotations

import random
import unittest

from pyrecest.evaluation.evaluate_for_simulation_config import get_all_seeds


class TestEvaluationSeedGeneration(unittest.TestCase):
    def test_nonconsecutive_generation_preserves_python_random_state(self) -> None:
        original_state = random.getstate()
        try:
            random.seed(314159)
            expected_next_value = random.random()

            random.seed(314159)
            generated_seeds = get_all_seeds(
                n_runs=4,
                seed_input=7,
                consecutive_seed=False,
            )
            actual_next_value = random.random()
        finally:
            random.setstate(original_state)

        self.assertEqual(
            generated_seeds,
            [1390851129, 4071050725, 647892280, 1695753999],
        )
        self.assertEqual(actual_next_value, expected_next_value)


if __name__ == "__main__":
    unittest.main()
