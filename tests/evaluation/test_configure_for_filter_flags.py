import unittest

import numpy as np
from pyrecest.evaluation.configure_for_filter import (
    _gen_next_state_with_noise_is_vectorized,
)


class ConfigureForFilterFlagValidationTest(unittest.TestCase):
    def test_vectorized_flag_defaults_to_false(self):
        self.assertFalse(_gen_next_state_with_noise_is_vectorized({}))

    def test_vectorized_flag_accepts_boolean_scalars(self):
        self.assertTrue(
            _gen_next_state_with_noise_is_vectorized(
                {"gen_next_state_with_noise_is_vectorized": True}
            )
        )
        self.assertFalse(
            _gen_next_state_with_noise_is_vectorized(
                {"genNextStateWithNoiseIsVectorized": np.bool_(False)}
            )
        )

    def test_vectorized_flag_rejects_truthy_strings_and_numeric_values(self):
        invalid_configs = (
            {"gen_next_state_with_noise_is_vectorized": "False"},
            {"gen_next_state_with_noise_is_vectorized": 1},
            {"genNextStateWithNoiseIsVectorized": np.array([False])},
        )

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "must be a boolean scalar"):
                    _gen_next_state_with_noise_is_vectorized(config)


if __name__ == "__main__":
    unittest.main()
