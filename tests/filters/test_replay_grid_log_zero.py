import unittest

import numpy as np

# pylint: disable=no-name-in-module
from pyrecest.filters import replay_grid_log_likelihood_values


class TestReplayGridLogZero(unittest.TestCase):
    def test_nearest_preserves_default_log_zero_for_nonfinite_grid_bin(self):
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        log_likelihood = np.asarray([0.0, -np.inf])

        result = replay_grid_log_likelihood_values(
            np.asarray([[1.0, 0.0], [0.0, 0.0]]),
            log_likelihood,
            bin_centers,
            interpolation="nearest",
        )

        self.assertTrue(np.isneginf(result[0]))
        self.assertEqual(result[1], 0.0)

    def test_nearest_uses_custom_log_zero_for_nonfinite_grid_bin(self):
        bin_centers = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        log_likelihood = np.asarray([0.0, -np.inf])

        result = replay_grid_log_likelihood_values(
            np.asarray([[1.0, 0.0]]),
            log_likelihood,
            bin_centers,
            interpolation="nearest",
            log_zero=-123.0,
        )

        np.testing.assert_allclose(result, [-123.0])


if __name__ == "__main__":
    unittest.main()
