import unittest

import numpy as np

# pylint: disable=no-name-in-module
from pyrecest.filters import grid_proposal_weights, replay_grid_log_likelihood_values


class TestReplayGridLogLikelihoodValidation(unittest.TestCase):
    def test_rejects_nan_and_positive_infinity(self):
        bin_centers = np.asarray([[0.0], [1.0]])
        positions = np.asarray([[0.0]])

        for invalid_value in (np.nan, np.inf):
            log_likelihood = np.asarray([0.0, invalid_value])
            with self.subTest(invalid_value=invalid_value, api="lookup"):
                with self.assertRaisesRegex(
                    ValueError, "finite values or -np.inf"
                ):
                    replay_grid_log_likelihood_values(
                        positions,
                        log_likelihood,
                        bin_centers,
                        interpolation="nearest",
                    )
            with self.subTest(invalid_value=invalid_value, api="proposal"):
                with self.assertRaisesRegex(
                    ValueError, "finite values or -np.inf"
                ):
                    grid_proposal_weights(log_likelihood)

    def test_negative_infinity_remains_zero_likelihood(self):
        bin_centers = np.asarray([[0.0], [1.0]])
        log_likelihood = np.asarray([0.0, -np.inf])

        np.testing.assert_allclose(grid_proposal_weights(log_likelihood), [1.0, 0.0])
        looked_up = replay_grid_log_likelihood_values(
            np.asarray([[1.0]]),
            log_likelihood,
            bin_centers,
            interpolation="nearest",
        )
        self.assertTrue(np.isneginf(looked_up[0]))


if __name__ == "__main__":
    unittest.main()
