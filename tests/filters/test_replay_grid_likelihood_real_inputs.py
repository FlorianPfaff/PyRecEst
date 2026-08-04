import unittest

import numpy as np

# pylint: disable=no-name-in-module
from pyrecest.filters import (
    grid_proposal_weights,
    replay_grid_log_likelihood_values,
)


class TestReplayGridLikelihoodRealInputs(unittest.TestCase):
    def test_rejects_complex_and_masked_log_likelihoods(self):
        invalid_values = (
            np.asarray([0.0 + 0.0j, 1.0 + 2.0j]),
            np.asarray([np.complex64(0.0 + 1.0j), 1.0], dtype=object),
            np.ma.array([0.0, 1.0], mask=[False, True]),
            [0.0, np.ma.masked],
        )
        bin_centers = np.asarray([[0.0], [1.0]])
        positions = np.asarray([[0.0]])

        for values in invalid_values:
            with self.subTest(values=repr(values), path="proposal"):
                with self.assertRaisesRegex(
                    ValueError, "log_likelihood must contain real unmasked values"
                ):
                    grid_proposal_weights(values)
            with self.subTest(values=repr(values), path="lookup"):
                with self.assertRaisesRegex(
                    ValueError, "log_likelihood must contain real unmasked values"
                ):
                    replay_grid_log_likelihood_values(
                        positions,
                        values,
                        bin_centers,
                        interpolation="nearest",
                    )

    def test_accepts_clear_masks_and_nonfinite_zero_mass_entries(self):
        values = np.ma.array([0.0, -np.inf], mask=False)
        bin_centers = np.asarray([[0.0], [1.0]])

        proposal = grid_proposal_weights(values)
        evaluated = replay_grid_log_likelihood_values(
            np.asarray([[0.0], [1.0]]),
            values,
            bin_centers,
            interpolation="nearest",
            log_zero=-123.0,
        )

        np.testing.assert_allclose(proposal, [1.0, 0.0])
        np.testing.assert_allclose(evaluated, [0.0, -123.0])


if __name__ == "__main__":
    unittest.main()
