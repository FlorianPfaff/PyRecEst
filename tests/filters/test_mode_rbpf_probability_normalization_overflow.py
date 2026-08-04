import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend as pyrecest_backend

from pyrecest.filters import ModeRBPFManifoldUKFTracker


@unittest.skipIf(
    pyrecest_backend.__backend_name__ != "numpy",
    reason="ModeRBPFManifoldUKFTracker is currently NumPy-backend only",
)
class TestModeRBPFProbabilityNormalizationOverflow(unittest.TestCase):
    def setUp(self):
        self.large = np.finfo(np.float64).max

    def test_stabilizes_transition_rows(self):
        matrix = np.array(
            [
                [self.large, self.large / 2.0, 0.0],
                [0.0, self.large, self.large / 2.0],
                [self.large / 2.0, 0.0, self.large],
            ]
        )

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = ModeRBPFManifoldUKFTracker._normalize_rows(matrix)

        npt.assert_allclose(
            normalized,
            np.array(
                [
                    [2.0 / 3.0, 1.0 / 3.0, 0.0],
                    [0.0, 2.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 3.0, 0.0, 2.0 / 3.0],
                ]
            ),
        )
        npt.assert_allclose(np.sum(normalized, axis=1), np.ones(3))

    def test_stabilizes_initial_mode_probabilities(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = ModeRBPFManifoldUKFTracker._normalize_probs(
                [self.large, self.large / 2.0, 0.0]
            )

        npt.assert_allclose(normalized, [2.0 / 3.0, 1.0 / 3.0, 0.0])
        npt.assert_allclose(np.sum(normalized), 1.0)

    def test_stabilizes_resampling_weights(self):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = ModeRBPFManifoldUKFTracker._normalize_resampling_weights(
                [self.large, self.large / 2.0, 0.0]
            )

        npt.assert_allclose(normalized, [2.0 / 3.0, 1.0 / 3.0, 0.0])
        npt.assert_allclose(np.sum(normalized), 1.0)

    def test_public_resampling_path_accepts_extreme_finite_weights(self):
        tracker = object.__new__(ModeRBPFManifoldUKFTracker)
        tracker.resampling_mode = "systematic"
        tracker.rng = np.random.default_rng(0)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            indices = tracker._sample_indices(
                np.array([self.large, self.large / 2.0, 0.0]), 6
            )

        npt.assert_array_equal(np.bincount(indices, minlength=3), [4, 2, 0])


if __name__ == "__main__":
    unittest.main()
