import unittest

import numpy as np
import pyrecest.backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Nearest-neighbor trackers are only supported on the numpy backend",
)
class NearestNeighborPredictionValidationTest(unittest.TestCase):
    def setUp(self):
        prior = [
            GaussianDistribution(np.zeros(2), np.eye(2)),
            GaussianDistribution(np.ones(2), np.eye(2)),
        ]
        self.tracker = GlobalNearestNeighbor(
            initial_prior=prior,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )

    def test_rejects_system_matrix_target_count_mismatch(self):
        for supplied_targets in (1, 3):
            with self.subTest(supplied_targets=supplied_targets):
                system_matrices = np.repeat(
                    np.eye(2)[:, :, None], supplied_targets, axis=2
                )
                with self.assertRaisesRegex(ValueError, "one system matrix per target"):
                    self.tracker.predict_linear(system_matrices, np.eye(2))

    def test_rejects_system_noise_target_count_mismatch(self):
        system_noises = np.repeat(np.eye(2)[:, :, None], 3, axis=2)

        with self.assertRaisesRegex(ValueError, "one system-noise matrix per target"):
            self.tracker.predict_linear(np.eye(2), system_noises)

    def test_rejects_input_target_count_mismatch(self):
        inputs = np.zeros((2, 3))

        with self.assertRaisesRegex(ValueError, "one input vector per target"):
            self.tracker.predict_linear(np.eye(2), np.eye(2), inputs)
