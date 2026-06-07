import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, diag, eye, pi
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="MEM-RBPF public covariance tests use numpy.testing assertions",
)
class TestMEMRBPFTrackerPublicCovariance(unittest.TestCase):
    def _make_tracker(self):
        tracker = MEMRBPFTracker.__new__(MEMRBPFTracker)
        tracker.n_particles = 2
        tracker.state_dim = 4
        tracker.kinematic_state = array([1.0, 2.0, 3.0, 4.0])
        tracker.covariance = diag(array([0.1, 0.2, 0.3, 0.4]))
        tracker.theta = array([0.0, pi])
        tracker.axis = array([[2.0, 1.0], [2.0, 1.0]])
        tracker.axis_covariances = array(
            [
                [[0.10, 0.01], [0.01, 0.20]],
                [[0.10, 0.01], [0.01, 0.20]],
            ]
        )
        tracker.weights = array([0.5, 0.5])
        tracker.measurement_matrix = eye(2, 4)
        return tracker

    def test_get_state_and_cov_returns_public_particle_covariance(self):
        tracker = self._make_tracker()

        state, covariance = tracker.get_state_and_cov()

        npt.assert_allclose(state, array([1.0, 2.0, 3.0, 4.0, 0.0, 4.0, 2.0]))
        self.assertEqual(covariance.shape, (7, 7))
        npt.assert_allclose(covariance[:4, :4], tracker.covariance)
        self.assertAlmostEqual(float(covariance[4, 4]), 0.0)
        self.assertAlmostEqual(float(covariance[5, 5]), 0.4)
        self.assertAlmostEqual(float(covariance[6, 6]), 0.8)
        self.assertAlmostEqual(float(covariance[5, 6]), 0.04)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= -1e-12))

    def test_particle_weight_normalization_falls_back_to_uniform(self):
        tracker = self._make_tracker()
        tracker.weights = array([np.nan, 0.0])

        npt.assert_allclose(tracker._normalized_particle_weights(), array([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
