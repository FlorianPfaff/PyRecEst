import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi, random, zeros, zeros_like
from pyrecest.distributions import HypertoroidalWNDistribution
from pyrecest.filters import HypertoroidalParticleFilter


class HypertoroidalParticleFilterTest(unittest.TestCase):
    def setUp(self):
        self.seed = 0
        self.covariance_matrix = array(
            [[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]]
        )
        self.mu = array([1.0, 1.0, 1.0]) + pi / 2
        self.hwnd = HypertoroidalWNDistribution(self.mu, self.covariance_matrix)
        self.hpf = HypertoroidalParticleFilter(500, 3)
        self.forced_mean = array([1.0, 2.0, 3.0])
        random.seed(self.seed)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported'"
    )
    def test_setting_state(self):
        self.hpf.filter_state = self.hwnd

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported'"
    )
    def test_predict_identity(self):
        self.hpf.predict_identity(
            HypertoroidalWNDistribution(zeros(3), 0.5 * self.covariance_matrix)
        )
        self.assertEqual(self.hpf.get_point_estimate().shape, (3,))

    def test_update_identity(self):
        self.hpf.update_identity(
            HypertoroidalWNDistribution(zeros(3), 0.5 * self.covariance_matrix),
            self.forced_mean,
        )
        self.assertEqual(self.hpf.get_point_estimate().shape, (3,))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported'"
    )
    def test_predict_update_cycle_3D(self):
        self.hpf.filter_state = self.hwnd
        for _ in range(5):
            self.hpf.predict_identity(
                HypertoroidalWNDistribution(zeros_like(self.mu), self.covariance_matrix)
            )
            for _ in range(3):
                self.test_update_identity()
        npt.assert_allclose(self.hpf.get_point_estimate(), self.forced_mean, atol=0.15)


if __name__ == "__main__":
    unittest.main()
