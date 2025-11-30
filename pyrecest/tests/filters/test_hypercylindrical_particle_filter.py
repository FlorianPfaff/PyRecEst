import numpy as np
import unittest
from pyrecest.filters.hypercylindrical_particle_filter import HypercylindricalParticleFilter
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import PartiallyWrappedNormalDistribution
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import HypercylindricalDiracDistribution
import numpy.testing as npt

class HypercylindricalParticleFilterTest(unittest.TestCase):
    def test_initialization(self):
        HypercylindricalParticleFilter(10, 2, 2)

    def test_setting_state(self):
        np.random.seed(0)
        n = 5000
        hwn = PartiallyWrappedNormalDistribution(np.array([1, 2, 3, 4]), np.diag([1, 2, 3, 4]), 2)
        ddist = HypercylindricalDiracDistribution.from_distribution(hwn, n)

        pf = HypercylindricalParticleFilter(200, 2, 2)
        pf.filter_state = ddist
        np.testing.assert_allclose(pf.get_point_estimate(), hwn.mu,atol=0.1)

    def test_predict_update_cycle_3d(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = np.array([1.0, 1.0, 1.0])
        bound_dim = 1
        lin_dim = 2
        hwnd = PartiallyWrappedNormalDistribution(mu, C, bound_dim)
        hpf = HypercylindricalParticleFilter(500, bound_dim, lin_dim)
        hpf.filter_state = hwnd
        forced_mean = np.array([1.0, 2.0, 3.0])

        for _ in range(50):
            hpf.predict_identity(PartiallyWrappedNormalDistribution(np.zeros(3), C, bound_dim))
            self.assertEqual(hpf.get_point_estimate().shape, (3,))
            for _ in range(3):
                hpf.update_identity(PartiallyWrappedNormalDistribution(np.zeros(3), 0.5 * C, bound_dim), forced_mean)

        self.assertEqual(hpf.get_point_estimate().shape, (3,))
        npt.assert_allclose(hpf.get_point_estimate(), forced_mean)

        n = 5
        samples = np.random.rand(3, n)
        weights = np.ones((1, n)) / n
        f = lambda x, w: np.mod(x + w, 2 * np.pi)
        hpf.filter_state = hwnd
        hpf.predict_nonlinear_nonadditive(f, samples, weights)
        est = hpf.get_point_estimate()
        self.assertEqual(hpf.get_point_estimate().shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(est - np.mod(hwnd.mu + np.mean(samples, axis=1).reshape(-1, 1), 2 * np.pi)), 0)

    def test_predict_update_cycle_3d_forced_particle_pos_no_pred(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = np.array([1, 1, 1]) + np.pi / 2
        bound_d = 1
        lin_d = 2
        hwnd = PartiallyWrappedNormalDistribution(mu, C, bound_d)
        hpf = HypercylindricalParticleFilter(500, bound_d, lin_d)
        hpf.filter_state = hwnd
        forced_mean = np.array([1, 10, 20])
        force_first_particle_pos = np.array([1.1, 10, 20])
        hpf.filter_state.d[:, 0] = force_first_particle_pos.flatten()

        for _ in range(50):
            self.assertEqual(hpf.get_point_estimate().shape, (3,))
            hpf.update_identity(PartiallyWrappedNormalDistribution(np.zeros((3,)), 0.5 * C, bound_d), forced_mean)
            hpf.update_identity(PartiallyWrappedNormalDistribution(np.zeros((3,)), 0.5 * C, bound_d), forced_mean)
            hpf.update_identity(PartiallyWrappedNormalDistribution(np.zeros((3,)), 0.5 * C, bound_d), forced_mean)

        self.assertEqual(hpf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(hpf.get_point_estimate(), force_first_particle_pos, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

