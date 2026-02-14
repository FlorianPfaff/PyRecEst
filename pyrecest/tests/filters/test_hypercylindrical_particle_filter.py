import unittest
from pyrecest.filters.hypercylindrical_particle_filter import HypercylindricalParticleFilter
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import PartiallyWrappedNormalDistribution
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import HypercylindricalDiracDistribution
import numpy.testing as npt
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, random, ones, mod, pi, linalg, ones, zeros, mean
# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
class HypercylindricalParticleFilterTest(unittest.TestCase):
    def test_initialization(self):
        HypercylindricalParticleFilter(10, 2, 2)

    def test_setting_state(self):
        random.seed(0)
        n = 5000
        hwn = PartiallyWrappedNormalDistribution(array([1., 2., 3., 4.]), diag([1., 2., 3., 4.]), 2)
        ddist = HypercylindricalDiracDistribution.from_distribution(hwn, n)

        pf = HypercylindricalParticleFilter(200, 2, 2)
        pf.filter_state = ddist
        npt.assert_allclose(pf.get_point_estimate(), hwn.mu,atol=0.1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_without_noise_wraps_periodic_dim(self):
        bound_dim = 1
        lin_dim = 1
        n_particles = 4
        initial_particles = array(
            [
                [0.1, 1.0],
                [2.0, -1.0],
                [3.5, 0.0],
                [6.2, 2.5],
            ]
        )
        hpf = HypercylindricalParticleFilter(n_particles, bound_dim, lin_dim)
        hpf.filter_state = HypercylindricalDiracDistribution(bound_dim, initial_particles)

        def motion_fn(x):
            return x + array([0.5, 0.5]).reshape((1,-1))

        hpf.predict_nonlinear(motion_fn, noise_distribution=None, function_is_vectorized=True)

        expected = motion_fn(initial_particles)
        expected[:, 0] = mod(expected[:, 0], 2 * pi)
        npt.assert_allclose(hpf.filter_state.d, expected)


    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_nonvectorized_applies_per_particle(self):
        bound_dim = 1
        lin_dim = 1
        n_particles = 3
        initial_particles = array(
            [
                [0.1, -1.0],
                [1.5, 0.5],
                [3.0, 2.0],
            ]
        )
        hpf = HypercylindricalParticleFilter(n_particles, bound_dim, lin_dim)
        hpf.filter_state = HypercylindricalDiracDistribution(bound_dim, initial_particles)

        def nonvectorized_step(x):
            # This function is intentionally not vectorized; it expects a single particle.
            assert x.shape == (2,)
            return x + array([0.2, 1.0])

        hpf.predict_nonlinear(nonvectorized_step, noise_distribution=None, function_is_vectorized=False)

        expected = initial_particles + array([0.2, 1.0])
        expected[:, 0] = mod(expected[:, 0], 2 * pi)
        npt.assert_allclose(hpf.filter_state.d, expected)


    def test_predict_update_cycle_3d(self):
        random.seed(0)
        C = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = array([1.0, 1.0, 1.0])
        bound_dim = 1
        lin_dim = 2
        hwnd = PartiallyWrappedNormalDistribution(mu, C, bound_dim)
        hpf = HypercylindricalParticleFilter(500, bound_dim, lin_dim)
        hpf.filter_state = hwnd
        forced_mean = array([1.0, 2.0, 3.0])

        for _ in range(50):
            hpf.predict_identity(PartiallyWrappedNormalDistribution(zeros(3), C, bound_dim))
            npt.assert_equal(hpf.get_point_estimate().shape, (3,))
            for _ in range(3):
                hpf.update_identity(PartiallyWrappedNormalDistribution(zeros(3), 0.5 * C, bound_dim), forced_mean)

        npt.assert_equal(hpf.get_point_estimate().shape, (3,))
        npt.assert_allclose(hpf.get_point_estimate(), forced_mean)

        n = 5
        samples = random.uniform(size=(3, n))
        weights = ones((n,)) / n
        f = lambda x, w: mod(x + w, 2 * pi)
        hpf.filter_state = hwnd
        hpf.predict_nonlinear_nonadditive(f, samples, weights)
        est = hpf.get_point_estimate()
        npt.assert_equal(hpf.get_point_estimate().shape, (3,))
        npt.assert_allclose(linalg.norm(est - mod(hwnd.mu + mean(samples, axis=1).reshape(-1, 1), 2 * pi)), 0)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_update_cycle_3d_forced_particle_pos_no_pred(self):
        random.seed(0)
        C = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = array([1., 1., 1.]) + pi / 2
        bound_d = 1
        lin_d = 2
        hwnd = PartiallyWrappedNormalDistribution(mu, C, bound_d)
        hpf = HypercylindricalParticleFilter(500, bound_d, lin_d)
        hpf.filter_state = hwnd
        forced_mean = array([1., 10., 20.])
        force_first_particle_pos = array([1.1, 10., 20.])
        hpf.filter_state.d[0, :] = force_first_particle_pos.flatten()

        for _ in range(50):
            npt.assert_equal(hpf.get_point_estimate().shape, (3,))
            hpf.update_identity(PartiallyWrappedNormalDistribution(zeros((3,)), 0.5 * C, bound_d), forced_mean)
            hpf.update_identity(PartiallyWrappedNormalDistribution(zeros((3,)), 0.5 * C, bound_d), forced_mean)
            hpf.update_identity(PartiallyWrappedNormalDistribution(zeros((3,)), 0.5 * C, bound_d), forced_mean)

        npt.assert_equal(hpf.get_point_estimate().shape, (3,))
        npt.assert_allclose(hpf.get_point_estimate(), force_first_particle_pos, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

