import unittest
import numpy as np
from pyrecest.filters.hyperhemispherical_grid_filter import HyperhemisphericalGridFilter
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import SdHalfCondSdHalfGridDistribution
from pyrecest.distributions import BinghamDistribution, HypersphericalMixture, VonMisesFisherDistribution

class HyperhemisphericalGridFilterTest(unittest.TestCase):
    def test_set_state_s2(self):
        np.random.seed(0)
        no_grid_points = 1001
        sgf = HyperhemisphericalGridFilter(no_grid_points, 3)

        self.assertEqual(sgf.get_estimate().grid_values.shape, (no_grid_points, 1))

        # Test if it is uniform at the beginning
        self.assertAlmostEqual(np.sum(np.abs(sgf.get_estimate().grid_values - (1 / sgf.get_estimate().get_manifold_size() * np.ones((no_grid_points, 1))))), 0, delta=1e-13)

        M = np.eye(3)
        Z = np.array([-2, -1, 0]).reshape(-1, 1)
        bd = BinghamDistribution(Z, M)
        bd.F = bd.F * bd.integrate_numerically()

        sgd_state = HyperhemisphericalGridDistribution.from_distribution(bd, no_grid_points)
        self.assertIsInstance(sgf.gd, HyperhemisphericalGridDistribution)
        sgf.set_state(sgd_state)
        self.assertIsInstance(sgf.gd, HyperhemisphericalGridDistribution)

        # Verify that it is no longer a uniform distribution
        self.assertGreater(np.sum(np.abs(sgf.get_estimate().grid_values - (1 / sgf.get_estimate().get_manifold_size()))), 60)

        # Verify estimate matches a mode of the Bingham
        self.assertAlmostEqual(np.min(np.linalg.norm(sgf.get_point_estimate() - np.hstack((bd.mode(), -bd.mode())), axis=0)), 0, delta=1e-11)

        # Verify errors
        full_grid = sgd_state.get_grid()
        sgd_state.grid = full_grid[:, -1]
        sgd_state.grid_values = sgd_state.grid_values[:-1]
        self.assertIsInstance(sgf.gd, HyperhemisphericalGridDistribution)
        sgf_tmp = sgf.copy()

        with self.assertRaises(ValueError):
            sgf_tmp.set_state(sgd_state)

        with self.assertRaises(ValueError):
            sgf.set_state(bd)

    def test_set_state_s3(self):
        no_grid_points = 1001
        sgf = HyperhemisphericalGridFilter(no_grid_points, 4)
        self.assertEqual(sgf.get_estimate().grid_values.shape, (no_grid_points, 1))

        # Test if it is uniform at the beginning
        self.assertAlmostEqual(np.sum(np.abs(np.diff(sgf.get_estimate().grid_values.T))), 0)

        M = np.eye(4)
        Z = np.array([-2, -1, -0.5, 0]).reshape(-1, 1)
        bd = BinghamDistribution(Z, M)
        bd.F = bd.F * bd.integrate_numerically()

        sgd_state = HyperhemisphericalGridDistribution.from_distribution(bd, no_grid_points)
        sgf.set_state(sgd_state)

        # Verify that it is no longer a uniform distribution
        self.assertGreater(np.sum(np.abs(np.diff(sgf.get_estimate().grid_values.T))), 5)
    def test_predict_converges_to_uniform_S2_S3(self):
        test_predict_converges_to_uniform(3, 501, [-2, -1, 0], 3, 5e-5, 'eq_point_set_symm', 6)
        test_predict_converges_to_uniform(4, 1001, [-2, -1, -0.5, 0], 5, 1e-3, 'eq_point_set', 8)
        def test_predict_converges_to_uniform(dim, no_grid_points, z_values, tol_verify_greater, tol_verify_equal, eq_point_set_type, eq_point_set_arg):
            sgf = HyperhemisphericalGridFilter(no_grid_points, dim)
            M = np.eye(dim)
            Z = np.array(z_values).reshape(-1, 1)
            bd = BinghamDistribution(Z, M)
            bd.F = bd.F * bd.integrate_numerically()
            sgf.set_state(HyperhemisphericalGridDistribution.from_distribution(bd, no_grid_points))

            # Verify that it is not a uniform distribution
            assert sum(abs(np.diff(sgf.get_estimate().grid_values.T))) > tol_verify_greater

            # Predict 10 times with VM-distributed noise
            def trans(xkk, xk):
                return 2 * np.hstack([HypersphericalMixture([VonMisesFisherDistribution(xk[:, i], 1), VonMisesFisherDistribution(-xk[:, i], 1)], [0.5, 0.5]).pdf(xkk) for i in range(xk.shape[1])])

            sdsd = SdHalfCondSdHalfGridDistribution.from_function(trans, no_grid_points, True, eq_point_set_type, eq_point_set_arg)
            manifold_size = sgf.get_estimate().get_manifold_size()

            for i in range(10):
                values_alternative_formula = (manifold_size / sgf.get_estimate().get_grid().shape[1]) * np.sum(sgf.get_estimate().grid_values.T * sdsd.grid_values, axis=1)
                sgf.predict_nonlinear_via_transition_density(sdsd)
                assert np.allclose(sgf.get_estimate().grid_values, values_alternative_formula, atol=1e-12)

            # Verify that it is approximately uniform now
            assert np.isclose(sum(abs(np.diff(sgf.get_estimate().grid_values.T))), 0, atol=tol_verify_equal)

if __name__ == '__main__':
    unittest.main()
