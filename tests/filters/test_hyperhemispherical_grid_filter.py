import unittest
import warnings

import pyrecest.backend
from pyrecest.backend import array, linalg
from pyrecest.distributions import (
    HyperhemisphericalWatsonDistribution,
    SdHalfCondSdHalfGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
)
from pyrecest.filters.hyperhemispherical_grid_filter import (
    HyperhemisphericalGridFilter,
)


class TestHyperhemisphericalGridFilter(unittest.TestCase):
    def setUp(self):
        self.n_grid = 50
        self.dim = 2  # manifold dim (S2-half)
        self.watson_init = HyperhemisphericalWatsonDistribution(
            array([0.0, 0.0, 1.0]), 5.0
        )
        self.watson_sys = WatsonDistribution(array([0.0, 0.0, 1.0]), 3.0)
        self.watson_meas = HyperhemisphericalWatsonDistribution(
            array([0.0, 0.0, 1.0]), 3.0
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_constructor(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        self.assertEqual(f.filter_state.grid.shape[0], self.n_grid)
        self.assertEqual(f.filter_state.grid.shape[1], self.dim + 1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_set_state_from_distribution(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.filter_state = self.watson_init
        self.assertIsInstance(f.filter_state, HyperhemisphericalGridDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_set_state_from_grid_distribution(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.filter_state = self.watson_init
        gd = f.filter_state
        # Setting from a grid distribution with the same grid should work silently.
        f2 = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f2.filter_state = gd
        self.assertIsInstance(f2.filter_state, HyperhemisphericalGridDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_get_point_estimate_uniform(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        # For a uniform distribution, the estimate should still be a unit vector.
        p = f.get_point_estimate()
        self.assertAlmostEqual(float(linalg.norm(p)), 1.0, places=5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_get_point_estimate_concentrated(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.filter_state = self.watson_init
        p = f.get_point_estimate()
        self.assertAlmostEqual(float(linalg.norm(p)), 1.0, places=5)
        # Should be close to [0, 0, 1]
        self.assertGreater(float(p[-1]), 0.9)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_get_point_estimate_matches_s3_scatter_mode(self):
        f = HyperhemisphericalGridFilter(8, 3)
        grid = f.filter_state.get_grid()
        weights = array([0.3, 0.4, 0.6, 1.1, 1.7, 0.8, 0.5, 1.3])
        f.filter_state = HyperhemisphericalGridDistribution(grid, weights)

        gd_full = f.filter_state.to_full_sphere()
        full_weights = gd_full.grid_values / pyrecest.backend.sum(gd_full.grid_values)
        scatter = gd_full.grid.T @ (gd_full.grid * full_weights[:, None])
        scatter = 0.5 * (scatter + scatter.T)
        _, eigenvectors = linalg.eigh(scatter)
        expected = eigenvectors[:, -1]
        if expected[-1] < 0:
            expected = -expected

        p = f.get_point_estimate()

        self.assertAlmostEqual(float(linalg.norm(p)), 1.0, places=5)
        self.assertGreaterEqual(float(p[-1]), 0.0)
        self.assertTrue(pyrecest.backend.allclose(p, expected, atol=1e-10))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_identity(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.filter_state = self.watson_init
        z = array([0.0, 0.0, 1.0])
        f.update_identity(self.watson_meas, z)
        p = f.get_point_estimate()
        self.assertAlmostEqual(float(linalg.norm(p)), 1.0, places=5)
        self.assertGreater(float(p[-1]), 0.9)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_sys_noise_to_transition_density(self):
        f_trans = HyperhemisphericalGridFilter.sys_noise_to_transition_density(
            self.watson_sys, self.n_grid
        )
        self.assertIsInstance(f_trans, SdHalfCondSdHalfGridDistribution)
        self.assertEqual(f_trans.grid_values.shape, (self.n_grid, self.n_grid))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_nonlinear_via_transition_density(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.filter_state = self.watson_init
        f_trans = HyperhemisphericalGridFilter.sys_noise_to_transition_density(
            self.watson_sys, self.n_grid
        )
        f.predict_nonlinear_via_transition_density(f_trans)
        p = f.get_point_estimate()
        self.assertAlmostEqual(float(linalg.norm(p)), 1.0, places=5)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_watson(self):
        f = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.filter_state = self.watson_init
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.predict_identity(self.watson_sys)
        p = f.get_point_estimate()
        self.assertAlmostEqual(float(linalg.norm(p)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
