import unittest
import warnings

import pyrecest.backend
from pyrecest.backend import allclose, array
from pyrecest.distributions import HyperhemisphericalWatsonDistribution
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
)
from pyrecest.filters.hyperhemispherical_grid_filter import (
    HyperhemisphericalGridFilter,
)
from pyrecest.models import (
    GridLikelihoodMeasurementModel,
    GridTransitionDensityFactoryModel,
    GridTransitionDensityModel,
)


class TestGridFilterModelAdapters(unittest.TestCase):
    def setUp(self):
        self.n_grid = 50
        self.dim = 2
        self.initial_distribution = HyperhemisphericalWatsonDistribution(
            array([0.0, 0.0, 1.0]), 5.0
        )
        self.measurement = array([0.0, 0.0, 1.0])
        self.system_noise = WatsonDistribution(array([0.0, 0.0, 1.0]), 3.0)

    def _initialized_filter(self):
        filter_instance = HyperhemisphericalGridFilter(self.n_grid, self.dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filter_instance.filter_state = self.initial_distribution
        return filter_instance

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_update_model_matches_update_nonlinear(self):
        reference_filter = self._initialized_filter()
        model_filter = self._initialized_filter()

        def likelihood(measurement, grid):
            return HyperhemisphericalWatsonDistribution(measurement, 2.0).pdf(grid)

        reference_filter.update_nonlinear(likelihood, self.measurement)
        model_filter.update_model(
            GridLikelihoodMeasurementModel(likelihood), self.measurement
        )

        self.assertTrue(
            bool(
                allclose(
                    reference_filter.filter_state.grid_values,
                    model_filter.filter_state.grid_values,
                )
            )
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_model_matches_transition_density_prediction(self):
        reference_filter = self._initialized_filter()
        model_filter = self._initialized_filter()
        transition_density = (
            HyperhemisphericalGridFilter.sys_noise_to_transition_density(
                self.system_noise, self.n_grid
            )
        )

        reference_filter.predict_nonlinear_via_transition_density(transition_density)
        model_filter.predict_model(GridTransitionDensityModel(transition_density))

        self.assertTrue(
            bool(
                allclose(
                    reference_filter.filter_state.grid_values,
                    model_filter.filter_state.grid_values,
                )
            )
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on JAX backend",
    )
    def test_predict_model_accepts_transition_density_factory(self):
        reference_filter = self._initialized_filter()
        model_filter = self._initialized_filter()

        transition_density = (
            HyperhemisphericalGridFilter.sys_noise_to_transition_density(
                self.system_noise, self.n_grid
            )
        )
        reference_filter.predict_nonlinear_via_transition_density(transition_density)

        def transition_density_factory(filter_instance):
            return HyperhemisphericalGridFilter.sys_noise_to_transition_density(
                self.system_noise,
                filter_instance.filter_state.grid_values.shape[0],
            )

        model_filter.predict_model(
            GridTransitionDensityFactoryModel(transition_density_factory)
        )

        self.assertTrue(
            bool(
                allclose(
                    reference_filter.filter_state.grid_values,
                    model_filter.filter_state.grid_values,
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
