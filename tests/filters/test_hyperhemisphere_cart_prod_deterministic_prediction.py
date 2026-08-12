import unittest

import numpy as np
import pyrecest.backend  # pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array  # pylint: disable=no-name-in-module,no-member
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from pyrecest.filters.hyperhemisphere_cart_prod_particle_filter import (
    HyperhemisphereCartProdParticleFilter,
)


class HyperhemisphereCartProdDeterministicPredictionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__  # pylint: disable=no-name-in-module,no-member
        in ("jax", "pytorch"),
        reason="Backend not supported",
    )
    def test_prediction_without_noise_applies_component_function(self):
        particles = array(
            [
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        state = HyperhemisphereCartProdDiracDistribution(
            particles,
            dim_hemisphere=2,
            n_hemispheres=2,
        )
        particle_filter = HyperhemisphereCartProdParticleFilter(2, 2, 2)
        particle_filter.set_state(state)
        prior_weights = state.w.copy()

        def swap_first_two_coordinates(component_particles):
            return component_particles[:, [1, 0, 2]]

        particle_filter.predict_nonlinear_each_part(swap_first_two_coordinates)

        expected_particles = array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(
            particle_filter.filter_state.d, expected_particles
        )
        np.testing.assert_array_equal(particle_filter.filter_state.w, prior_weights)


if __name__ == "__main__":
    unittest.main()
