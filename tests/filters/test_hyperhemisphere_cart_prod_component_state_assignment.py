import unittest

import numpy as np

import pyrecest.backend  # pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array  # pylint: disable=no-name-in-module,no-member
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_watson_distribution import (
    HyperhemisphericalWatsonDistribution,
)
from pyrecest.filters.hyperhemisphere_cart_prod_particle_filter import (
    HyperhemisphereCartProdParticleFilter,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__  # pylint: disable=no-name-in-module,no-member
    in ("jax", "pytorch"),
    reason="Backend not supported",
)
class HyperhemisphereCartProdComponentStateAssignmentTest(unittest.TestCase):
    @staticmethod
    def _filter_with_nonuniform_weights():
        particle_filter = HyperhemisphereCartProdParticleFilter(4, 2, 2)
        particles = array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )
        state = HyperhemisphereCartProdDiracDistribution(
            particles,
            w=array([0.7, 0.1, 0.1, 0.1]),
            dim_hemisphere=2,
            n_hemispheres=2,
        )
        particle_filter.set_state(state)
        return particle_filter

    @staticmethod
    def _component_distribution():
        return HyperhemisphericalWatsonDistribution(
            array([0.0, 0.0, 1.0]), 2.0
        )

    def test_set_state_expands_component_distribution_across_product(self):
        particle_filter = self._filter_with_nonuniform_weights()

        particle_filter.set_state(self._component_distribution())

        self.assertEqual(particle_filter.filter_state.d.shape, (4, 6))
        self.assertEqual(
            particle_filter.filter_state.as_component_array().shape, (4, 2, 3)
        )
        np.testing.assert_allclose(
            particle_filter.filter_state.w, np.full(4, 0.25)
        )
        self.assertTrue(
            np.all(
                np.asarray(particle_filter.filter_state.as_component_array())[..., -1]
                >= 0.0
            )
        )

    def test_filter_state_distribution_assignment_resets_weights(self):
        particle_filter = self._filter_with_nonuniform_weights()

        particle_filter.filter_state = self._component_distribution()

        np.testing.assert_allclose(
            particle_filter.filter_state.w, np.full(4, 0.25)
        )


if __name__ == "__main__":
    unittest.main()
