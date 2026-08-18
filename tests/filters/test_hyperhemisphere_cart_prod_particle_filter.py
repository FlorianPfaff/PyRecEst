import unittest

import pyrecest.backend  # pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (  # pylint: disable=no-name-in-module,no-member,redefined-builtin
    abs,
    allclose,
    array,
    hstack,
    ones,
    sum,
    zeros,
)
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from pyrecest.filters.hyperhemisphere_cart_prod_particle_filter import (
    HyperhemisphereCartProdParticleFilter,
)


class HyperHemisphereCartProdParticleFilterTest(unittest.TestCase):
    def test_init(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        self.assertEqual(
            pf.filter_state.d.shape,
            (n_particles, (dim_hemisphere + 1) * n_hemispheres),
        )
        expected_component = hstack(
            (
                zeros((n_particles, dim_hemisphere)),
                ones((n_particles, 1)),
            )
        )
        expected_particles = hstack([expected_component] * n_hemispheres)
        self.assertTrue(bool(allclose(pf.filter_state.d, expected_particles)))

    def test_constructor_rejects_invalid_particle_count(self):
        for n_particles in (True, 0, -1, 1.5):
            with self.subTest(n_particles=n_particles):
                with self.assertRaisesRegex(ValueError, "n_particles"):
                    HyperhemisphereCartProdParticleFilter(n_particles, 3, 2)

    def test_constructor_rejects_invalid_hemisphere_dimension(self):
        for dim_hemisphere in (True, 0, -1, 1.5):
            with self.subTest(dim_hemisphere=dim_hemisphere):
                with self.assertRaisesRegex(ValueError, "dim_hemisphere"):
                    HyperhemisphereCartProdParticleFilter(5, dim_hemisphere, 2)

    def test_constructor_rejects_invalid_hemisphere_count(self):
        for n_hemispheres in (True, 0, -1, 1.5):
            with self.subTest(n_hemispheres=n_hemispheres):
                with self.assertRaisesRegex(ValueError, "n_hemispheres"):
                    HyperhemisphereCartProdParticleFilter(5, 3, n_hemispheres)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__  # pylint: disable=no-name-in-module,no-member
        in (
            "jax",
            "pytorch",
        ),
        reason="Backend not supported'",
    )
    def test_set_state(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        self.assertEqual(
            pf.filter_state.d.shape,
            (n_particles, (dim_hemisphere + 1) * n_hemispheres),
        )
        pf.filter_state = VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0)

    def test_set_state_rejects_invalid_distribution(self):
        pf = HyperhemisphereCartProdParticleFilter(5, 2, 2)

        with self.assertRaisesRegex(TypeError, "AbstractHyperhemispherical"):
            pf.set_state(object())

    def test_set_state_accepts_matching_cart_prod_dirac_state(self):
        pf = HyperhemisphereCartProdParticleFilter(5, 2, 2)
        new_state = HyperhemisphereCartProdDiracDistribution(
            array([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]] * 5),
            dim_hemisphere=2,
            n_hemispheres=2,
        )

        pf.set_state(new_state)

        self.assertEqual(pf.filter_state.d.shape, (5, 6))
        self.assertEqual(pf.filter_state.dim_hemisphere, 2)
        self.assertEqual(pf.filter_state.n_hemispheres, 2)

    def test_set_state_rejects_cart_prod_topology_mismatch(self):
        pf = HyperhemisphereCartProdParticleFilter(5, 2, 2)
        new_state = HyperhemisphereCartProdDiracDistribution(
            array([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]] * 5),
            dim_hemisphere=1,
            n_hemispheres=3,
        )

        with self.assertRaisesRegex(ValueError, "topology"):
            pf.set_state(new_state)

    def test_filter_state_rejects_wrong_dimension_distribution(self):
        pf = HyperhemisphereCartProdParticleFilter(5, 2, 2)

        with self.assertRaisesRegex(ValueError, "dimension"):
            pf.filter_state = VonMisesFisherDistribution(
                array([0.0, 0.0, 0.0, 1.0]), 1.0
            )

    def test_predict_each_part_rejects_unsupported_options(self):
        pf = HyperhemisphereCartProdParticleFilter(5, 2, 2)

        def identity_function(x):
            return x

        with self.assertRaisesRegex(ValueError, "vectorized"):
            pf.predict_nonlinear_each_part(
                identity_function, function_is_vectorized=False
            )

        with self.assertRaisesRegex(ValueError, "Noise dimension"):
            pf.predict_nonlinear_each_part(
                identity_function,
                VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0),
            )

        with self.assertRaisesRegex(ValueError, "shifting"):
            pf.predict_nonlinear_each_part(
                identity_function, shift_instead_of_add=False
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__  # pylint: disable=no-name-in-module,no-member
        in (
            "jax",
            "pytorch",
        ),
        reason="Backend not supported'",
    )
    def test_predict(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        pf.filter_state = VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0)

        noise_distribution = VonMisesFisherDistribution(
            array([0.0, 0.0, 0.0, 1.0]), 1.0
        )

        def identity_function(x):
            return x

        pf.predict_nonlinear_each_part(identity_function, noise_distribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__  # pylint: disable=no-name-in-module,no-member
        in (
            "jax",
            "pytorch",
        ),
        reason="Backend not supported'",
    )
    def test_update(self):
        n_particles = 1000
        dim_hemisphere = 3
        n_hemispheres = 2
        pf = HyperhemisphereCartProdParticleFilter(
            n_particles, dim_hemisphere, n_hemispheres
        )
        pf.filter_state = VonMisesFisherDistribution(array([0.0, 0.0, 0.0, 1.0]), 1.0)

        def likelihood_function(x):
            return abs(sum(x, axis=1))  # noqa: A001

        pf.update_nonlinear_using_likelihood(likelihood_function)


if __name__ == "__main__":
    unittest.main()
