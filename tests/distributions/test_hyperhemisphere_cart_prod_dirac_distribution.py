import unittest

import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)


class HyperhemisphereCartProdDiracDistributionTest(unittest.TestCase):
    def test_constructor_rejects_missing_dimensions(self):
        particles = array([[1.0, 0.0, 0.0]])

        with self.assertRaisesRegex(ValueError, "Hemisphere dimension"):
            HyperhemisphereCartProdDiracDistribution(particles, n_hemispheres=1)

        with self.assertRaisesRegex(ValueError, "Number of hemispheres"):
            HyperhemisphereCartProdDiracDistribution(particles, dim_hemisphere=2)

    def test_apply_function_requires_multiple_input_support(self):
        particles = array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        dist = HyperhemisphereCartProdDiracDistribution(
            particles, dim_hemisphere=2, n_hemispheres=2
        )

        with self.assertRaisesRegex(ValueError, "multiple inputs"):
            dist.apply_function_component_wise(
                lambda xs: xs[0], f_supports_multiple=False
            )

    def test_apply_function_component_wise_preserves_flat_storage(self):
        particles = array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        dist = HyperhemisphereCartProdDiracDistribution(
            particles, dim_hemisphere=2, n_hemispheres=2
        )

        transformed = dist.apply_function_component_wise(lambda xs: xs)

        npt.assert_allclose(transformed.d, particles)


if __name__ == "__main__":
    unittest.main()
