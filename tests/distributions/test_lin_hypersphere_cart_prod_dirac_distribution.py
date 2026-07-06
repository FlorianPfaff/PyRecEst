import numpy as np
from pyrecest.distributions.cart_prod.lin_hypersphere_cart_prod_dirac_distribution import (
    LinHypersphereCartProdDiracDistribution,
)


class ConcreteLinHypersphereCartProdDiracDistribution(
    LinHypersphereCartProdDiracDistribution
):
    def marginalize_linear(self):
        raise NotImplementedError


def test_constructor_preserves_generic_hypersphere_and_linear_dimensions():
    particles = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
        ]
    )

    dist = ConcreteLinHypersphereCartProdDiracDistribution(bound_dim=1, d=particles)

    assert dist.bound_dim == 1
    assert dist.lin_dim == 1
    assert dist.dim == 2
    assert dist.input_dim == particles.shape[-1]
