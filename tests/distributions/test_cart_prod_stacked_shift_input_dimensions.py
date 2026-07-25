from pyrecest.backend import allclose, array
from pyrecest.distributions import (
    GaussianDistribution,
    HyperhemisphericalWatsonDistribution,
)
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


def test_shift_partitions_offsets_by_component_input_dimensions():
    distribution = CartProdStackedDistribution(
        [
            GaussianDistribution(array([1.0]), array([[1.0]])),
            HyperhemisphericalWatsonDistribution(array([0.0, 0.0, 1.0]), 2.0),
        ]
    )

    assert distribution.dim == 3
    assert distribution.input_dim == 4

    shifted = distribution.shift(array([2.0, 1.0, 0.0, 0.0]))

    assert allclose(shifted.dists[0].mu, array([3.0]))
    assert allclose(shifted.dists[1].mu, array([1.0, 0.0, 0.0]))
