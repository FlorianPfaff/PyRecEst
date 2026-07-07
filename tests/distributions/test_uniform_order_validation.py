import pytest
from pyrecest.backend import array
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)


def test_uniform_moment_rejects_invalid_order():
    dist = HypertoroidalUniformDistribution(2)

    for order in ("0", True, array([0]), 1.5):
        with pytest.raises(ValueError):
            dist.trigonometric_moment(order)
