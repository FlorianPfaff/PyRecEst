import pytest
from pyrecest.distributions import VonMisesDistribution


@pytest.mark.parametrize(
    "mu",
    (
        [0.0, 1.0],
        [[0.0, 1.0]],
        float("nan"),
        float("inf"),
        -float("inf"),
    ),
)
def test_constructor_rejects_non_scalar_or_nonfinite_mean(mu):
    with pytest.raises(ValueError):
        VonMisesDistribution(mu, 2.0)


@pytest.mark.parametrize(
    "mu",
    (
        [0.0, 1.0],
        [[0.0, 1.0]],
        float("nan"),
        float("inf"),
        -float("inf"),
    ),
)
def test_set_mean_rejects_non_scalar_or_nonfinite_mean(mu):
    dist = VonMisesDistribution(0.3, 2.0)

    with pytest.raises(ValueError):
        dist.set_mean(mu)
