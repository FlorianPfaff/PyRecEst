import numpy as np
import pytest
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    GaussVonMisesDistribution,
)


def _make_distribution(alpha):
    return GaussVonMisesDistribution(
        mu=2.0,
        P=1.3,
        alpha=alpha,
        beta=0.0,
        Gamma=0.001,
        kappa=0.7,
    )


@pytest.mark.parametrize(
    "alpha",
    [True, np.bool_(False), float("nan"), float("inf"), -float("inf")],
)
def test_constructor_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        _make_distribution(alpha)


def test_constructor_accepts_and_wraps_numpy_scalar_alpha():
    distribution = _make_distribution(np.float64(-0.5))

    assert np.isclose(distribution.alpha, 2.0 * np.pi - 0.5)
