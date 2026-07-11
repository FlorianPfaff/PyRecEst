import math

import numpy as np
import pytest
from pyrecest import backend
from pyrecest.backend import all as backend_all
from pyrecest.backend import allclose, array, isfinite, sum
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    GaussVonMisesDistribution,
)
from scipy.special import ive


def _large_kappa_distribution():
    return GaussVonMisesDistribution(
        mu=array([0.0]),
        P=array([[1.0]]),
        alpha=0.3,
        beta=array([0.0]),
        Gamma=array([[0.0]]),
        kappa=1000.0,
    )


def test_large_kappa_mode_pdf_is_finite():
    dist = _large_kappa_distribution()

    value = dist.pdf(dist.mode())
    expected = 1.0 / (
        math.sqrt(2.0 * math.pi)
        * 2.0
        * math.pi
        * ive(0, dist.kappa)
    )

    assert math.isfinite(value)
    assert np.isclose(value, expected, rtol=1e-12, atol=0.0)


@pytest.mark.skipif(
    backend.__backend_name__ == "jax",
    reason="Deterministic Horwood sampling is not supported on JAX.",
)
def test_large_kappa_horwood_samples_are_finite():
    dist = _large_kappa_distribution()
    points, weights = dist.sample_deterministic_horwood()

    assert bool(backend_all(isfinite(points)))
    assert bool(backend_all(isfinite(weights)))
    assert bool(allclose(sum(weights), array(1.0), rtol=1e-12, atol=1e-12))
