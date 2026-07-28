"""Regression tests for custom-distribution integral normalization."""

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.distributions.circle.custom_circular_distribution import (
    CustomCircularDistribution,
)


class _TupleIntegralCircularDistribution(CustomCircularDistribution):
    """Circular test distribution emulating ``scipy.integrate.quad`` output."""

    def integrate(self, integration_boundaries=None):
        del integration_boundaries
        return 2.0 * self.scale_by, 1.0e-12


class _FixedIntegralCircularDistribution(CustomCircularDistribution):
    """Circular test distribution returning a configured integral value."""

    def __init__(self, integral):
        super().__init__(lambda xs: xs * 0.0 + 1.0)
        self.integral = integral

    def integrate(self, integration_boundaries=None):
        del integration_boundaries
        return self.integral


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Custom-distribution normalization is NumPy-only",
)
def test_normalize_accepts_value_error_integral_tuple():
    distribution = _TupleIntegralCircularDistribution(lambda xs: xs * 0.0 + 1.0)

    normalized = distribution.normalize(verify=True)

    assert normalized.scale_by == pytest.approx(0.5)
    assert normalized.integrate()[0] == pytest.approx(1.0)
    assert distribution.scale_by == pytest.approx(1.0)


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Custom-distribution normalization is NumPy-only",
)
@pytest.mark.parametrize(
    "integral",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        np.array([1.0]),
        "1.0",
        True,
        np.timedelta64(1, "ns"),
        (0.0, 1.0e-12),
    ),
)
def test_normalize_rejects_invalid_integrals(integral):
    distribution = _FixedIntegralCircularDistribution(integral)

    with pytest.raises(ValueError, match="finite positive scalar"):
        distribution.normalize()

    assert distribution.scale_by == pytest.approx(1.0)
