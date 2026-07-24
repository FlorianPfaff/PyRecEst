"""Regression tests for tuple-valued custom-distribution integrals."""

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
