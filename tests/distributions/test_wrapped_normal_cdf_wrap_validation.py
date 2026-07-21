"""Regression tests for wrapped-normal CDF truncation-order validation."""

import numpy as np
import pytest
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)


@pytest.mark.parametrize("n_wraps", [-1, True, False, 1.5])
def test_wrapped_normal_cdf_rejects_invalid_wrap_counts(n_wraps):
    distribution = WrappedNormalDistribution(6.1, 0.5)

    with pytest.raises(ValueError, match="non-negative integer"):
        distribution.cdf(np.array([0.2]), n_wraps=n_wraps)


def test_wrapped_normal_cdf_accepts_numpy_integer_wrap_count():
    distribution = WrappedNormalDistribution(6.1, 0.5)

    result = distribution.cdf(np.array([0.2]), n_wraps=np.int64(4))

    assert np.isfinite(np.asarray(result)).all()
