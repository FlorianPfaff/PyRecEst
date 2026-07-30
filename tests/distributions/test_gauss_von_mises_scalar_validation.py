"""Regression tests for strict Gauss-von-Mises scalar validation."""

import numpy as np
import pytest
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    GaussVonMisesDistribution,
    _validate_positive_sample_count,
)


def _make_distribution(**overrides):
    parameters = {
        "mu": 2.0,
        "P": 1.3,
        "alpha": 0.2,
        "beta": 0.0,
        "Gamma": 0.001,
        "kappa": 0.7,
    }
    parameters.update(overrides)
    return GaussVonMisesDistribution(**parameters)


@pytest.mark.parametrize(
    "invalid_count",
    [
        np.timedelta64(3, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000003"),
        np.array(np.timedelta64(3, "ns"), dtype=object),
        np.ma.array(3, mask=True),
        "3",
        3.0 + 0.0j,
    ],
)
def test_rejects_nonreal_or_masked_sample_counts(invalid_count):
    with pytest.raises(ValueError, match="integer"):
        _validate_positive_sample_count(invalid_count)


@pytest.mark.parametrize("parameter_name", ["alpha", "kappa"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        np.timedelta64(3, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000003"),
        np.array(np.timedelta64(3, "ns"), dtype=object),
        np.ma.array(0.5, mask=True),
        "0.5",
        0.5 + 0.0j,
    ],
)
def test_constructor_rejects_nonreal_or_masked_scalars(parameter_name, invalid_value):
    with pytest.raises(ValueError, match=parameter_name):
        _make_distribution(**{parameter_name: invalid_value})


def test_unmasked_numeric_scalars_remain_supported():
    assert _validate_positive_sample_count(np.ma.array(3, mask=False)) == 3

    distribution = _make_distribution(
        alpha=np.ma.array(0.5, mask=False),
        kappa=np.ma.array(0.7, mask=False),
    )

    assert np.isclose(distribution.alpha, 0.5)
    assert np.isclose(distribution.kappa, 0.7)
