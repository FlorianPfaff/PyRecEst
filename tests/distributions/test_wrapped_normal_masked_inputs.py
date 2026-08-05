"""Regression tests for masked wrapped-normal public inputs."""

import numpy as np
import pytest
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)


@pytest.mark.parametrize(
    ("parameter", "kwargs"),
    [
        ("mu", {"mu": np.ma.array(0.3, mask=True), "sigma": 0.5}),
        ("sigma", {"mu": 0.3, "sigma": np.ma.array(0.5, mask=True)}),
        (
            "mu",
            {"mu": [np.ma.array(0.3, mask=True)], "sigma": 0.5},
        ),
    ],
)
def test_wrapped_normal_constructor_rejects_masked_parameters(parameter, kwargs):
    with pytest.raises(ValueError, match=rf"{parameter}.*masked"):
        WrappedNormalDistribution(**kwargs)


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("pdf", {"xs": np.ma.array([0.2], mask=[True])}),
        ("cdf", {"xs": np.ma.array([0.2], mask=[True])}),
        (
            "cdf",
            {
                "xs": np.array([0.2]),
                "starting_point": np.ma.array(0.1, mask=True),
            },
        ),
        (
            "trigonometric_moment",
            {"n": np.ma.array(2, mask=True)},
        ),
    ],
)
def test_wrapped_normal_methods_reject_masked_inputs(method_name, kwargs):
    distribution = WrappedNormalDistribution(0.3, 0.5)

    with pytest.raises(ValueError, match="masked"):
        getattr(distribution, method_name)(**kwargs)


def test_wrapped_normal_from_moment_rejects_masked_moment():
    with pytest.raises(ValueError, match="masked"):
        WrappedNormalDistribution.from_moment(
            np.ma.array(0.5 + 0.2j, mask=True)
        )


def test_wrapped_normal_accepts_fully_unmasked_wrappers():
    distribution = WrappedNormalDistribution(
        np.ma.array(0.3, mask=False),
        np.ma.array(0.5, mask=False),
    )

    density = distribution.pdf(np.ma.array([0.2], mask=[False]))
    cumulative = distribution.cdf(
        np.ma.array([0.2], mask=[False]),
        starting_point=np.ma.array(0.1, mask=False),
    )
    moment = distribution.trigonometric_moment(
        np.ma.array(2, mask=False)
    )
    reconstructed = WrappedNormalDistribution.from_moment(
        np.ma.array(0.5 + 0.2j, mask=False)
    )

    assert np.isfinite(np.asarray(density)).all()
    assert np.isfinite(np.asarray(cumulative)).all()
    assert np.isfinite(np.asarray(moment)).all()
    assert np.isfinite(np.asarray(reconstructed.sigma)).all()
