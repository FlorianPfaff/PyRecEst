import numpy as np
import pytest
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


@pytest.mark.backend_portable
@pytest.mark.parametrize("method_name", ["pdf", "cdf"])
@pytest.mark.parametrize(
    "xs",
    [
        True,
        np.bool_(False),
        float("nan"),
        float("inf"),
        -float("inf"),
        [0.0, float("nan")],
        1.0 + 2.0j,
        "0.5",
    ],
)
def test_wrapped_cauchy_rejects_nonfinite_or_nonreal_evaluation_points(method_name, xs):
    distribution = WrappedCauchyDistribution(0.0, 0.5)

    with pytest.raises(ValueError, match="xs must contain only finite real values"):
        getattr(distribution, method_name)(xs)


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    ("parameter_name", "value"),
    [
        ("mu", True),
        ("mu", 1.0 + 2.0j),
        ("gamma", np.bool_(True)),
        ("gamma", "0.5"),
    ],
)
def test_wrapped_cauchy_rejects_nonreal_scalar_parameters(parameter_name, value):
    parameters = {"mu": 0.0, "gamma": 0.5}
    parameters[parameter_name] = value

    with pytest.raises(ValueError, match=parameter_name):
        WrappedCauchyDistribution(**parameters)
