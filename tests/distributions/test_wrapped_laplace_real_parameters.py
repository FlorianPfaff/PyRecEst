import pytest
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


@pytest.mark.parametrize("parameter_name", ["lambda_", "kappa_"])
@pytest.mark.parametrize("invalid_value", [True, 1.0 + 1.0j])
def test_wrapped_laplace_rejects_non_real_rate_parameters(
    parameter_name, invalid_value
):
    parameters = {"lambda_": 2.0, "kappa_": 1.3}
    parameters[parameter_name] = invalid_value

    with pytest.raises(ValueError, match=parameter_name):
        WrappedLaplaceDistribution(**parameters)
