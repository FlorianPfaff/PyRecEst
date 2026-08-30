import pytest

from pyrecest.backend import array
from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("argument", ["alpha", "beta"])
def test_angular_error_rejects_nonfinite_angles(bad_value, argument):
    alpha = array([0.0])
    beta = array([0.0])
    if argument == "alpha":
        alpha = array([bad_value])
    else:
        beta = array([bad_value])

    with pytest.raises(ValueError, match="finite"):
        AbstractHypertoroidalDistribution.angular_error(alpha, beta)
