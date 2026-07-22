import numpy as np
import pytest
from pyrecest.distributions.circle.wrapped_exponential_distribution import (
    WrappedExponentialDistribution,
)


@pytest.mark.parametrize(
    "xs",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        [0.0, float("nan")],
        True,
        np.bool_(False),
        1.0 + 2.0j,
        "1.0",
    ],
)
def test_wrapped_exponential_pdf_rejects_nonfinite_or_nonreal_points(xs):
    distribution = WrappedExponentialDistribution(2.0)

    with pytest.raises(ValueError, match="finite real"):
        distribution.pdf(xs)
