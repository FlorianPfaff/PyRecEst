import numpy as np
import pytest
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution


@pytest.mark.parametrize(
    "sample_count",
    [
        np.timedelta64(4, "ns"),
        np.timedelta64(4, "us"),
    ],
)
def test_sampling_rejects_numpy_timedelta_counts(sample_count):
    dist = EllipsoidalBallUniformDistribution(
        array([0.0, 0.0]), diag(array([1.0, 1.0]))
    )

    with pytest.raises(ValueError, match="positive integer"):
        dist.sample(sample_count)
