import numpy as np
import pytest
from pyrecest.distributions import WrappedNormalDistribution


@pytest.mark.parametrize(
    "n",
    [
        np.timedelta64(3, "ns"),
        np.timedelta64(3, "us"),
        np.datetime64(3, "ns"),
        np.array(np.timedelta64(3, "ns")),
    ],
)
def test_sample_rejects_temporal_counts(n):
    distribution = WrappedNormalDistribution(0.2, 0.5)

    with pytest.raises(ValueError, match="positive integer"):
        distribution.sample(n)


def test_sample_preserves_numpy_integer_counts():
    distribution = WrappedNormalDistribution(0.2, 0.5)

    samples = distribution.sample(np.int64(3))

    assert np.asarray(samples).shape == (3,)
