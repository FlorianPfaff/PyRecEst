import numpy as np
import pytest
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution


def _standard_gaussian():
    return GaussianDistribution(array([0.0, 0.0]), diag(array([1.0, 1.0])))


@pytest.mark.parametrize(
    "sample_count",
    [
        np.timedelta64(4, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000004", "ns"),
        np.asarray(np.timedelta64(4, "ns")),
        np.asarray(np.datetime64("1970-01-01T00:00:00.000000004", "ns")),
        np.array(np.timedelta64(4, "ns"), dtype=object),
        np.array(np.datetime64("1970-01-01T00:00:00.000000004", "ns"), dtype=object),
    ],
)
def test_gaussian_sample_rejects_temporal_sample_counts(sample_count):
    gaussian = _standard_gaussian()

    with pytest.raises(ValueError, match="positive integer"):
        gaussian.sample(sample_count)
