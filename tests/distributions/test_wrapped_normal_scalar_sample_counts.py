import numpy as np
import pytest

from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution


@pytest.mark.parametrize(
    "count",
    [
        np.array(3, dtype=np.int64),
        np.array(3, dtype=np.uint64),
        array(3),
        np.ma.array(3, mask=False),
    ],
)
def test_sample_accepts_exact_scalar_integer_arrays(count):
    distribution = WrappedNormalDistribution(0.2, 0.5)

    samples = distribution.sample(count)

    assert np.asarray(samples).shape == (3,)


def test_sample_rejects_masked_scalar_count():
    distribution = WrappedNormalDistribution(0.2, 0.5)

    with pytest.raises(ValueError, match="positive integer"):
        distribution.sample(np.ma.array(3, mask=True))
