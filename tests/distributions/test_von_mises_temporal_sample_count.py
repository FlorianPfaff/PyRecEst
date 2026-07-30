import numpy as np
import pytest
from pyrecest.distributions import VonMisesDistribution


@pytest.mark.parametrize(
    "n",
    (
        np.timedelta64(3, "ns"),
        np.timedelta64(3, "us"),
    ),
)
def test_von_mises_sample_rejects_temporal_counts(n):
    distribution = VonMisesDistribution(0.3, 1.0)

    with pytest.raises(ValueError, match="positive integer"):
        distribution.sample(n)
