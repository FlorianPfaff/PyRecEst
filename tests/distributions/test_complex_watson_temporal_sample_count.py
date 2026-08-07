import numpy as np
import pyrecest.backend
import pytest
from pyrecest.backend import array, complex128
from pyrecest.distributions import ComplexWatsonDistribution

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Complex Watson sampling is not supported on the JAX backend",
)


@pytest.mark.parametrize(
    "invalid_count",
    [
        np.timedelta64(3, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000003", "ns"),
        np.array(np.timedelta64(3, "ns"), dtype=object),
    ],
)
def test_complex_watson_rejects_temporal_sample_counts(invalid_count):
    distribution = ComplexWatsonDistribution(
        array([1.0, 0.0], dtype=complex128),
        2.0,
    )

    with pytest.raises(ValueError, match="integer"):
        distribution.sample(invalid_count)
