import numpy as np
import pytest
from pyrecest.utils.metrics import chi_square_confidence_bounds

_TEMPORAL_COUNTS = (
    np.timedelta64(2, "ns"),
    np.timedelta64(2, "us"),
    np.datetime64("1970-01-01T00:00:00.000000002", "ns"),
)


@pytest.mark.parametrize("value", _TEMPORAL_COUNTS)
def test_chi_square_bounds_reject_temporal_degrees_of_freedom(value):
    with pytest.raises(
        ValueError,
        match="degrees_of_freedom must be a positive integer",
    ):
        chi_square_confidence_bounds(value)


@pytest.mark.parametrize("value", _TEMPORAL_COUNTS)
def test_chi_square_bounds_reject_temporal_sample_counts(value):
    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        chi_square_confidence_bounds(2, n_samples=value)
