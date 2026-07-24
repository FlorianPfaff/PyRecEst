import numpy as np
import pytest
from pyrecest.evaluation.generate_measurements import (
    _as_nonnegative_measurement_count,
    _as_shapely_scalar,
    generate_n_measurements_PPP,
)

_TEMPORAL_SCALARS = (
    np.datetime64("1970-01-01T00:00:00.000000001"),
    np.timedelta64(1, "ns"),
)


@pytest.mark.parametrize("value", _TEMPORAL_SCALARS)
def test_measurement_count_rejects_temporal_scalar_payloads(value):
    with pytest.raises(ValueError, match="measurement count"):
        _as_nonnegative_measurement_count(value)


@pytest.mark.parametrize("value", _TEMPORAL_SCALARS)
def test_shapely_scalar_rejects_temporal_scalar_payloads(value):
    with pytest.raises(ValueError, match="groundtruth x must be a finite scalar"):
        _as_shapely_scalar(value, "groundtruth x")


@pytest.mark.parametrize("value", _TEMPORAL_SCALARS)
@pytest.mark.parametrize("argument", ("area", "intensity_lambda"))
def test_poisson_measurement_count_rejects_temporal_scalar_payloads(value, argument):
    kwargs = {"area": 1.0, "intensity_lambda": 1.0}
    kwargs[argument] = value

    with pytest.raises(ValueError, match=argument):
        generate_n_measurements_PPP(**kwargs)


@pytest.mark.parametrize("value", _TEMPORAL_SCALARS)
def test_object_wrapped_temporal_scalars_are_also_rejected(value):
    wrapped = np.array(value, dtype=object)

    with pytest.raises(ValueError, match="measurement count"):
        _as_nonnegative_measurement_count(wrapped)
    with pytest.raises(ValueError, match="groundtruth x must be a finite scalar"):
        _as_shapely_scalar(wrapped, "groundtruth x")
    with pytest.raises(ValueError, match="area"):
        generate_n_measurements_PPP(wrapped, 1.0)
