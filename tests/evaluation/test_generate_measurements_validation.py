import numpy as np
import pytest
from pyrecest.evaluation.generate_measurements import (
    _as_nonnegative_measurement_count,
    _as_shapely_scalar,
    generate_n_measurements_PPP,
)


@pytest.mark.parametrize(
    ("area", "intensity_lambda", "message"),
    [
        (-1.0, 1.0, "area"),
        (1.0, -0.5, "intensity_lambda"),
        (float("inf"), 1.0, "area"),
        (1.0, float("nan"), "intensity_lambda"),
    ],
)
def test_generate_n_measurements_ppp_rejects_invalid_rate_inputs(
    area, intensity_lambda, message
):
    with pytest.raises(ValueError, match=message):
        generate_n_measurements_PPP(area, intensity_lambda)


@pytest.mark.parametrize(
    "measurement_count",
    [
        "3",
        b"3",
        np.str_("3"),
        np.bytes_(b"3"),
        np.array("3"),
        np.array(b"3"),
    ],
)
def test_measurement_count_rejects_text_like_integer_scalars(measurement_count):
    with pytest.raises(ValueError, match="measurement count"):
        _as_nonnegative_measurement_count(measurement_count)


def test_shapely_scalar_accepts_single_value_arrays():
    assert _as_shapely_scalar(np.array([1.25]), "groundtruth x") == pytest.approx(1.25)


@pytest.mark.parametrize(
    "value",
    [
        np.array([1.0, 2.0]),
        np.array([[1.0], [2.0]]),
        True,
        "1.0",
        np.inf,
    ],
)
def test_shapely_scalar_rejects_ambiguous_or_invalid_values(value):
    with pytest.raises(ValueError, match="groundtruth x must be a finite scalar"):
        _as_shapely_scalar(value, "groundtruth x")


def test_generate_n_measurements_ppp_returns_python_int_for_zero_rate():
    count = generate_n_measurements_PPP(0.0, 123.0)

    assert count == 0
    assert isinstance(count, int)
