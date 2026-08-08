import numpy as np
import pytest
from pyrecest.evaluation import (
    classify_inside_outside,
    surface_band_mask,
    surface_band_probability_from_signed_distance,
)


@pytest.mark.parametrize(
    "values",
    (
        np.ma.array([0.0, 0.1], mask=[False, True]),
        [0.0, np.ma.masked],
        np.array([0.0, np.ma.masked], dtype=object),
    ),
)
def test_surface_numeric_fields_reject_masked_values(values) -> None:
    with pytest.raises(ValueError, match="values"):
        surface_band_mask(values, 0.1)
    with pytest.raises(ValueError, match="values"):
        classify_inside_outside(values)
    with pytest.raises(ValueError, match="distance"):
        surface_band_probability_from_signed_distance(values, [0.1, 0.1], 0.1)
    with pytest.raises(ValueError, match="distance_std"):
        surface_band_probability_from_signed_distance([0.0, 0.1], values, 0.1)


def test_surface_scalar_controls_reject_masked_payloads() -> None:
    masked_float = np.ma.array(0.1, mask=True)
    masked_bool = np.ma.array(True, mask=True)

    with pytest.raises(ValueError, match="threshold"):
        surface_band_mask([0.0], masked_float)
    with pytest.raises(TypeError, match="negative_inside"):
        classify_inside_outside([0.0], negative_inside=masked_bool)
    with pytest.raises(ValueError, match="epsilon"):
        surface_band_probability_from_signed_distance([0.0], [0.1], masked_float)
    with pytest.raises(ValueError, match="min_std"):
        surface_band_probability_from_signed_distance(
            [0.0], [0.1], 0.1, min_std=masked_float
        )


def test_surface_numeric_fields_accept_clear_mask_wrappers() -> None:
    values = np.ma.array([-0.05, 0.2], mask=False)
    distance_std = np.ma.array([0.1, 0.2], mask=False)

    np.testing.assert_array_equal(surface_band_mask(values, 0.1), [True, False])
    np.testing.assert_array_equal(classify_inside_outside(values), [-1, 1])
    probabilities = surface_band_probability_from_signed_distance(
        values, distance_std, 0.1
    )
    assert probabilities.shape == (2,)
    assert np.isfinite(probabilities).all()
