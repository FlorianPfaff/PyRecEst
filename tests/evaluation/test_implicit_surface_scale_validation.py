import numpy as np
import pytest
from pyrecest.evaluation import (
    surface_band_mask,
    surface_band_probability_from_signed_distance,
)


def test_surface_band_scale_controls_reject_bools_and_non_scalars() -> None:
    invalid_values = (True, np.array([0.1]))

    for invalid_value in invalid_values:
        with pytest.raises(ValueError, match="threshold"):
            surface_band_mask(np.asarray([0.0]), invalid_value)
        with pytest.raises(ValueError, match="epsilon"):
            surface_band_probability_from_signed_distance(
                np.asarray([0.0]),
                np.asarray([1.0]),
                invalid_value,
            )
        with pytest.raises(ValueError, match="min_std"):
            surface_band_probability_from_signed_distance(
                np.asarray([0.0]),
                np.asarray([1.0]),
                0.1,
                min_std=invalid_value,
            )


def test_surface_band_scale_controls_accept_scalar_arrays() -> None:
    mask = surface_band_mask(np.asarray([0.0, 0.2]), np.array(0.1))
    probability = surface_band_probability_from_signed_distance(
        np.asarray([0.0]),
        np.asarray([1.0]),
        np.array(0.1),
        min_std=np.array(0.01),
    )

    assert mask.tolist() == [True, False]
    assert probability.shape == (1,)
    assert 0.0 <= probability[0] <= 1.0
