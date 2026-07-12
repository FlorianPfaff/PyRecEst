import numpy as np
import pytest
from pyrecest.models import (
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trusted_std": [1.0, 2.0], "dimension_order": ["y", "x"]},
        {"weak_std": (3.0,), "dimension_order": ["z"]},
    ],
)
def test_block_diag_rejects_dimension_order_for_positional_stds(kwargs) -> None:
    with pytest.raises(ValueError, match="dimension_order requires mapping"):
        block_diag_measurement_covariance(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stds": [1.0, 2.0], "dimension_order": ["y", "x"]},
        {"measurement_noise_cov": np.eye(2), "dimension_order": ["y", "x"]},
    ],
)
def test_weak_dimension_model_rejects_ignored_dimension_order(kwargs) -> None:
    with pytest.raises(ValueError, match="dimension_order requires mapping"):
        WeakDimensionMeasurementModel(np.eye(2), **kwargs)


def test_mapping_stds_still_honor_dimension_order() -> None:
    covariance = block_diag_measurement_covariance(
        trusted_std={"x": 1.0},
        weak_std={"y": 2.0},
        dimension_order=["y", "x"],
    )

    assert np.allclose(covariance, np.diag([4.0, 1.0]))
