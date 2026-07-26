import numpy as np
import pytest
from pyrecest.models.weak_measurement import (
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
)


def test_block_covariance_rejects_unhashable_dimension_order_entries_cleanly():
    with pytest.raises(ValueError, match="dimension_order entries must be hashable"):
        block_diag_measurement_covariance(
            trusted_std={"x": 1.0},
            dimension_order=[["x"]],
        )


def test_weak_dimension_model_rejects_unhashable_dimension_order_entries_cleanly():
    with pytest.raises(ValueError, match="dimension_order entries must be hashable"):
        WeakDimensionMeasurementModel(
            np.eye(1),
            stds={"x": 1.0},
            dimension_order=[["x"]],
        )
