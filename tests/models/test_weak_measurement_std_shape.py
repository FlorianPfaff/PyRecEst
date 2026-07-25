import numpy as np
import pytest

from pyrecest.models import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
)


@pytest.mark.parametrize(
    "stds",
    [
        np.array([[1.0, 2.0]]),
        np.array([[1.0], [2.0]]),
    ],
)
def test_measurement_standard_deviations_reject_matrix_shapes(stds) -> None:
    constructors = (
        lambda: diagonal_measurement_covariance(stds),
        lambda: block_diag_measurement_covariance(trusted_std=stds),
        lambda: MaskedLinearMeasurementModel(
            state_dim=2,
            observed_dims=[0, 1],
            stds=stds,
        ),
        lambda: WeakDimensionMeasurementModel(np.eye(2), stds=stds),
    )

    for constructor in constructors:
        with pytest.raises(ValueError, match="one-dimensional"):
            constructor()


def test_scalar_standard_deviation_remains_supported() -> None:
    covariance = diagonal_measurement_covariance(2.0)

    assert covariance.shape == (1, 1)
    assert np.allclose(covariance, [[4.0]])
