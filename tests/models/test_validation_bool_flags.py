from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pyrecest import backend
from pyrecest.models.validation import (
    infer_state_dim_from_distribution,
    validate_covariance_matrix,
    validate_state_vector,
)


def test_validate_state_vector_rejects_nonboolean_allow_scalar() -> None:
    invalid_flags = ("False", 1, np.array([True]))

    for flag in invalid_flags:
        with pytest.raises(TypeError, match="allow_scalar"):
            validate_state_vector(1.0, allow_scalar=flag)


@pytest.mark.parametrize("flag", [True, np.bool_(True), np.array(True)])
def test_validate_state_vector_accepts_boolean_allow_scalar(flag: bool) -> None:
    vector = validate_state_vector(1.0, allow_scalar=flag)

    assert int(backend.ndim(vector)) == 1
    assert tuple(int(dim) for dim in backend.shape(vector)) == (1,)


def test_validate_covariance_matrix_rejects_nonboolean_flags() -> None:
    invalid_flags = ("False", 1, np.array([True]))

    for flag in invalid_flags:
        with pytest.raises(TypeError, match="allow_scalar"):
            validate_covariance_matrix(1.0, allow_scalar=flag)
        with pytest.raises(TypeError, match="check_symmetric"):
            validate_covariance_matrix([[1.0]], check_symmetric=flag)


def test_validate_covariance_matrix_accepts_scalar_array_bool_flags() -> None:
    covariance = validate_covariance_matrix(
        1.0, allow_scalar=np.array(True), check_symmetric=np.array(True)
    )

    assert tuple(int(dim) for dim in backend.shape(covariance)) == (1, 1)


def test_infer_state_dim_rejects_nonboolean_allow_methods() -> None:
    distribution = SimpleNamespace(mean=lambda: np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match="allow_methods"):
        infer_state_dim_from_distribution(distribution, allow_methods="False")


def test_infer_state_dim_accepts_scalar_array_allow_methods() -> None:
    distribution = SimpleNamespace(mean=lambda: np.array([1.0, 2.0]))

    assert (
        infer_state_dim_from_distribution(distribution, allow_methods=np.array(True))
        == 2
    )
