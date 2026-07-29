from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.models.validation import (
    infer_state_dim_from_distribution,
    validate_covariance_matrix,
    validate_state_vector,
    validate_transition_matrix,
)


@pytest.mark.parametrize(
    ("validator", "value", "message"),
    (
        (
            lambda value: validate_state_vector(value, state_dim=2),
            np.ma.array([1.0, 9.0], mask=[False, True]),
            "state must not contain masked values",
        ),
        (
            lambda value: validate_transition_matrix(
                value, state_dim=2, pred_dim=2
            ),
            np.ma.array(
                [[1.0, 0.0], [0.0, 1.0]],
                mask=[[False, False], [True, False]],
            ),
            "system_matrix must not contain masked values",
        ),
        (
            lambda value: validate_state_vector(value, state_dim=2),
            [1.0, True],
            "state must contain numeric non-boolean values",
        ),
        (
            lambda value: validate_transition_matrix(
                value, state_dim=2, pred_dim=2
            ),
            [[1.0, 0.0], [False, 1.0]],
            "system_matrix must contain numeric non-boolean values",
        ),
    ),
)
def test_model_array_validation_rejects_values_hidden_by_coercion(
    validator, value, message
) -> None:
    with pytest.raises(ValueError, match=message):
        validator(value)


def test_model_scalar_metadata_rejects_masked_values() -> None:
    with pytest.raises(TypeError, match="allow_scalar must be a boolean"):
        validate_state_vector(
            1.0,
            state_dim=1,
            allow_scalar=np.ma.array(True, mask=True),
        )

    with pytest.raises(TypeError, match="dim must be an integer or None"):
        validate_state_vector(
            [1.0, 2.0],
            state_dim=np.ma.array(2, mask=True),
        )

    with pytest.raises(ValueError, match="symmetric_rtol must be a finite"):
        validate_covariance_matrix(
            [[1.0]],
            check_symmetric=True,
            symmetric_rtol=np.ma.array(1e-7, mask=True),
        )


def test_dimension_inference_rejects_masked_explicit_dimension() -> None:
    distribution = SimpleNamespace(dim=np.ma.array(4, mask=True))

    with pytest.raises(ValueError, match="Could not infer"):
        infer_state_dim_from_distribution(distribution)


def test_model_validation_accepts_fully_unmasked_masked_arrays() -> None:
    state = validate_state_vector(
        np.ma.array([1.0, 2.0], mask=False),
        state_dim=np.ma.array(2, mask=False),
    )
    covariance = validate_covariance_matrix(
        np.ma.array([[1.0]], mask=False),
        check_symmetric=np.ma.array(True, mask=False),
        symmetric_rtol=np.ma.array(1e-7, mask=False),
    )

    npt.assert_allclose(np.asarray(state), [1.0, 2.0])
    npt.assert_allclose(np.asarray(covariance), [[1.0]])
