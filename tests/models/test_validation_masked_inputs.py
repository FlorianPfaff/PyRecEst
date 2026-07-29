from types import SimpleNamespace

import numpy as np
import pytest
from pyrecest import backend
from pyrecest.models.additive_noise import (
    AdditiveNoiseMeasurementModel,
    AdditiveNoiseTransitionModel,
)
from pyrecest.models.validation import (
    infer_state_dim_from_distribution,
    validate_covariance_matrix,
    validate_measurement_matrix,
    validate_noise_covariance,
    validate_state_vector,
)


@pytest.mark.parametrize(
    "call",
    [
        lambda: validate_state_vector(
            np.ma.array([1.0, 999.0], mask=[False, True]), state_dim=2
        ),
        lambda: validate_covariance_matrix(
            np.ma.array([[1.0, 9.0], [9.0, 1.0]], mask=[[False, True], [True, False]])
        ),
        lambda: validate_noise_covariance([[np.ma.array(1.0, mask=True)]]),
        lambda: validate_measurement_matrix(
            np.array([[np.ma.masked]], dtype=object), state_dim=1, meas_dim=1
        ),
    ],
)
def test_model_validators_reject_masked_numeric_inputs(call):
    with pytest.raises(ValueError, match="masked"):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: validate_state_vector([1.0, 2.0], state_dim=np.ma.array(2, mask=True)),
        lambda: validate_state_vector(1.0, allow_scalar=np.ma.array(True, mask=True)),
        lambda: validate_covariance_matrix(
            [[1.0]], check_symmetric=np.ma.array(False, mask=True)
        ),
        lambda: infer_state_dim_from_distribution(
            SimpleNamespace(mean=lambda: np.array([1.0, 2.0])),
            allow_methods=np.ma.array(True, mask=True),
        ),
    ],
)
def test_model_validators_reject_masked_metadata(call):
    with pytest.raises(TypeError):
        call()


@pytest.mark.parametrize("keyword", ["symmetric_rtol", "symmetric_atol"])
def test_covariance_validator_rejects_masked_tolerances(keyword):
    with pytest.raises(ValueError, match="finite nonnegative scalar"):
        validate_covariance_matrix(
            [[1.0]],
            check_symmetric=True,
            **{keyword: np.ma.array(1e-7, mask=True)},
        )


def test_dimension_inference_does_not_use_masked_hidden_payload():
    distribution = SimpleNamespace(
        dim=np.ma.array(7, mask=True),
        mu=np.array([1.0, 2.0]),
    )

    assert infer_state_dim_from_distribution(distribution) == 2


def test_fully_unmasked_masked_arrays_remain_supported():
    state = validate_state_vector(
        np.ma.array([1.0, 2.0], mask=False),
        state_dim=np.ma.array(2, mask=False),
    )
    covariance = validate_covariance_matrix(
        np.ma.array([[1.0]], mask=False),
        allow_scalar=np.ma.array(False, mask=False),
        check_symmetric=np.ma.array(True, mask=False),
        symmetric_rtol=np.ma.array(1e-7, mask=False),
    )

    transition_model = AdditiveNoiseTransitionModel(
        lambda value: value, vectorized=np.ma.array(True, mask=False)
    )

    assert tuple(int(dim) for dim in backend.shape(state)) == (2,)
    assert tuple(int(dim) for dim in backend.shape(covariance)) == (1, 1)
    assert transition_model.vectorized is True


@pytest.mark.parametrize("payload", [True, False])
def test_additive_noise_models_reject_masked_vectorized_flags(payload):
    flag = np.ma.array(payload, mask=True)

    with pytest.raises(TypeError, match="vectorized"):
        AdditiveNoiseTransitionModel(lambda state: state, vectorized=flag)
    with pytest.raises(TypeError, match="vectorized"):
        AdditiveNoiseMeasurementModel(lambda state: state, vectorized=flag)

    model = AdditiveNoiseTransitionModel(lambda state: state)
    with pytest.raises(TypeError, match="vectorized"):
        model.vectorized = flag
    with pytest.raises(TypeError, match="function_is_vectorized"):
        model.function_is_vectorized = flag
