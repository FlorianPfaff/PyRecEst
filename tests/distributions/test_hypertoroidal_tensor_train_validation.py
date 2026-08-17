"""Validation regressions for hypertoroidal tensor-train data."""

import numpy as np
import pytest
from pyrecest.distributions.hypertorus._tensor_train import TensorTrain


@pytest.mark.parametrize(
    "cores",
    [
        (np.ones((1, 0, 1)),),
        (np.ones((1, 2, 0)), np.ones((0, 2, 1))),
    ],
)
def test_tensor_train_rejects_zero_sized_modes_and_ranks(cores):
    with pytest.raises(ValueError, match="positive"):
        TensorTrain(cores)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_tensor_train_rejects_nonfinite_core_and_dense_data(invalid_value):
    core = np.ones((1, 3, 1), dtype=np.complex128)
    core[0, 1, 0] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        TensorTrain((core,))

    dense = np.ones((2, 2), dtype=np.complex128)
    dense[0, 1] = invalid_value
    with pytest.raises(ValueError, match="finite"):
        TensorTrain.from_dense(dense)
