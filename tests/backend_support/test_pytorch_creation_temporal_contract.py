import numpy as np
import pytest
from pyrecest.backend_support._pytorch_creation_shape_contract import (
    _pytorch_creation_scalar,
    _pytorch_creation_shape,
)


class _NoTensorTorch:
    @staticmethod
    def is_tensor(value):
        del value
        return False


@pytest.mark.parametrize(
    "shape",
    [
        np.timedelta64(3, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000003"),
        np.asarray(np.timedelta64(3, "ns")),
        np.asarray(np.datetime64("1970-01-01T00:00:00.000000003")),
        np.asarray([np.timedelta64(3, "ns")]),
        np.asarray([np.datetime64("1970-01-01T00:00:00.000000003")]),
    ],
)
def test_pytorch_creation_shape_rejects_native_temporal_dtypes(shape):
    with pytest.raises(TypeError, match="shape dimensions must be integers"):
        _pytorch_creation_shape(shape, np, _NoTensorTorch)


@pytest.mark.parametrize(
    "value",
    [
        np.timedelta64(3, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000003"),
        np.asarray(np.timedelta64(3, "ns")),
        np.asarray(np.datetime64("1970-01-01T00:00:00.000000003")),
    ],
)
def test_pytorch_creation_scalar_rejects_native_temporal_dtypes(value):
    with pytest.raises(TypeError, match="arange start must be numeric"):
        _pytorch_creation_scalar(
            value, np, _NoTensorTorch, argument_name="arange start"
        )


def test_pytorch_creation_shape_still_accepts_numpy_integer_scalars():
    assert _pytorch_creation_shape(np.asarray(3), np, _NoTensorTorch) == (3,)


def test_pytorch_creation_scalar_still_accepts_numpy_numeric_scalars():
    assert (
        _pytorch_creation_scalar(
            np.asarray(3.5),
            np,
            _NoTensorTorch,
            argument_name="arange start",
        )
        == 3.5
    )
