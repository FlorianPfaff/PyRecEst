import pytest

pytest.importorskip("torch")

from pyrecest._backend import pytorch as pytorch_backend


@pytest.mark.parametrize(
    ("function_name", "args"),
    (("sqrtm", ()), ("fractional_matrix_power", (0.5,))),
)
@pytest.mark.parametrize(
    "dtype",
    (
        pytorch_backend.float32,
        pytorch_backend.float64,
        pytorch_backend.complex64,
    ),
)
def test_square_matrix_functions_preserve_empty_batches(function_name, args, dtype):
    values = pytorch_backend.empty((2, 0, 3, 3), dtype=dtype)

    result = getattr(pytorch_backend.linalg, function_name)(values, *args)

    assert tuple(result.shape) == (2, 0, 3, 3)
    assert result.dtype == dtype
    assert result.device == values.device


@pytest.mark.parametrize(
    ("function_name", "args"),
    (("sqrtm", ()), ("fractional_matrix_power", (0.5,))),
)
def test_square_matrix_functions_reject_empty_rectangular_batches(function_name, args):
    values = pytorch_backend.empty((0, 3, 2), dtype=pytorch_backend.float32)

    with pytest.raises(ValueError, match="square matrices"):
        getattr(pytorch_backend.linalg, function_name)(values, *args)


@pytest.mark.parametrize(
    ("side", "factor_shape"),
    (("right", (2, 0, 2, 2)), ("left", (2, 0, 3, 3))),
)
def test_polar_preserves_empty_rectangular_batches(side, factor_shape):
    values = pytorch_backend.empty((2, 0, 3, 2), dtype=pytorch_backend.complex64)

    unitary, factor = pytorch_backend.linalg.polar(values, side=side)

    assert tuple(unitary.shape) == (2, 0, 3, 2)
    assert tuple(factor.shape) == factor_shape
    assert unitary.dtype == pytorch_backend.complex64
    assert factor.dtype == pytorch_backend.complex64
    assert unitary.device == values.device
    assert factor.device == values.device


def test_polar_validates_side_before_empty_batch_fast_path():
    values = pytorch_backend.empty((0, 3, 2), dtype=pytorch_backend.float32)

    with pytest.raises(ValueError, match="either 'right' or 'left'"):
        pytorch_backend.linalg.polar(values, side="invalid")
