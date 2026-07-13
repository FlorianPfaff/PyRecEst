import numpy as np
import pyrecest.backend as backend
import pytest
from scipy.signal import fftconvolve as scipy_fftconvolve


def _skip_unless_pytorch():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")


def test_pytorch_fftconvolve_accepts_scalar_array_axis():
    _skip_unless_pytorch()

    first = backend.asarray([[1.0, 2.0], [3.0, 4.0]])
    second = backend.asarray([[0.5, 1.5]])

    actual = backend.to_numpy(
        backend.signal.fftconvolve(first, second, mode="same", axes=np.array(0))
    )
    expected = scipy_fftconvolve(
        backend.to_numpy(first), backend.to_numpy(second), mode="same", axes=0
    )

    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "axes",
    ["0", np.array(0.0)],
)
def test_pytorch_fftconvolve_rejects_non_integer_axes(axes):
    _skip_unless_pytorch()

    first = backend.asarray([1.0, 2.0])
    second = backend.asarray([3.0, 4.0])

    with pytest.raises(TypeError, match="axes must be None"):
        backend.signal.fftconvolve(first, second, axes=axes)


@pytest.mark.parametrize(
    "axes",
    [np.bool_(True), np.bool_(False), np.array(True), np.array(False)],
)
def test_pytorch_fftconvolve_rejects_numpy_boolean_axes(axes):
    _skip_unless_pytorch()

    first = backend.asarray([1.0, 2.0])
    second = backend.asarray([3.0, 4.0])

    with pytest.raises(ValueError, match="axes must be None"):
        backend.signal.fftconvolve(first, second, axes=axes)


def test_pytorch_fftconvolve_rejects_incompatible_non_convolved_axes():
    _skip_unless_pytorch()

    first = backend.ones((3, 1))
    second = backend.ones((2, 4))

    with pytest.raises(ValueError, match="incompatible shapes"):
        backend.signal.fftconvolve(first, second, axes=(1,))
