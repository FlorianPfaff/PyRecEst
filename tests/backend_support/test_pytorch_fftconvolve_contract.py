import numpy as np
import pyrecest.backend as backend
import pytest
from scipy.signal import fftconvolve as scipy_fftconvolve


def _as_numpy(value):
    return backend.to_numpy(value)


def test_pytorch_fftconvolve_matches_scipy_full_same_valid():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray([[1.0, 2.0, 0.0], [0.5, -1.0, 3.0]])
    second = backend.asarray([[2.0, -1.0], [0.0, 0.25]])

    for mode in ("full", "same", "valid"):
        actual = _as_numpy(backend.signal.fftconvolve(first, second, mode=mode))
        expected = scipy_fftconvolve(_as_numpy(first), _as_numpy(second), mode=mode)
        assert np.allclose(actual, expected)


def test_pytorch_fftconvolve_matches_scipy_selected_axes():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray(np.arange(12.0).reshape(2, 3, 2))
    second = backend.ones((2, 2, 2))

    actual = _as_numpy(
        backend.signal.fftconvolve(first, second, mode="same", axes=(1, 2))
    )
    expected = scipy_fftconvolve(
        _as_numpy(first), _as_numpy(second), mode="same", axes=(1, 2)
    )

    assert np.allclose(actual, expected)


def test_pytorch_fftconvolve_same_crops_broadcast_non_convolved_axes():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray([[2.0]])
    second = backend.asarray([[1.0, 3.0, 5.0]])

    actual = _as_numpy(
        backend.signal.fftconvolve(first, second, mode="same", axes=(0,))
    )
    expected = scipy_fftconvolve(
        _as_numpy(first), _as_numpy(second), mode="same", axes=(0,)
    )

    assert actual.shape == expected.shape == _as_numpy(first).shape
    assert np.allclose(actual, expected)


def test_pytorch_fftconvolve_valid_allows_mixed_singleton_axes():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray([[1.0, 2.0]])
    second = backend.asarray([[0.5], [1.5]])

    actual = _as_numpy(backend.signal.fftconvolve(first, second, mode="valid"))
    expected = scipy_fftconvolve(_as_numpy(first), _as_numpy(second), mode="valid")

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)


@pytest.mark.parametrize("axes", [(), []])
def test_pytorch_fftconvolve_rejects_empty_axes_for_arrays(axes):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray([1.0, 2.0])
    second = backend.asarray([3.0, 4.0])

    with pytest.raises(ValueError, match="axes cannot be empty"):
        backend.signal.fftconvolve(first, second, axes=axes)


def test_pytorch_fftconvolve_scalar_empty_axes_matches_scipy():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = backend.asarray(2.0)
    second = backend.asarray(3.0)

    actual = _as_numpy(backend.signal.fftconvolve(first, second, axes=()))
    expected = scipy_fftconvolve(_as_numpy(first), _as_numpy(second), axes=())

    assert np.allclose(actual, expected)
