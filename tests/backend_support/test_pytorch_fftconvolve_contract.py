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
