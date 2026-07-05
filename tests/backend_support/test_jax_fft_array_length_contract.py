import numpy as np
import pyrecest.backend as backend
import pytest


def _as_numpy(value):
    return np.asarray(backend.to_numpy(value))


def test_jax_real_fft_accepts_singleton_array_lengths():
    if backend.__backend_name__ != "jax":
        pytest.skip("JAX-specific FFT backend contract")

    samples = np.arange(6.0).reshape(2, 3)
    backend_samples = backend.asarray(samples)
    expected_spectrum = np.fft.rfft(samples, n=samples.shape[1], axis=1)
    lengths = (
        np.array(samples.shape[1]),
        np.array([samples.shape[1]]),
        np.int64(samples.shape[1]),
        backend.asarray(samples.shape[1]),
        backend.asarray([samples.shape[1]]),
    )

    for length in lengths:
        actual_spectrum = _as_numpy(
            backend.fft.rfft(backend_samples, n=length, axis=1)
        )
        assert np.allclose(actual_spectrum, expected_spectrum)

        actual_roundtrip = _as_numpy(
            backend.fft.irfft(actual_spectrum, n=length, axis=1)
        )
        assert np.allclose(actual_roundtrip, samples)
