import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("torch")

import pyrecest._backend.pytorch.fft as pytorch_fft  # noqa: E402


@pytest.mark.backend_portable
def test_raw_pytorch_real_fft_accepts_numpy_scalar_array_axis_alias():
    vector = np.arange(4.0)
    scalar_axes = [np.array(0), np.int64(0)]

    for axis in scalar_axes:
        spectrum = pytorch_fft.rfft(vector.tolist(), axis=axis)
        npt.assert_allclose(spectrum.numpy(), np.fft.rfft(vector, axis=axis))

        reconstructed = pytorch_fft.irfft(spectrum, n=vector.size, axis=axis)
        expected = np.fft.irfft(
            np.fft.rfft(vector, axis=axis), n=vector.size, axis=axis
        )
        npt.assert_allclose(reconstructed.numpy(), expected)


@pytest.mark.backend_portable
def test_raw_pytorch_real_fft_accepts_numpy_scalar_array_dim():
    vector = np.arange(4.0)
    axis = np.array(0)

    spectrum = pytorch_fft.rfft(vector.tolist(), dim=axis)
    npt.assert_allclose(spectrum.numpy(), np.fft.rfft(vector, axis=int(axis)))
