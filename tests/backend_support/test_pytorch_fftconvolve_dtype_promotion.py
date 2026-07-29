import numpy as np
import numpy.testing as npt
import pytest
from scipy.signal import fftconvolve as scipy_fftconvolve

pytest.importorskip("torch")

import pyrecest._backend.pytorch.signal as pytorch_signal  # noqa: E402


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3, 4], dtype=np.int16),
        ),
        (
            np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64),
            np.array([3, 4], dtype=np.int32),
        ),
        (
            np.array([1, 2], dtype=np.int16),
            np.array([3, 4], dtype=np.int64),
        ),
    ],
    ids=["float-and-integer", "complex-and-integer", "integer-pair"],
)
def test_raw_pytorch_fftconvolve_matches_scipy_dtype_promotion(first, second):
    actual = pytorch_signal.fftconvolve(first, second).cpu().numpy()
    expected = scipy_fftconvolve(first, second)

    assert actual.dtype == expected.dtype
    npt.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.backend_portable
def test_raw_pytorch_fftconvolve_preserves_cancelling_integer_residual():
    first = np.array([1.0, 1.0], dtype=np.float32)
    second = np.array([2**24 + 1, -(2**24)], dtype=np.int64)

    actual = pytorch_signal.fftconvolve(first, second).cpu().numpy()
    expected = np.convolve(first.astype(np.float64), second.astype(np.float64))

    assert actual.dtype == np.float64
    npt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-6)
    assert actual[1] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.backend_portable
def test_raw_pytorch_fftconvolve_promotes_half_precision_for_cpu_fft():
    first = np.array([1.0, 2.0], dtype=np.float16)
    second = np.array([3.0, 4.0], dtype=np.float16)

    actual = pytorch_signal.fftconvolve(first, second).cpu().numpy()
    expected = scipy_fftconvolve(first, second)

    assert actual.dtype == expected.dtype == np.float32
    npt.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
