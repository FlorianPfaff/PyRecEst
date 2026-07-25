import numpy as np
import pyrecest.backend as backend
import pytest
from scipy.signal import fftconvolve as scipy_fftconvolve


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            np.array([1, 2], dtype=np.uint64),
            np.array([3, 4], dtype=np.int64),
        ),
        (
            np.array([1, 2], dtype=np.uint64),
            np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64),
        ),
    ],
    ids=["unsigned-and-signed", "unsigned-and-complex"],
)
def test_pytorch_fftconvolve_handles_unsigned_mixed_dtypes(first, second):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    actual = backend.to_numpy(backend.signal.fftconvolve(first, second))
    expected = scipy_fftconvolve(first, second)

    assert np.allclose(actual, expected)
