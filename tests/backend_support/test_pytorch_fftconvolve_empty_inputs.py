import numpy as np
import pyrecest.backend as backend
import pytest
from scipy.signal import fftconvolve as scipy_fftconvolve


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
@pytest.mark.parametrize(
    ("first_shape", "second_shape"),
    [
        ((0, 2), (3, 2)),
        ((3, 2), (0, 2)),
    ],
    ids=["left-empty", "right-empty"],
)
def test_pytorch_fftconvolve_empty_input_matches_scipy(
    mode,
    first_shape,
    second_shape,
):
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific signal backend contract")

    first = np.empty(first_shape)
    second = np.ones(second_shape)

    actual = backend.to_numpy(backend.signal.fftconvolve(first, second, mode=mode))
    expected = scipy_fftconvolve(first, second, mode=mode)

    assert actual.shape == expected.shape == (0,)
    assert actual.size == 0
