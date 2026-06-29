from __future__ import annotations

import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_fft_accepts_array_like_inputs_and_numpy_axis_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

values = [[1.0, 2.0], [3.0, 4.0]]
fftn_result = backend.fft.fftn(values, axes=(0, 1))
np.testing.assert_allclose(
    backend.to_numpy(fftn_result),
    np.fft.fftn(np.asarray(values), axes=(0, 1)),
)

rfft_result = backend.fft.rfft([0.0, 1.0, 0.0, -1.0], axis=0)
round_trip = backend.fft.irfft(rfft_result, n=4, axis=0)
np.testing.assert_allclose(
    backend.to_numpy(round_trip),
    np.asarray([0.0, 1.0, 0.0, -1.0]),
)

np.testing.assert_array_equal(
    backend.to_numpy(backend.fft.fftshift([0, 1, 2, 3], axes=0)),
    np.asarray([2, 3, 0, 1]),
)
np.testing.assert_array_equal(
    backend.to_numpy(backend.fft.ifftshift([2, 3, 0, 1], axes=0)),
    np.asarray([0, 1, 2, 3]),
)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_pytorch_fft_rejects_conflicting_axis_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

try:
    backend.fft.rfft([0.0, 1.0], dim=0, axis=1)
except TypeError as exc:
    assert "dim" in str(exc)
    assert "axis" in str(exc)
else:
    raise AssertionError("conflicting dim/axis aliases were accepted")
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
