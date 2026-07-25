import importlib.util

import numpy as np
import numpy.testing as npt
import pytest


def _skip_without_torch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")


@pytest.mark.backend_portable
def test_raw_pytorch_isclose_accepts_equal_nan_keyword():
    _skip_without_torch()

    import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel

    left = [np.nan, 1.0, 1.0]
    right = [np.nan, 1.0 + 1e-9, 2.0]

    result = raw_pytorch.isclose(left, right, equal_nan=True)
    expected = np.isclose(
        left,
        right,
        rtol=raw_pytorch.rtol,
        atol=raw_pytorch.atol,
        equal_nan=True,
    )
    npt.assert_array_equal(raw_pytorch.to_numpy(result), expected)


@pytest.mark.backend_portable
def test_public_pytorch_isclose_accepts_equal_nan_keyword_when_selected():
    _skip_without_torch()

    import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel

    if backend.__backend_name__ != "pytorch":
        pytest.skip("public backend is not PyTorch")

    left = [np.nan, 1.0, 1.0]
    right = [np.nan, 1.0 + 1e-9, 2.0]

    result = backend.isclose(left, right, equal_nan=True)
    expected = np.isclose(
        left, right, rtol=backend.rtol, atol=backend.atol, equal_nan=True
    )
    npt.assert_array_equal(backend.to_numpy(result), expected)

    result_without_nan_match = backend.isclose(left, right, equal_nan=False)
    expected_without_nan_match = np.isclose(
        left,
        right,
        rtol=backend.rtol,
        atol=backend.atol,
        equal_nan=False,
    )
    npt.assert_array_equal(
        backend.to_numpy(result_without_nan_match), expected_without_nan_match
    )
