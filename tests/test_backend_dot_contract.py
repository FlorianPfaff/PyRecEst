import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
import pytest
from pyrecest.backend import array


def _as_numpy(value):
    return np.asarray(backend.to_numpy(value))


def _assert_dot_matches_numpy(left, right):
    result = backend.dot(array(left), array(right))
    expected = np.dot(np.asarray(left), np.asarray(right))
    npt.assert_allclose(_as_numpy(result), expected)
    assert result.shape == expected.shape


def test_backend_dot_matches_numpy_contract_for_supported_backends():
    if backend.__backend_name__ == "pytorch":
        pytest.skip("PyTorch has a backend-specific dot helper contract")

    _assert_dot_matches_numpy(
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    )
    _assert_dot_matches_numpy([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    _assert_dot_matches_numpy([1.0, 2.0], [[5.0, 6.0], [7.0, 8.0]])
    _assert_dot_matches_numpy(2.0, [[5.0, 6.0], [7.0, 8.0]])


def test_backend_dot_matches_numpy_contract_for_high_rank_inputs():
    if backend.__backend_name__ == "pytorch":
        pytest.skip("PyTorch has a backend-specific dot helper contract")

    left = np.arange(24, dtype=float).reshape(2, 3, 4)
    right = np.arange(120, dtype=float).reshape(5, 4, 6)

    _assert_dot_matches_numpy(left, right)
