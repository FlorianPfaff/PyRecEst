import numpy as np
import pyrecest.backend as backend
import pytest


def _as_numpy(value):
    return backend.to_numpy(value)


def test_pytorch_concatenate_axis_none_flattens_inputs():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific concatenate contract")

    first = backend.asarray([[1, 2], [3, 4]])
    second = backend.asarray([[5], [6]])

    actual = _as_numpy(backend.concatenate((first, second), axis=None))
    expected = np.concatenate((_as_numpy(first), _as_numpy(second)), axis=None)

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)


def test_raw_pytorch_concatenate_axis_none_is_patched_under_numpy_backend():
    import pyrecest._backend.pytorch as raw_pytorch

    torch = pytest.importorskip("torch")
    first = torch.tensor([[1, 2], [3, 4]])
    second = torch.tensor([[5], [6]])

    actual = raw_pytorch.to_numpy(raw_pytorch.concatenate((first, second), axis=None))
    expected = np.concatenate((first.numpy(), second.numpy()), axis=None)

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)
