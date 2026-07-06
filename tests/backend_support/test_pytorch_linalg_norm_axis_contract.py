import numpy as np
import pyrecest.backend as backend
import pytest


def _skip_unless_pytorch():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific linalg backend contract")


@pytest.mark.parametrize("axis", [True, np.bool_(False)])
def test_pytorch_linalg_norm_rejects_boolean_axis(axis):
    _skip_unless_pytorch()

    values = backend.ones((2, 2))

    with pytest.raises(TypeError, match="axis must be None"):
        backend.linalg.norm(values, axis=axis)


def test_pytorch_linalg_norm_accepts_numpy_integer_scalar_axis():
    _skip_unless_pytorch()

    values = backend.ones((2, 2))
    result = backend.linalg.norm(values, axis=np.array(1))

    assert backend.to_numpy(result).tolist() == pytest.approx([2**0.5, 2**0.5])
