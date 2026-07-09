import pyrecest.backend as backend
import pytest
from pyrecest.backend import array


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_prod_accepts_dim_and_keepdim_aliases():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific prod alias regression test")

    values = array([[2.0, 3.0], [4.0, 5.0]])

    alias_result = backend.prod(values, dim=1, keepdim=True)
    axis_result = backend.prod(values, axis=1, keepdims=True)

    assert alias_result.shape == (2, 1)
    assert _to_python(alias_result) == [[6.0], [20.0]]
    assert _to_python(axis_result) == [[6.0], [20.0]]


def test_pytorch_prod_rejects_conflicting_axis_aliases():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific prod alias regression test")

    values = array([[2.0, 3.0], [4.0, 5.0]])

    with pytest.raises(TypeError, match="both 'axis' and 'dim'"):
        backend.prod(values, axis=0, dim=1)
    with pytest.raises(TypeError, match="both 'keepdims' and 'keepdim'"):
        backend.prod(values, axis=1, keepdims=True, keepdim=False)
