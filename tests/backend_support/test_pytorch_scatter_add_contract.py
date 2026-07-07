import pyrecest.backend as backend
import pytest


def _to_python(value):
    converted = backend.to_numpy(value)
    return converted.tolist() if hasattr(converted, "tolist") else converted


@pytest.mark.backend_portable
def test_pytorch_scatter_add_accepts_array_like_inputs():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    result = backend.scatter_add([10, 20, 30], 0, [0, 2], [1, 2])

    assert _to_python(result) == [11, 20, 32]


@pytest.mark.backend_portable
def test_pytorch_scatter_add_rejects_boolean_dim():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    with pytest.raises(TypeError):
        backend.scatter_add([1, 2], True, [0], [1])
