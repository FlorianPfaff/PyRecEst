import importlib.util

import pyrecest.backend as backend
import pytest
from tests.support.backend_runner import run_backend_code


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_set_diag_accepts_array_like_matrix_inputs():
    result = backend.set_diag([[1, 2], [3, 4]], [9, 8])

    assert _to_python(result) == [[9, 2], [3, 8]]


def test_raw_pytorch_set_diag_accepts_array_like_matrix_inputs():
    pytest.importorskip("torch")

    import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel

    result = raw_pytorch.set_diag([[1, 2], [3, 4]], [9, 8])

    assert raw_pytorch.to_numpy(result).tolist() == [[9, 2], [3, 8]]


def _run_contract(backend_name, code):
    result = run_backend_code(backend_name, code, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "jax"])
def test_public_set_diag_preserves_backend_mutation_and_keyword_contract(backend_name):
    if importlib.util.find_spec(backend_name) is None:
        pytest.skip(f"{backend_name} is not installed")

    _run_contract(
        backend_name,
        """
import numpy as np
import numpy.testing as npt
import pytest
import pyrecest.backend as backend

values = backend.array(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
original = backend.to_numpy(values).copy()
new_diag = np.array([8, 9], dtype=np.float32)
expected = original.copy()
expected[..., [0, 1], [0, 1]] = new_diag
result = backend.set_diag(x=values, new_diag=new_diag)
npt.assert_array_equal(backend.to_numpy(result), expected)
assert result.dtype == values.dtype
npt.assert_array_equal(new_diag, [8, 9])
if backend.__backend_name__ == "numpy":
    assert result is values
    npt.assert_array_equal(values, expected)
else:
    assert result is not values
    npt.assert_array_equal(backend.to_numpy(values), original)

list_input = [[1, 2], [3, 4]]
result = backend.set_diag(x=list_input, new_diag=[9, 8])
npt.assert_array_equal(backend.to_numpy(result), [[9, 2], [3, 8]])
assert list_input == [[1, 2], [3, 4]]

with pytest.raises(IndexError):
    backend.set_diag(x=[1, 2], new_diag=[3, 4])
with pytest.raises(ValueError):
    backend.set_diag(x=[[1, 2], [3, 4]], new_diag=[5, 6, 7])
""",
    )


_TORCH_SET_DIAG_CONTRACT = """
import numpy as np
import numpy.testing as npt
import pytest
import torch
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

set_diag_functions = [raw_pytorch.set_diag]
if public_backend.__backend_name__ == "pytorch":
    set_diag_functions.append(public_backend.set_diag)

cases = [
    ([[1, 2], [3, 4]], [9, 8]),
    (np.arange(6, dtype=np.float32).reshape(2, 3), np.array([8, 9], dtype=np.float32)),
    (torch.arange(6, dtype=torch.int64).reshape(2, 3), torch.tensor([8, 9], dtype=torch.int32)),
    (torch.arange(6, dtype=torch.float32).reshape(2, 3).T, [8, 9]),
    (torch.tensor([[1 + 2j, 3 - 1j], [4 + 1j, 5 + 2j]], dtype=torch.complex128), [8 - 2j, 9 + 3j]),
    (torch.arange(12, dtype=torch.float64).reshape(2, 2, 3), np.array([8, 9], dtype=np.float64)),
    (torch.arange(12, dtype=torch.float64).reshape(2, 2, 3), torch.tensor([[8, 9], [6, 7]])),
    (torch.ones((2, 2, 3), dtype=torch.float64), 5),
]
for set_diag in set_diag_functions:
    for values, new_diag in cases:
        if torch.is_tensor(values):
            values = values.to(device)
        elif device != "cpu":
            continue
        original = as_numpy(values).copy()
        original_diag = as_numpy(new_diag).copy()
        expected = original.copy()
        indices = np.arange(min(expected.shape[-2:]))
        expected[..., indices, indices] = original_diag
        result = set_diag(x=values, new_diag=new_diag)
        npt.assert_array_equal(as_numpy(result), expected)
        assert result is not values
        if torch.is_tensor(values):
            assert result.dtype == values.dtype
            assert result.device == values.device
            assert result.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
        elif isinstance(values, np.ndarray):
            assert as_numpy(result).dtype == values.dtype
        result.fill_(0)
        npt.assert_array_equal(as_numpy(values), original)
        npt.assert_array_equal(as_numpy(new_diag), original_diag)

    values = torch.arange(12, dtype=torch.float64, device=device).reshape(2, 3, 2).transpose(-1, -2)
    values = values.detach().requires_grad_()
    new_diag = torch.tensor([2.0, 4.0], dtype=torch.float64, device=device, requires_grad=True)
    original = values.detach().clone()
    assert not values.is_contiguous()
    weights = torch.arange(12, dtype=torch.float64, device=device).reshape(2, 2, 3)
    result = set_diag(x=values, new_diag=new_diag)
    (result * weights).sum().backward()
    expected_input_grad = weights.clone()
    expected_input_grad[..., [0, 1], [0, 1]] = 0
    expected_diag_grad = weights[..., [0, 1], [0, 1]].sum(dim=0)
    torch.testing.assert_close(values.grad, expected_input_grad)
    torch.testing.assert_close(new_diag.grad, expected_diag_grad)
    torch.testing.assert_close(values.detach(), original)
    assert result.device == values.device

    with pytest.raises(IndexError):
        set_diag(x=[1, 2], new_diag=[3, 4])
    with pytest.raises(RuntimeError):
        set_diag(x=[[1, 2], [3, 4]], new_diag=[5, 6, 7])
"""


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_set_diag_values_storage_and_gradients(backend_name):
    pytest.importorskip("torch")
    _run_contract(backend_name, 'device = "cpu"\n' + _TORCH_SET_DIAG_CONTRACT)


@pytest.mark.backend_portable
def test_pytorch_set_diag_cuda_device_and_gradients():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available; CPU set_diag contracts are tested separately"
        )
    for backend_name in ("numpy", "pytorch"):
        _run_contract(backend_name, 'device = "cuda"\n' + _TORCH_SET_DIAG_CONTRACT)


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch", "jax"])
def test_set_diag_ownership_and_identity_after_metadata_imports(backend_name):
    dependency = "torch" if backend_name == "pytorch" else backend_name
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")

    _run_contract(
        backend_name,
        """
import importlib
import importlib.util
import pyrecest.backend as backend

public_set_diag = backend.set_diag
raw_pytorch = None
if importlib.util.find_spec("torch") is not None:
    import pyrecest._backend.pytorch as raw_pytorch
    raw_set_diag = raw_pytorch.set_diag
    assert raw_set_diag.__module__ == raw_pytorch.__name__
    if backend.__backend_name__ == "pytorch":
        assert public_set_diag is raw_set_diag

for module_name in ("pyrecest.backend_support", "pyrecest.evidence"):
    importlib.reload(importlib.import_module(module_name))
    assert backend.set_diag is public_set_diag
    if raw_pytorch is not None:
        assert raw_pytorch.set_diag is raw_set_diag
""",
    )
