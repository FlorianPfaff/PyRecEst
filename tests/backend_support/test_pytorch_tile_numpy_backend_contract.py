import importlib.util
import os
import subprocess
import sys

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_tile_module_uses_torch_tensor_under_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "numpy"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as public_backend
from pyrecest._backend import pytorch as pytorch_backend

assert public_backend.__backend_name__ == "numpy"
values = pytorch_backend.array([[1, 2], [3, 4]])

result = pytorch_backend.tile(values, 2)
assert tuple(result.shape) == (2, 4)
assert pytorch_backend.to_numpy(result).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]
assert result.device == values.device
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_tile_is_native_and_stable_after_metadata_imports(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import importlib
import inspect
import numpy as np
import numpy.testing as npt
import torch
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

raw_tile = raw_pytorch.tile
public_tile = public_backend.tile
assert raw_tile.__module__ == raw_pytorch.__name__
assert tuple(inspect.signature(raw_tile).parameters) == ("x", "reps")
if public_backend.__backend_name__ == "pytorch":
    assert public_tile is raw_tile

for module_name in ("pyrecest.backend_support", "pyrecest.evidence"):
    module = importlib.import_module(module_name)
    importlib.reload(module)
    assert raw_pytorch.tile is raw_tile
    assert public_backend.tile is public_tile
    values = torch.tensor([[1, 2], [3, 4]])
    result = raw_tile(x=values, reps=torch.tensor([2, 1]))
    npt.assert_array_equal(result.numpy(), np.tile(values.numpy(), (2, 1)))
"""
    result = run_backend_code(backend_name, code, timeout=60)
    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_tile_cuda_device_repetitions_and_gradients():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; CPU tile contracts are tested separately")

    code = """
import numpy as np
import numpy.testing as npt
import torch
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

tile_functions = [raw_pytorch.tile]
if public_backend.__backend_name__ == "pytorch":
    tile_functions.append(public_backend.tile)

for tile in tile_functions:
    for values_device in ("cpu", "cuda"):
        for reps_device in ("cpu", "cuda"):
            for repetitions in ((), (2,), (2, 1), (0, 2)):
                base = torch.arange(6, dtype=torch.float64, device=values_device, requires_grad=True)
                values = base.reshape(2, 3).T
                original = values.detach().clone()
                reps = torch.tensor(repetitions, dtype=torch.int64, device=reps_device)
                result = tile(x=values, reps=reps)
                assert result.device == values.device
                assert result.dtype == values.dtype
                npt.assert_array_equal(
                    result.detach().cpu().numpy(),
                    np.tile(original.cpu().numpy(), repetitions),
                )
                result.sum().backward()
                torch.testing.assert_close(base.grad, torch.full_like(base, np.prod(repetitions)))
                if result.numel():
                    assert result.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
                    with torch.no_grad():
                        result.fill_(0)
                torch.testing.assert_close(values.detach(), original)
                assert reps.device.type == reps_device
                assert reps.cpu().tolist() == list(repetitions)
"""
    for backend_name in ("numpy", "pytorch"):
        result = run_backend_code(backend_name, code, timeout=60)
        assert result.returncode == 0, result.stderr
