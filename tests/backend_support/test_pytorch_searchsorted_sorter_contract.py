import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _searchsorted_sorter_validation_code(target_module):
    return f"""
import numpy as np
import torch
import pyrecest.evidence  # noqa: F401 ensure runtime backend patches are installed
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

target = {target_module}
boundaries = [3.0, 1.0, 5.0]
values = [2.0]

invalid_sorters = (
    [1.0, 0.0, 2.0],
    [1.5, 0.0, 2.0],
    [True, False, True],
    np.asarray([1, 0, 2], dtype=object),
    np.asarray([1, 0, 2], dtype=np.uint64),
    np.asarray([[1, 0, 2]]),
    torch.tensor([1.0, 0.0, 2.0]),
    torch.tensor([True, False, True]),
)
for sorter in invalid_sorters:
    try:
        target.searchsorted(boundaries, values, sorter=sorter)
    except TypeError:
        pass
    else:
        raise AssertionError(f"accepted invalid sorter {{sorter!r}}")

valid = target.searchsorted(
    boundaries,
    values,
    sorter=np.asarray([1, 0, 2], dtype=np.uint8),
)
assert target.to_numpy(valid).tolist() == [1]
print("ok")
"""


def test_raw_pytorch_searchsorted_rejects_invalid_sorters_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        _searchsorted_sorter_validation_code("raw_pytorch"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_searchsorted_rejects_invalid_sorters():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        _searchsorted_sorter_validation_code("backend"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
