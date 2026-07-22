import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


_SPLIT_SCALAR_SECTIONS_CODE = r"""
import numpy as np
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


split_functions = [raw_pytorch.split]
if backend.__backend_name__ == "pytorch":
    split_functions.append(backend.split)

for split_func in split_functions:
    for sections in (
        2.5,
        np.array(2.5),
        np.float64(2.5),
        torch.tensor(2.5),
    ):
        try:
            split_func([0, 1, 2, 3], sections)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"{split_func.__module__} silently truncated fractional "
                f"sections {sections!r}"
            )

    for sections in (
        2,
        2.0,
        np.array(2.0),
        np.float64(2.0),
        torch.tensor(2),
        torch.tensor(2.0),
    ):
        parts = split_func([0, 1, 2, 3], sections)
        assert [raw_pytorch.to_numpy(part).tolist() for part in parts] == [
            [0, 1],
            [2, 3],
        ]

print("ok")
"""


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_split_rejects_fractional_scalar_section_counts(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(backend_name, _SPLIT_SCALAR_SECTIONS_CODE)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
