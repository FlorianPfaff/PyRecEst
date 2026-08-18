"""Regression tests for PyTorch assignment index dtype handling."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _assignment_index_contract_code(target_name):
    return f"""
import numpy as np
import torch
import pyrecest  # noqa: F401 - triggers backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

target = {target_name}

for helper_name in ("assignment", "assignment_by_sum"):
    helper = getattr(target, helper_name)

    for invalid_indices in (
        [1.5],
        np.asarray([1.5]),
        np.asarray([1.0 + 0.0j]),
        np.asarray(["1"]),
        torch.tensor([1.5]),
        torch.tensor([1.0 + 0.0j]),
    ):
        try:
            helper(target.zeros(3), 9.0, invalid_indices)
        except IndexError:
            pass
        else:
            raise AssertionError(
                f"{{helper_name}} accepted non-integer indices "
                f"{{invalid_indices!r}}"
            )

    mixed = helper(target.zeros(3), [3.0, 4.0], [False, 1])
    assert target.to_numpy(mixed).tolist() == [3.0, 4.0, 0.0]

    boolean_mask = helper(
        target.zeros(3),
        [5.0, 6.0],
        [True, False, True],
    )
    assert target.to_numpy(boolean_mask).tolist() == [5.0, 0.0, 6.0]

    numpy_boolean_mask = helper(
        target.zeros(3),
        [7.0, 8.0],
        np.asarray([True, False, True]),
    )
    assert target.to_numpy(numpy_boolean_mask).tolist() == [7.0, 0.0, 8.0]

    uint8_index = helper(
        target.zeros(3),
        2.5,
        torch.tensor([1], dtype=torch.uint8),
    )
    assert target.to_numpy(uint8_index).tolist() == [0.0, 2.5, 0.0]

print("ok")
"""


@pytest.mark.parametrize("target_name", ["backend", "raw_pytorch"])
def test_pytorch_assignment_validates_index_dtypes(target_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        _assignment_index_contract_code(target_name),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_raw_pytorch_assignment_contract_is_installed_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        _assignment_index_contract_code("raw_pytorch"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
