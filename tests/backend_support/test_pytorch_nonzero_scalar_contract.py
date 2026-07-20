import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_nonzero_rejects_zero_dimensional_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest._backend.pytorch as raw_pytorch
import pyrecest.backend as backend
import torch


def assert_rejects_scalar(helper, value):
    try:
        helper(value)
    except ValueError as exc:
        assert "nonzero on 0d arrays is not allowed" in str(exc)
    else:
        raise AssertionError("nonzero accepted a zero-dimensional input")


for nonzero in (backend.nonzero, raw_pytorch.nonzero):
    for scalar in (0, 1, torch.tensor(0), torch.tensor(1)):
        assert_rejects_scalar(nonzero, scalar)

    rows, columns = nonzero([[0, 2], [3, 0]])
    assert rows.tolist() == [0, 1]
    assert columns.tolist() == [1, 0]

print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
