"""Regression tests for PyTorch registration host conversion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_registration_converts_scipy_inputs_through_backend_bridge():
    """Gradient-tracking tensors must cross SciPy's host boundary safely."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import torch
from pyrecest.utils._point_set_registration_common import solve_gated_assignment
from pyrecest.utils.point_set_registration import joint_registration_assignment

reference = torch.tensor(
    [[0.0], [2.0]], dtype=torch.float64, requires_grad=True
)
moving = torch.tensor(
    [[1.0], [3.0]], dtype=torch.float64, requires_grad=True
)

registration = joint_registration_assignment(
    reference,
    moving,
    model="translation",
    max_iterations=2,
)
assert registration.assignment.detach().cpu().tolist() == [0, 1]
assert torch.allclose(
    registration.transform.offset,
    torch.tensor([1.0], dtype=torch.float64),
)

costs = torch.tensor(
    [[0.0, 4.0], [4.0, 0.0]],
    dtype=torch.float64,
    requires_grad=True,
)
assignment = solve_gated_assignment(costs)
assert assignment.detach().cpu().tolist() == [0, 1]
assert costs.requires_grad
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
