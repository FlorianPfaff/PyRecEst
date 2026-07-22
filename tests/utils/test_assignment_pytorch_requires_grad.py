"""Regression for PyTorch assignment costs that track gradients."""

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_assignment_accepts_requires_grad_cost_matrix():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import torch

from pyrecest.utils import (
    min_cost_max_cardinality_assignment,
    murty_k_best_assignments,
)

costs = torch.tensor(
    [[1.0, 3.0], [4.0, 2.0]],
    dtype=torch.float64,
    requires_grad=True,
)

ranked = murty_k_best_assignments(
    costs,
    k=2,
    row_non_assignment_costs=100.0,
    col_non_assignment_costs=100.0,
)
assert ranked[0]["assignment"].detach().cpu().tolist() == [0, 1]
assert ranked[0]["cost"] == 3.0

best = min_cost_max_cardinality_assignment(costs)
assert best["assignment"].detach().cpu().tolist() == [0, 1]
assert best["cost"] == 3.0
""",
    )

    assert result.returncode == 0, result.stderr
