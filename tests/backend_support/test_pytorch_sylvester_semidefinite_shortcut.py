"""Regression tests for PyTorch Sylvester solver shortcuts."""

import pytest

from tests.support.backend_runner import run_backend_code


def test_pytorch_semidefinite_sylvester_shortcut_respects_nonzero_denominators():
    pytest.importorskip("torch")
    code = """
import torch
from pyrecest.backend import linalg

# Symmetric positive-semidefinite factor with a one-dimensional nullspace.
eigvecs = torch.tensor(
    [
        [-0.23813772, -0.95532958, 0.17434201],
        [0.89798926, -0.14791816, 0.41440983],
        [-0.37000772, 0.25586247, 0.89310060],
    ],
    dtype=torch.float64,
)
eigvals = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
a = eigvecs @ torch.diag(eigvals) @ eigvecs.T

# The shortcut accepts almost skew-symmetric q.  The tiny diagonal entries are
# within that tolerance, but they still have nonzero Sylvester denominators in
# the non-null eigenspaces and must not be divided by denominator + I.
tilde_q = torch.tensor(
    [
        [0.0, 0.3, -0.2],
        [-0.3, 1.0e-7, 0.4],
        [0.2, -0.4, -1.0e-7],
    ],
    dtype=torch.float64,
)
q = eigvecs @ tilde_q @ eigvecs.T

solution = linalg.solve_sylvester(a, a, q)
residual = torch.linalg.norm(a @ solution + solution @ a - q)
assert torch.isfinite(solution).all()
assert residual.item() < 1e-10, residual.item()
"""
    result = run_backend_code("pytorch", code)
    assert result.returncode == 0, result.stderr
