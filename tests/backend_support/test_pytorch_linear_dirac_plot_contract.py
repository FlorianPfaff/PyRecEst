"""Regression coverage for plotting autograd-backed PyTorch Dirac particles."""

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_linear_dirac_plot_detaches_tensors_for_matplotlib():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)

locations = torch.tensor([0.0, 1.0], requires_grad=True)
weights = torch.tensor([0.25, 0.75], requires_grad=True)
distribution = LinearDiracDistribution(locations, weights)

plt.show = lambda: None
distribution.plot()
plt.close("all")
""",
    )

    assert result.returncode == 0, result.stderr
