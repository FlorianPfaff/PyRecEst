"""Regression coverage for PyTorch Gaussian-mixture statistics."""

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_gaussian_mixture_statistics_stack_component_means():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        r"""
import numpy as np
import numpy.testing as npt

from pyrecest.backend import array, diag, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture

component1 = GaussianDistribution(
    array([0.0, 1.0]),
    diag(array([1.0, 2.0])),
)
component2 = GaussianDistribution(
    array([2.0, 3.0]),
    diag(array([3.0, 4.0])),
)
mixture = GaussianMixture([component1, component2], array([0.25, 0.75]))

expected_mean = np.array([1.5, 2.5])
expected_covariance = np.array([[3.25, 0.75], [0.75, 4.25]])

npt.assert_allclose(to_numpy(mixture.mean()), expected_mean)
npt.assert_allclose(to_numpy(mixture.covariance()), expected_covariance)
matched = mixture.to_gaussian()
npt.assert_allclose(to_numpy(matched.mu), expected_mean)
npt.assert_allclose(to_numpy(matched.C), expected_covariance)
""",
    )

    assert result.returncode == 0, result.stderr
