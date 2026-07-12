import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_gaussian_distribution_copies_mutable_parameter_storage(backend_name):
    if backend_name == "pytorch" and importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution

mu = array([1.0, 2.0])
covariance = array([[2.0, 0.25], [0.25, 3.0]])
distribution = GaussianDistribution(mu, covariance)

mu[0] = 99.0
covariance[0, 0] = 99.0

npt.assert_allclose(to_numpy(distribution.mu), [1.0, 2.0])
npt.assert_allclose(to_numpy(distribution.C), [[2.0, 0.25], [0.25, 3.0]])

new_mean = array([3.0, 4.0])
moved = distribution.set_mean(new_mean)
new_mean[0] = 99.0

npt.assert_allclose(to_numpy(moved.mu), [3.0, 4.0])
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env(backend_name),
    )
