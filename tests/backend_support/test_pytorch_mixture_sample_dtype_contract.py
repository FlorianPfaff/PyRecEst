"""Regression test for PyTorch mixture-sample dtype preservation."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_subprocess_env(backend_name):
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
def test_pytorch_mixture_sampling_preserves_component_dtype():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("pytorch")
    code = """
import pyrecest.backend as backend
from pyrecest.distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture

assert backend.__backend_name__ == "pytorch"
precise_value = 1.0 + 2.0**-40
component = LinearDiracDistribution(
    backend.asarray([precise_value], dtype=backend.float64),
    backend.asarray([1.0], dtype=backend.float64),
)
mixture = LinearMixture(
    [component],
    backend.asarray([1.0], dtype=backend.float64),
)

backend.random.seed(0)
samples = mixture.sample(1)

assert samples.dtype == backend.float64
assert float(samples[0, 0]) == precise_value
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
