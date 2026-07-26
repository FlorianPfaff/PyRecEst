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
def test_pytorch_dirac_distribution_clones_input_tensor_storage():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import torch
from pyrecest.distributions import LinearDiracDistribution

samples = torch.tensor([[1.0], [2.0]])
weights = torch.tensor([0.25, 0.75])
dist = LinearDiracDistribution(samples, weights)

samples[0, 0] = 99.0
weights[0] = 99.0

assert dist.d[0, 0].item() == 1.0
assert dist.w[0].item() == 0.25
assert dist.d.data_ptr() != samples.data_ptr()
assert dist.w.data_ptr() != weights.data_ptr()
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )
