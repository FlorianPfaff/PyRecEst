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
def test_circular_grid_get_closest_point_uses_backend_integer_indices():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import math
import numpy as np

import pyrecest.backend as backend
from pyrecest.distributions.circle.circular_grid_distribution import CircularGridDistribution

assert getattr(backend, "__backend_name__", None) == "pytorch"

dist = CircularGridDistribution(backend.ones(4))
points, indices = dist.get_closest_point(backend.asarray([0.1, 2.9, 6.1]))

indices_np = backend.to_numpy(indices)
assert indices_np.dtype.kind in "iu"
assert indices_np.tolist() == [0, 2, 0]
np.testing.assert_allclose(backend.to_numpy(points), [0.0, math.pi, 0.0])

scalar_point, scalar_index = dist.get_closest_point(backend.asarray(math.pi / 2 + 0.1))
assert int(backend.to_numpy(scalar_index).item()) == 1
np.testing.assert_allclose(float(backend.to_numpy(scalar_point)), math.pi / 2)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env("pytorch"),
    )
