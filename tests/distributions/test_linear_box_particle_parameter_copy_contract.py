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
def test_constructor_copies_mutable_supports_and_weights(backend_name):
    if backend_name == "pytorch" and importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = r"""
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)

lower = array([[0.0], [2.0]])
upper = array([[1.0], [3.0]])
weights = array([0.25, 0.75])
distribution = LinearBoxParticleDistribution(lower, upper, weights)

lower[0, 0] = -99.0
upper[0, 0] = 99.0
weights[:] = array([0.75, 0.25])

npt.assert_allclose(to_numpy(distribution.lower), [[0.0], [2.0]])
npt.assert_allclose(to_numpy(distribution.upper), [[1.0], [3.0]])
npt.assert_allclose(to_numpy(distribution.w), [0.25, 0.75])

boxes = array([[[0.0], [1.0]], [[2.0], [3.0]]])
packed_distribution = LinearBoxParticleDistribution(boxes)
boxes[0, 0, 0] = -99.0
boxes[0, 1, 0] = 99.0

npt.assert_allclose(to_numpy(packed_distribution.lower), [[0.0], [2.0]])
npt.assert_allclose(to_numpy(packed_distribution.upper), [[1.0], [3.0]])
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env(backend_name),
    )
