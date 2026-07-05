import importlib.util
import os
import subprocess
import sys

import pytest


@pytest.mark.backend_portable
def test_affine_transform_homogeneous_matrix_works_on_jax_backend():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "jax"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import numpy.testing as npt

from pyrecest.backend import array, to_numpy
from pyrecest.utils.point_set_registration import AffineTransform

transform = AffineTransform(
    array([[1.0, 2.0], [3.0, 4.0]]),
    array([5.0, 6.0]),
)
homogeneous = transform.homogeneous_matrix()

npt.assert_allclose(
    to_numpy(homogeneous),
    [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [0.0, 0.0, 1.0]],
)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
