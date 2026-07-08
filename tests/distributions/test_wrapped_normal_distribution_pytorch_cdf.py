import importlib.util
import os
import subprocess
import sys

import pytest


def _run_python_with_backend(code: str, backend_name: str) -> None:
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name

    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_wrapped_normal_cdf_uses_backend_erf_under_pytorch():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_python_with_backend(
        r"""
import math

import pyrecest.backend as backend
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)

assert backend.__backend_name__ == "pytorch"

dist = WrappedNormalDistribution(0.2, 0.7)
xs = backend.asarray([0.1, 0.4, 1.0])
values = dist.cdf(xs, starting_point=0.0, n_wraps=4)

assert hasattr(values, "detach")
assert values.shape == (3,)
values_np = backend.to_numpy(values)
assert (values_np >= -1e-7).all()
assert (values_np <= 1.0 + 1e-7).all()
assert values_np[0] <= values_np[1] <= values_np[2]

scalar_value = dist.cdf(backend.asarray(0.4), starting_point=0.0, n_wraps=4)
assert scalar_value.ndim == 0
assert math.isfinite(float(backend.to_numpy(scalar_value)))
""",
        backend_name="pytorch",
    )
