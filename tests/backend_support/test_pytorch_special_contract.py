import importlib.util
import math
import os
import subprocess
import sys

import pytest


def _run_python(code, *, backend_name=None):
    env = os.environ.copy()
    if backend_name is None:
        env.pop("PYRECEST_BACKEND", None)
    else:
        env["PYRECEST_BACKEND"] = backend_name

    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
def test_raw_pytorch_special_helpers_are_patched_under_default_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_python(r"""
import math

import pyrecest  # noqa: F401
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_backend

assert public_backend.__backend_name__ == "numpy"


def as_list(value):
    converted = raw_backend.to_numpy(value)
    actual = converted.tolist() if hasattr(converted, "tolist") else converted
    return actual if isinstance(actual, list) else [actual]


def assert_close(values, expected):
    actual = as_list(values)
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-6)


root_pi = math.sqrt(math.pi)
values = [-0.5, 0.5, 2.5]
assert_close(raw_backend.gamma(values), [-2.0 * root_pi, root_pi, 0.75 * root_pi])
assert_close(
    raw_backend.gammaln(values),
    [math.log(2.0 * root_pi), math.log(root_pi), math.log(0.75 * root_pi)],
)
assert_close(raw_backend.erf([0.0, 1.0]), [0.0, math.erf(1.0)])
assert_close(
    raw_backend.polygamma(1, [1.0, 2.0]),
    [math.pi**2 / 6.0, math.pi**2 / 6.0 - 1.0],
)

pole_values = as_list(raw_backend.gamma([-1.0, 0.0, -0.0]))
assert math.isnan(pole_values[0])
assert math.isinf(pole_values[1]) and pole_values[1] > 0
assert math.isinf(pole_values[2]) and pole_values[2] < 0
""")


@pytest.mark.backend_portable
def test_public_pytorch_special_helpers_accept_arraylike_and_out():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    _run_python(
        r"""
import math

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert backend.__backend_name__ == "pytorch"


def as_list(module, value):
    converted = module.to_numpy(value)
    actual = converted.tolist() if hasattr(converted, "tolist") else converted
    return actual if isinstance(actual, list) else [actual]


def assert_close(module, values, expected):
    actual = as_list(module, values)
    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected):
        assert math.isclose(actual_value, expected_value, rel_tol=1e-6, abs_tol=1e-6)


root_pi = math.sqrt(math.pi)
values = [-0.5, 0.5, 2.5]
out = backend.empty_like(backend.asarray([0.0, 0.0, 0.0]))
returned = backend.gamma(values, out=out)

assert returned is out
assert_close(backend, out, [-2.0 * root_pi, root_pi, 0.75 * root_pi])
for special_backend in (backend, raw_backend):
    assert_close(special_backend, special_backend.gamma(values), [-2.0 * root_pi, root_pi, 0.75 * root_pi])
    assert_close(
        special_backend,
        special_backend.polygamma(1, [1.0, 2.0]),
        [math.pi**2 / 6.0, math.pi**2 / 6.0 - 1.0],
    )
""",
        backend_name="pytorch",
    )
