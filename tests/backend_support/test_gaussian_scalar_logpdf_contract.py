import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


@pytest.mark.parametrize("backend_name", ["numpy", "pytorch", "jax"])
def test_one_dimensional_gaussian_scalar_logpdf_preserves_scalar_shape(backend_name):
    if backend_name == "pytorch" and importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    if backend_name == "jax" and importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest.distributions import GaussianDistribution


distribution = GaussianDistribution(backend.array(0.0), backend.array(1.0))
log_density = distribution.ln_pdf(backend.array(0.0))
density = distribution.pdf(backend.array(0.0))
batch_log_density = distribution.ln_pdf(backend.array([0.0]))

assert tuple(backend.shape(backend.asarray(log_density))) == ()
assert tuple(backend.shape(backend.asarray(density))) == ()
assert tuple(backend.shape(backend.asarray(batch_log_density))) == (1,)
assert float(backend.to_numpy(log_density)) < 0.0
assert float(backend.to_numpy(density)) > 0.0
print("ok")
"""
    result = run_backend_code(backend_name, code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
