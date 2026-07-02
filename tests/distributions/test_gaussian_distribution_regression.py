import importlib.util
import math
import os
import subprocess
import sys
import unittest

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution


def _scalar(value):
    return float(np.asarray(to_numpy(value)).reshape(-1)[0])


def _subprocess_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


class GaussianDistributionRegressionTest(unittest.TestCase):
    def test_standard_normal_pdf_and_log_pdf_match_reference_values(self):
        distribution = GaussianDistribution(array([0.0]), array([[1.0]]))

        log_pdf = _scalar(distribution.ln_pdf(array([0.0])))
        pdf = _scalar(distribution.pdf(array([0.0])))

        self.assertAlmostEqual(log_pdf, -0.5 * math.log(2.0 * math.pi), places=12)
        self.assertAlmostEqual(pdf, 1.0 / math.sqrt(2.0 * math.pi), places=12)

    def test_gaussian_product_matches_closed_form_scalar_reference(self):
        first = GaussianDistribution(array([0.0]), array([[1.0]]))
        second = GaussianDistribution(array([1.0]), array([[4.0]]))

        product = first.multiply(second)

        npt.assert_allclose(
            to_numpy(product.mu), np.array([0.2]), rtol=1e-12, atol=1e-12
        )
        npt.assert_allclose(
            to_numpy(product.C), np.array([[0.8]]), rtol=1e-12, atol=1e-12
        )

    def test_invalid_covariance_shape_raises_value_error(self):
        with self.assertRaises(ValueError):
            GaussianDistribution(array([0.0, 1.0]), array([[1.0]]))

    def test_invalid_marginal_dimensions_raise_value_error(self):
        distribution = GaussianDistribution(
            array([0.0, 1.0]), array([[1.0, 0.0], [0.0, 1.0]])
        )

        with self.assertRaises(ValueError):
            distribution.marginalize_out([2])


def test_marginalize_out_uses_backend_index_arrays():
    distribution = GaussianDistribution(
        array([1.0, 2.0, 3.0]),
        array([[4.0, 0.1, 0.2], [0.1, 5.0, 0.3], [0.2, 0.3, 6.0]]),
    )

    marginalized = distribution.marginalize_out([1])

    npt.assert_allclose(to_numpy(marginalized.mu), np.array([1.0, 3.0]))
    npt.assert_allclose(to_numpy(marginalized.C), np.array([[4.0, 0.2], [0.2, 6.0]]))


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    ("backend_name", "module_name"),
    [("pytorch", "torch"), ("jax", "jax")],
)
def test_scalar_gaussian_log_pdf_preserves_scalar_shape_across_backends(
    backend_name, module_name
):
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"{module_name} is not installed")

    code = """
import numpy as np
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution

distribution = GaussianDistribution(array([0.0]), array([[1.0]]))
log_pdf = distribution.ln_pdf(array(0.0))
pdf = distribution.pdf(array(0.0))
assert np.asarray(to_numpy(log_pdf)).shape == ()
assert np.asarray(to_numpy(pdf)).shape == ()
np.testing.assert_allclose(
    np.asarray(to_numpy(log_pdf)),
    np.array(-0.5 * np.log(2.0 * np.pi)),
)
np.testing.assert_allclose(
    np.asarray(to_numpy(pdf)),
    np.array(1.0 / np.sqrt(2.0 * np.pi)),
)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env(backend_name))


@pytest.mark.backend_portable
def test_jax_marginalize_out_accepts_list_dimensions():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("jax is not installed")

    code = """
import numpy as np
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import GaussianDistribution

distribution = GaussianDistribution(
    array([1.0, 2.0, 3.0]),
    array([[4.0, 0.1, 0.2], [0.1, 5.0, 0.3], [0.2, 0.3, 6.0]]),
)
marginalized = distribution.marginalize_out([1])
np.testing.assert_allclose(to_numpy(marginalized.mu), np.array([1.0, 3.0]))
np.testing.assert_allclose(
    to_numpy(marginalized.C), np.array([[4.0, 0.2], [0.2, 6.0]])
)
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=_subprocess_env("jax"))


if __name__ == "__main__":
    unittest.main()
