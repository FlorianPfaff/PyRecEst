import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from pyrecest._backend.jax import random  # noqa: E402


@pytest.mark.parametrize(
    "bad_mean",
    [
        True,
        np.bool_(True),
        jnp.array([True, False]),
        [0.0, np.bool_(True)],
        np.array([0.0, np.bool_(False)], dtype=object),
    ],
)
def test_multivariate_normal_rejects_boolean_mean(bad_mean):
    with pytest.raises(TypeError, match="mean must be real numeric, not boolean"):
        random.multivariate_normal(bad_mean, jnp.eye(2))


@pytest.mark.parametrize("bad_mean", [[0.0, np.nan], [0.0, np.inf]])
def test_multivariate_normal_rejects_nonfinite_mean(bad_mean):
    with pytest.raises(ValueError, match="mean must be finite"):
        random.multivariate_normal(bad_mean, jnp.eye(2))


def test_multivariate_normal_rejects_nonvector_mean():
    with pytest.raises(ValueError, match="mean must be 1-dimensional"):
        random.multivariate_normal(jnp.zeros((1, 2)), jnp.eye(2))


@pytest.mark.parametrize(
    "bad_cov",
    [
        True,
        [[True, False], [False, True]],
        [[1.0, np.bool_(False)], [0.0, 1.0]],
        np.array([[1.0, 0.0], [0.0, np.bool_(True)]], dtype=object),
    ],
)
def test_multivariate_normal_rejects_boolean_covariance(bad_cov):
    with pytest.raises(TypeError, match="cov must be real numeric, not boolean"):
        random.multivariate_normal(jnp.zeros(2), bad_cov)


@pytest.mark.parametrize(
    "bad_cov",
    [
        [[1.0, 0.0], [0.0, np.nan]],
        [[1.0, 0.0], [0.0, np.inf]],
    ],
)
def test_multivariate_normal_rejects_nonfinite_covariance(bad_cov):
    with pytest.raises(ValueError, match="cov must be finite"):
        random.multivariate_normal(jnp.zeros(2), bad_cov)


@pytest.mark.parametrize(
    ("bad_cov", "message", "kwargs"),
    [
        ([1.0, 1.0], "cov must be a 2-dimensional square matrix", {}),
        (jnp.ones((2, 3)), "cov must have shape", {}),
        ([[1.0, 0.1], [0.0, 1.0]], "cov must be symmetric", {}),
        (
            [[1.0, 0.0], [0.0, -0.1]],
            "cov must be positive semidefinite",
            {"check_valid": "raise"},
        ),
    ],
)
def test_multivariate_normal_rejects_invalid_covariance_geometry(
    bad_cov, message, kwargs
):
    with pytest.raises(ValueError, match=message):
        random.multivariate_normal(jnp.zeros(2), bad_cov, **kwargs)


@pytest.mark.parametrize("size", [None, 3, (2, 3)])
def test_multivariate_normal_accepts_finite_positive_semidefinite_inputs(size):
    random.seed(0)

    sample = random.multivariate_normal([0, 1], [[2, 0], [0, 1]], size=size)

    expected_shape = (2,) if size is None else (*np.atleast_1d(size), 2)
    assert sample.shape == expected_shape
