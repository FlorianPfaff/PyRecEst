import numpy as np
import pytest

jax = pytest.importorskip("jax")
from pyrecest._backend.jax import random  # noqa: E402


def test_multivariate_normal_zero_covariance_returns_mean():
    random.seed(0)
    mean = np.array([1.5, -2.0])

    sample = np.asarray(
        random.multivariate_normal(mean, np.zeros((2, 2)), size=8)
    )

    assert np.isfinite(sample).all()
    np.testing.assert_allclose(
        sample,
        np.broadcast_to(mean, sample.shape),
        rtol=0.0,
        atol=0.0,
    )


def test_multivariate_normal_rejects_negative_covariance_at_its_scale():
    with pytest.raises(ValueError, match="cov must be positive semidefinite"):
        random.multivariate_normal([0.0], [[-1.0e-10]])
