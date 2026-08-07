import numpy as np
import pytest

jax = pytest.importorskip("jax")
from pyrecest._backend.jax import random  # noqa: E402


def test_multivariate_normal_zero_covariance_returns_mean():
    random.seed(0)
    mean = np.array([1.5, -2.0])

    sample = np.asarray(random.multivariate_normal(mean, np.zeros((2, 2)), size=8))

    assert np.isfinite(sample).all()
    np.testing.assert_allclose(
        sample,
        np.broadcast_to(mean, sample.shape),
        rtol=0.0,
        atol=0.0,
    )


def test_multivariate_normal_rank_one_covariance_returns_finite_samples():
    random.seed(0)

    sample = np.asarray(
        random.multivariate_normal(
            [1.5, -2.0],
            [[1.0, 1.0], [1.0, 1.0]],
            size=8,
        )
    )

    assert np.isfinite(sample).all()
