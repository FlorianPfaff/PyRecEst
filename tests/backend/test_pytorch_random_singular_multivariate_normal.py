import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pyrecest._backend.pytorch import random  # noqa: E402


def test_multivariate_normal_zero_covariance_returns_mean():
    random.seed(0)
    mean = np.array([1.5, -2.0])

    sample = (
        random.multivariate_normal(mean, np.zeros((2, 2)), size=8)
        .detach()
        .cpu()
        .numpy()
    )

    assert np.isfinite(sample).all()
    np.testing.assert_allclose(
        sample,
        np.broadcast_to(mean, sample.shape),
        rtol=0.0,
        atol=0.0,
    )


def test_multivariate_normal_rank_one_covariance_stays_on_support():
    random.seed(0)
    mean = np.array([1.5, -2.0])

    sample = (
        random.multivariate_normal(
            mean,
            [[1.0, 1.0], [1.0, 1.0]],
            size=8,
        )
        .detach()
        .cpu()
        .numpy()
    )

    assert np.isfinite(sample).all()
    np.testing.assert_allclose(
        sample[:, 0] - mean[0],
        sample[:, 1] - mean[1],
        rtol=1e-6,
        atol=1e-6,
    )
