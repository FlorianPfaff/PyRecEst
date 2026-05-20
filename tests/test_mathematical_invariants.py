import math

import numpy as np
import numpy.testing as npt
import pyrecest.backend as be


def _to_numpy(value):
    try:
        value = be.to_numpy(value)
    except Exception:
        pass
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = np.asarray(value.tolist())
    return np.asarray(value)


def test_covariance_construction_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(12345)
    for _ in range(10):
        samples = be.array(rng.normal(size=(5, 3)))
        covariance = be.matmul(be.transpose(samples), samples)
        covariance_np = _to_numpy(covariance)
        npt.assert_allclose(covariance_np, covariance_np.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(covariance_np)
        assert np.min(eigvals) >= -1e-8


def test_weights_normalize_to_probability_vector():
    rng = np.random.default_rng(6789)
    for _ in range(10):
        raw = be.array(rng.uniform(0.01, 2.0, size=12))
        weights = raw / be.sum(raw)
        weights_np = _to_numpy(weights)
        assert np.all(weights_np >= 0.0)
        npt.assert_allclose(np.sum(weights_np), 1.0, atol=1e-7)


def test_circular_trigonometric_representation_is_periodic():
    angles = be.array(np.linspace(-math.pi, math.pi, 17))
    shifted = angles + 2.0 * math.pi
    npt.assert_allclose(
        _to_numpy(be.sin(angles)), _to_numpy(be.sin(shifted)), atol=1e-7
    )
    npt.assert_allclose(
        _to_numpy(be.cos(angles)), _to_numpy(be.cos(shifted)), atol=1e-7
    )
