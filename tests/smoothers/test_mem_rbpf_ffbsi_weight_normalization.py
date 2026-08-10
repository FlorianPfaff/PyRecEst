import numpy as np
import numpy.testing as npt
import pytest
from pyrecest import backend
from pyrecest.smoothers import MEMRBPFFFBSiSmoother, MEMRBPFForwardRecord

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="MEM-RBPF FFBSi tests use NumPy sampling paths",
)


def test_safe_probs_preserves_extreme_finite_weight_ratios():
    max_float = np.finfo(float).max

    probabilities = MEMRBPFFFBSiSmoother._safe_probs(
        np.array([max_float, max_float / 2.0, 0.0])
    )

    npt.assert_allclose(probabilities, np.array([2.0 / 3.0, 1.0 / 3.0, 0.0]))
    npt.assert_allclose(np.sum(probabilities), 1.0)


def test_smooth_accepts_extreme_finite_particle_weights():
    max_float = np.finfo(float).max
    axis_covariance = np.repeat((0.1 * np.eye(2))[np.newaxis, :, :], 2, axis=0)
    record = MEMRBPFForwardRecord(
        kinematic_state=np.array([0.0]),
        covariance=np.array([[1.0]]),
        theta=np.array([0.0, 0.2]),
        axis_mean=np.array([[2.0, 1.0], [1.8, 0.8]]),
        axis_covariance=axis_covariance,
        weights=np.array([max_float, max_float / 2.0]),
    )

    result = MEMRBPFFFBSiSmoother(n_trajectories=8, sample_axis=False).smooth(
        [record], rng=0, full_axis_lengths=False
    )

    assert result.index_samples.shape == (8, 1)
    assert np.all(np.isfinite(result.states))
