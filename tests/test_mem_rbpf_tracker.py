import numpy as np
import pytest

from pyrecest import backend
from pyrecest.backend import array, diag, eye, random
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker, MemRbpfTracker

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="MEM-RBPF tests cover resampling paths currently supported on numpy only",
)


def _make_tracker():
    random.seed(0)
    return MEMRBPFTracker(
        kinematic_state=array([0.0, 0.0, 1.0, -0.5]),
        covariance=eye(4),
        shape_state=array([0.2, 2.0, 1.0]),
        shape_covariance=diag(array([0.05, 0.1, 0.1])),
        meas_noise_cov=0.05 * eye(2),
        sys_noise=0.01 * eye(4),
        shape_sys_noise=diag(array([0.01, 0.01, 0.01])),
        n_particles=32,
        resampling_threshold=16,
        axis_floor=1e-3,
    )


def test_mem_rbpf_predict_update_smoke():
    tracker = _make_tracker()
    tracker.predict()
    tracker.update(np.array([[1.2, 0.1], [0.8, -0.2], [1.0, 0.2]]))

    estimate = tracker.get_point_estimate()
    extent = tracker.get_point_estimate_extent()
    contour = tracker.get_contour_points(12)

    assert estimate.shape == (7,)
    assert extent.shape == (2, 2)
    assert contour.shape == (12, 2)
    assert np.all(np.isfinite(np.asarray(estimate)))
    assert np.all(np.linalg.eigvalsh(np.asarray(extent)) >= -1e-10)
    assert np.isclose(np.sum(np.asarray(tracker.weights)), 1.0)


def test_mem_rbpf_original_parameter_constructor_alias():
    random.seed(1)
    tracker = MEMRBPFTracker.from_original_parameters(
        m_init=array([0.0, 0.0, 0.0, 0.0]),
        p_init=array([0.0, 2.0, 1.0]),
        p_kinematic_init=eye(4),
        p_shape_init=diag(array([0.01, 0.1, 0.1])),
        r=0.05 * eye(2),
        q_kinematic=0.01 * eye(4),
        q_shape=diag(array([0.02, 0.01, 0.01])),
        n_particles=8,
    )

    assert isinstance(tracker, MemRbpfTracker)
    assert tracker.get_state().shape == (7,)
    assert tracker.get_state_array(with_weight=True).shape == (8, 8)
