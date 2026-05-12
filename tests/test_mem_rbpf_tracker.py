import numpy as np
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker, MemRbpfTracker


def _make_tracker():
    return MEMRBPFTracker(
        kinematic_state=np.array([0.0, 0.0, 1.0, -0.5]),
        covariance=np.eye(4),
        shape_state=np.array([0.2, 2.0, 1.0]),
        shape_covariance=np.diag([0.05, 0.1, 0.1]),
        meas_noise_cov=0.05 * np.eye(2),
        sys_noise=0.01 * np.eye(4),
        shape_sys_noise=np.diag([0.01, 0.01, 0.01]),
        n_particles=32,
        rng=np.random.default_rng(0),
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
    tracker = MEMRBPFTracker.from_original_parameters(
        m_init=np.zeros(4),
        p_init=np.array([0.0, 2.0, 1.0]),
        p_kinematic_init=np.eye(4),
        p_shape_init=np.diag([0.01, 0.1, 0.1]),
        r=0.05 * np.eye(2),
        q_kinematic=0.01 * np.eye(4),
        q_shape=np.diag([0.02, 0.01, 0.01]),
        n_particles=8,
        rng=np.random.default_rng(1),
    )

    assert isinstance(tracker, MemRbpfTracker)
    assert tracker.get_state().shape == (7,)
    assert tracker.get_state_array(with_weight=True).shape == (8, 8)
