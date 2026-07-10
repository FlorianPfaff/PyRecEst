import numpy as np
import pytest
from pyrecest import backend
from pyrecest.backend import array, diag, eye, random
from pyrecest.filters.mem_rbpf_tracker import MEMRBPFTracker, MemRbpfTracker

pytestmark = pytest.mark.skipif(
    backend.__backend_name__ != "numpy",
    reason="MEM-RBPF tests cover resampling paths currently supported on numpy only",
)


def _make_tracker(**kwargs):
    random.seed(0)
    parameters = {
        "kinematic_state": array([0.0, 0.0, 1.0, -0.5]),
        "covariance": eye(4),
        "shape_state": array([0.2, 2.0, 1.0]),
        "shape_covariance": diag(array([0.05, 0.1, 0.1])),
        "meas_noise_cov": 0.05 * eye(2),
        "sys_noise": 0.01 * eye(4),
        "shape_sys_noise": diag(array([0.01, 0.01, 0.01])),
        "n_particles": 32,
        "resampling_threshold": 16,
        "axis_floor": 1e-3,
    }
    parameters.update(kwargs)
    return MEMRBPFTracker(**parameters)


def test_mem_rbpf_validates_particle_count():
    for invalid in (True, np.bool_(True), 1.5, np.array(1.5), np.array([3]), 0, -1):
        with pytest.raises(ValueError, match="n_particles"):
            _make_tracker(n_particles=invalid)

    for valid in (np.int64(3), np.array(3)):
        tracker = _make_tracker(n_particles=valid)
        assert tracker.n_particles == 3
        assert tracker.weights.shape == (3,)


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


def test_mem_rbpf_systematic_resampling_closes_roundoff_gap(monkeypatch):
    tracker = _make_tracker(
        n_particles=3,
        resampling_mode="systematic",
        resampling_threshold=0,
    )
    tracker.weights = array(
        [0.20381898702851367, 0.7463113329614236, 0.0498696800100626]
    )
    monkeypatch.setattr(
        random,
        "uniform",
        lambda size=None: array(np.nextafter(1.0, 0.0)),
    )

    indices = np.asarray(tracker._resample_indices())

    np.testing.assert_array_equal(indices, np.array([1, 1, 2]))
    tracker.resample()
    assert tracker.theta.shape == (3,)


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


def test_mem_rbpf_shape_estimate_is_invariant_to_pi_shifted_orientation():
    tracker = MEMRBPFTracker(
        kinematic_state=array([0.0, 0.0, 0.0, 0.0]),
        covariance=eye(4),
        shape_state=array([0.0, 2.0, 1.0]),
        shape_covariance=diag(array([1e-6, 1e-6, 1e-6])),
        meas_noise_cov=0.05 * eye(2),
        sys_noise=0.01 * eye(4),
        shape_sys_noise=diag(array([0.0, 0.0, 0.0])),
        n_particles=2,
        resampling_threshold=0,
    )

    angle = 0.37
    tracker.theta = array([angle, angle + np.pi])
    tracker.axis = array([[2.0, 1.0], [2.0, 1.0]])
    tracker.weights = array([0.5, 0.5])

    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    expected_extent = rotation @ np.diag([4.0, 1.0]) @ rotation.T

    assert np.allclose(
        np.asarray(tracker.get_point_estimate_extent()),
        expected_extent,
    )
