"""Public 3-D tracker contracts and compatibility with the planar filter."""

import importlib.util

import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial.transform import Rotation

import pyrecest.backend as backend
from pyrecest.filters import MEMEKFStar3DTracker, MEMEKFStarTracker, MemEkfStar3DTracker
from tests.support.backend_runner import run_backend_code


def make_tracker(**overrides):
    parameters = {
        "kinematic_state": [0.3, -0.5, 0.4, 0.1, 0.2, -0.1],
        "covariance": [0.3, 0.4, 0.5, 0.01, 0.02, 0.03],
        "orientation": Rotation.from_rotvec([0.2, -0.3, 0.4]).as_matrix(),
        "semi_axes": [3.0, 2.0, 1.0],
        "shape_covariance": [0.02, 0.03, 0.01, 0.1, 0.2, 0.05],
    }
    parameters.update(overrides)
    return MEMEKFStar3DTracker(**parameters)


def snapshot(tracker):
    return {
        name: np.array(backend.to_numpy(getattr(tracker, name)), copy=True)
        for name in (
            "kinematic_state",
            "covariance",
            "orientation",
            "semi_axes",
            "shape_covariance",
        )
    }


def assert_unchanged(tracker, before):
    for name, value in before.items():
        npt.assert_array_equal(backend.to_numpy(getattr(tracker, name)), value)


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("covariance", [0.3, 0.4, -0.5, 0.1, 0.2, 0.3]),
        ("shape_covariance", np.full((6, 6), np.nan)),
        ("multiplicative_noise_cov", [0.2, 0.0, 0.2]),
        ("orientation", np.diag([1.0, 1.0, -1.0])),
        ("orientation", np.diag([1.0, 1.0, 1.1])),
        ("orientation", np.full((3, 3), np.nan)),
        ("kinematic_state", [0.0, 0.0, np.inf]),
        ("semi_axes", [3.0, 2.0, 0.0]),
        ("semi_axes", [3.0, 2.0, np.nan]),
        ("minimum_axis_length", 0.0),
        ("minimum_axis_length", np.inf),
        ("covariance_regularization", -0.1),
        ("covariance_regularization", np.nan),
    ],
)
def test_invalid_constructor_parameters_have_named_errors(name, invalid):
    with pytest.raises(ValueError, match=name):
        make_tracker(**{name: invalid})


@pytest.mark.parametrize(
    "parameters",
    [
        {"system_matrix": np.eye(5)},
        {"system_matrix": np.full((6, 6), np.nan)},
        {"sys_noise": [0.1, 0.0, 0.1, -0.1, 0.0, 0.0]},
        {"inputs": [0.0, 0.0, 0.0]},
        {"inputs": [0.0, 0.0, 0.0, 0.0, np.inf, 0.0]},
        {"shape_sys_noise": [0.0, 0.0, -0.1, 0.0, 0.0, 0.0]},
        {"rotation_increment": [0.1, np.nan, 0.2]},
        {"rotation_increment": np.eye(3)},
    ],
)
def test_prediction_validation_preserves_all_state_and_history(parameters):
    tracker = make_tracker(log_prior_estimates=True, log_prior_extents=True)
    before = snapshot(tracker)
    arguments = {"system_matrix": np.eye(6), "inputs": np.ones(6)}
    arguments.update(parameters)
    with pytest.raises(ValueError):
        tracker.predict_linear(**arguments)
    assert_unchanged(tracker, before)
    assert tracker.prior_estimates_over_time.shape[1] == 0
    assert tracker.prior_extents_over_time.shape[1] == 0


@pytest.mark.parametrize(
    "parameters",
    [
        {"measurements": [[0.0, 0.0, 0.0], [0.0, np.nan, 1.0]]},
        {"meas_mat": np.eye(2, 6)},
        {"meas_noise_cov": [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
        {"multiplicative_noise_cov": [0.2, 0.0, 0.2]},
    ],
)
def test_update_validation_preserves_state(parameters):
    tracker = make_tracker(log_posterior_estimates=True, log_posterior_extents=True)
    before = snapshot(tracker)
    arguments = {"measurements": [1.0, 2.0, 3.0]}
    arguments.update(parameters)
    with pytest.raises(ValueError):
        tracker.update(**arguments)
    assert_unchanged(tracker, before)
    assert tracker.posterior_estimates_over_time.shape[1] == 0
    assert tracker.posterior_extents_over_time.shape[1] == 0


def test_psd_initial_covariances_allow_known_center_and_shape():
    tracker = make_tracker(covariance=0.0, shape_covariance=0.0)
    before = snapshot(tracker)
    tracker.predict(np.eye(6), sys_noise=0.0, shape_sys_noise=0.0)
    tracker.update([[1.0, 0.5, -0.2], [-0.4, 0.7, 0.9]])
    assert_unchanged(tracker, before)


def test_custom_measurement_matrix_matches_linear_update_with_known_shape():
    generator = np.random.default_rng(731)
    factor = generator.normal(size=(8, 8))
    covariance = factor @ factor.T + np.eye(8)
    measurement_matrix = np.eye(8)[[4, 1, 6], :]
    state = np.arange(8, dtype=float) / 10.0
    tracker = make_tracker(
        kinematic_state=state,
        covariance=covariance,
        measurement_matrix=measurement_matrix,
        shape_covariance=0.0,
    )
    extent = backend.to_numpy(tracker.get_point_estimate_extent())
    measurement = [0.8, -0.6, 1.3]
    measurement_noise = np.diag([0.1, 0.2, 0.3])
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T
        + 0.2 * extent
        + measurement_noise
    )
    gain = np.linalg.solve(innovation_covariance, measurement_matrix @ covariance).T
    expected_state = state + gain @ (measurement - measurement_matrix @ state)
    expected_covariance = covariance - gain @ innovation_covariance @ gain.T
    tracker.update(measurement, meas_noise_cov=measurement_noise)
    npt.assert_allclose(
        backend.to_numpy(tracker.kinematic_state), expected_state, atol=1e-12
    )
    npt.assert_allclose(
        backend.to_numpy(tracker.covariance), expected_covariance, atol=1e-12
    )


@pytest.mark.parametrize("scaling_factor", [0.5, 1.0, 2.0])
def test_surface_points_lie_on_scaled_ellipsoid_with_custom_center(scaling_factor):
    measurement_matrix = np.eye(6)[[3, 1, 5], :]
    tracker = make_tracker(measurement_matrix=measurement_matrix)
    points = backend.to_numpy(
        tracker.get_contour_points(211, scaling_factor=scaling_factor)
    )
    center = measurement_matrix @ backend.to_numpy(tracker.kinematic_state)
    extent = backend.to_numpy(tracker.get_point_estimate_extent())
    centered = points - center
    squared_radius = np.einsum("ni,ij,nj->n", centered, np.linalg.inv(extent), centered)
    assert points.shape == (211, 3)
    npt.assert_allclose(squared_radius, scaling_factor**2, atol=1e-13)


def test_planar_subproblem_matches_existing_mem_ekf_star():
    angle = 0.3
    state_2d = np.array([0.1, -0.2])
    covariance_2d = np.diag([0.2, 0.3])
    shape_covariance_2d = np.diag([0.01, 0.15, 0.08])
    planar = MEMEKFStarTracker(
        state_2d, covariance_2d, [angle, 3.0, 1.5], shape_covariance_2d
    )
    shape_covariance_3d = np.zeros((6, 6))
    planar_indices = [2, 3, 4]
    shape_covariance_3d[np.ix_(planar_indices, planar_indices)] = shape_covariance_2d
    spatial = make_tracker(
        kinematic_state=np.r_[state_2d, 0.0],
        covariance=np.diag([0.2, 0.3, 0.0]),
        orientation=Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix(),
        semi_axes=[3.0, 1.5, 1.0],
        shape_covariance=shape_covariance_3d,
        multiplicative_noise_cov=0.25,
    )
    planar.update([1.3, -0.8], meas_noise_cov=0.05)
    spatial.update([1.3, -0.8, 0.0], meas_noise_cov=0.05)
    npt.assert_allclose(
        backend.to_numpy(spatial.kinematic_state[:2]),
        backend.to_numpy(planar.kinematic_state),
        atol=1e-12,
    )
    npt.assert_allclose(
        backend.to_numpy(spatial.covariance[:2, :2]),
        backend.to_numpy(planar.covariance),
        atol=1e-12,
    )
    npt.assert_allclose(
        backend.to_numpy(spatial.semi_axes[:2]),
        backend.to_numpy(planar.shape_state[1:]),
        atol=1e-12,
    )
    planar_rotation = Rotation.from_rotvec(
        [0.0, 0.0, float(planar.shape_state[0])]
    ).as_matrix()
    npt.assert_allclose(
        backend.to_numpy(spatial.orientation), planar_rotation, atol=1e-12
    )
    spatial_shape_covariance = backend.to_numpy(spatial.shape_covariance)
    npt.assert_allclose(
        spatial_shape_covariance[np.ix_(planar_indices, planar_indices)],
        backend.to_numpy(planar.shape_covariance),
        atol=1e-12,
    )


def test_corrected_axis_is_clipped_to_configured_positive_boundary():
    tracker = make_tracker(
        kinematic_state=np.zeros(3),
        covariance=0.0,
        orientation=np.eye(3),
        semi_axes=[1.0, 2.0, 3.0],
        shape_covariance=[0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        minimum_axis_length=0.99,
    )
    tracker.update([0.0, 0.0, 0.0])
    npt.assert_allclose(backend.to_numpy(tracker.semi_axes), [0.99, 2.0, 3.0])
    assert np.linalg.eigvalsh(backend.to_numpy(tracker.extent)).min() > 0.0


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "jax", "pytorch"])
def test_lazy_public_exports_and_backend_update(backend_name):
    dependency = {"numpy": "numpy", "jax": "jax", "pytorch": "torch"}[backend_name]
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    result = run_backend_code(
        backend_name,
        """
import sys
import numpy as np
import pyrecest.backend as backend
import pyrecest.filters as filters
assert 'pyrecest.filters.mem_ekf_star_3d_tracker' not in sys.modules
tracker_class = filters.MEMEKFStar3DTracker
assert tracker_class is filters.MemEkfStar3DTracker
tracker = tracker_class([0.0, 0.0, 0.0], 0.1, np.eye(3), [3.0, 2.0, 1.0], 0.01)
tracker.predict(np.eye(3), rotation_increment=[0.1, -0.05, 0.03])
tracker.update([1.0, -0.4, 0.3], meas_noise_cov=0.02)
assert isinstance(tracker.orientation, type(backend.eye(3)))
np.testing.assert_allclose(backend.to_numpy(tracker.orientation.T @ tracker.orientation), np.eye(3), atol=2e-6)
assert np.isfinite(backend.to_numpy(tracker.get_point_estimate())).all()
print(backend.__backend_name__)
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == backend_name


def test_public_class_alias():
    assert MemEkfStar3DTracker is MEMEKFStar3DTracker
