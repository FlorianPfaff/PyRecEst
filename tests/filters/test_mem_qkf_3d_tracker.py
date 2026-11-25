"""Numerical references and geometric invariants for the spatial MEM-QKF."""

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

import pyrecest.backend as backend
from pyrecest.filters import MEMQKF3DTracker, MemQkf3DTracker


def make_tracker(**overrides):
    parameters = {
        "kinematic_state": [0.2, -0.3, 0.5],
        "covariance": [0.25, 0.2, 0.15],
        "orientation": Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix(),
        "semi_axes": [3.0, 1.8, 1.0],
        "shape_covariance": [0.025, 0.018, 0.03, 0.09, 0.05, 0.07],
    }
    parameters.update(overrides)
    return MEMQKF3DTracker(**parameters)


@pytest.mark.parametrize("dtype", [int, np.float32, np.float64])
def test_numeric_input_types_match_floating_reference(dtype):
    def run(dtype):
        def convert(value):
            converted = np.asarray(value, dtype=dtype)
            return converted.tolist() if dtype is int else converted

        tracker = MEMQKF3DTracker(
            convert([0, 0, 0]),
            convert(np.eye(3)),
            convert(np.eye(3)),
            convert([3, 2, 1]),
            convert(np.eye(6)),
            measurement_matrix=convert(np.eye(3)),
        )
        tracker.predict(convert(np.eye(3)), rotation_increment=convert([0, 0, 0]))
        tracker.update(convert([2, 1, 1]), meas_noise_cov=np.float32(0.125))
        return backend.to_numpy(tracker.get_point_estimate())

    assert_allclose(run(dtype), run(np.float64), rtol=2e-6, atol=2e-7)


def test_orientation_quadratic_moments_against_independent_gaussian_integration():
    # Rank-one tangent uncertainty about an oblique axis. Integrate the noisy
    # measurements themselves (rather than evaluating a conditional moment).
    direction = np.array([0.48, -0.6, 0.64])
    direction /= np.linalg.norm(direction)
    sigma = 0.045
    tracker = make_tracker(
        shape_covariance=block_diag(
            sigma**2 * np.outer(direction, direction), np.eye(3)
        )
    )
    source = np.array([[2.0, 0.12, -0.03], [0.12, 0.9, 0.07], [-0.03, 0.07, 0.4]])
    noise = np.array([[0.22, 0.07, 0.04], [0.07, 0.18, 0.0], [0.04, 0.0, 0.15]])
    products = []
    deltas = []
    weights = []
    nodes, one_dim_weights = np.polynomial.hermite_e.hermegauss(15)
    one_dim_weights /= np.sqrt(2.0 * np.pi)
    normal_nodes, normal_weights = np.polynomial.hermite_e.hermegauss(3)
    normal_weights /= np.sqrt(2.0 * np.pi)
    prior = backend.to_numpy(tracker.orientation)
    for node, weight in zip(nodes, one_dim_weights):
        delta = sigma * node * direction
        frame = prior @ Rotation.from_rotvec(delta).as_matrix()
        root = np.linalg.cholesky(frame @ source @ frame.T + noise)
        for index in product(range(3), repeat=3):
            point = root @ normal_nodes[list(index)]
            products.append([point[i] * point[j] for i, j in tracker._quadratic_pairs])
            deltas.append(delta)
            weights.append(weight * np.prod(normal_weights[list(index)]))
    products, deltas, weights = map(np.asarray, (products, deltas, weights))
    expected_mean = weights @ products
    deviations = products - expected_mean
    expected_covariance = deviations.T @ (weights[:, None] * deviations)
    expected_cross = deltas.T @ (weights[:, None] * deviations)
    moments = tracker._orientation_moments(backend.array(source), backend.array(noise))
    for actual, expected in zip(
        moments, (expected_mean, expected_covariance, expected_cross)
    ):
        assert_allclose(backend.to_numpy(actual), expected, atol=1e-7, rtol=1e-7)


def test_axis_posterior_matches_scalar_gaussian_product_fourth_moment():
    axis, prior, source, noise, measurement = 2.6, 0.45, 0.32, 0.12, 2.4
    tracker = make_tracker(
        kinematic_state=np.zeros(3),
        covariance=0.0,
        orientation=np.eye(3),
        semi_axes=[axis, 1.5, 0.8],
        shape_covariance=[0.0, 0.0, 0.0, prior, 0.0, 0.0],
        multiplicative_noise_cov=source,
    )
    a2 = axis**2 + prior
    a4 = axis**4 + 6 * axis**2 * prior + 3 * prior**2
    mean_square = source * a2 + noise
    fourth = 3 * (source**2 * a4 + 2 * source * noise * a2 + noise**2)
    variance = fourth - mean_square**2
    cross = 2 * source * axis * prior
    tracker.update([measurement, 0.7, -0.5], meas_noise_cov=noise)
    assert_allclose(
        backend.to_numpy(tracker.semi_axes),
        [axis + cross / variance * (measurement**2 - mean_square), 1.5, 0.8],
    )
    assert_allclose(
        backend.to_numpy(tracker.shape_covariance)[3, 3], prior - cross**2 / variance
    )


def test_fixed_shape_matches_linear_kalman_update_with_custom_measurement_matrix():
    h = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.0]])
    state = np.array([0.3, -0.2, 0.0, 0.7])
    covariance = np.diag([0.2, 0.3, 0.4, 0.1])
    z = np.array([1.3, -0.5, 0.1])
    tracker = make_tracker(
        kinematic_state=state,
        covariance=covariance,
        shape_covariance=0.0,
        measurement_matrix=h,
    )
    extent = backend.to_numpy(tracker.get_point_estimate_extent())
    gain = np.linalg.solve(
        h @ covariance @ h.T + 0.2 * extent + 0.1 * np.eye(3), h @ covariance
    ).T
    tracker.update(z, meas_noise_cov=0.1)
    assert_allclose(
        backend.to_numpy(tracker.kinematic_state), state + gain @ (z - h @ state)
    )
    assert_allclose(
        backend.to_numpy(tracker.covariance), covariance - gain @ h @ covariance
    )
    assert_allclose(
        backend.to_numpy(tracker.get_point_estimate_extent()), extent, atol=1e-14
    )


def test_world_rotation_equivariance_and_sequential_scan_layout():
    tracker = make_tracker()
    turned = Rotation.from_rotvec([-0.65, 0.5, 0.2]).as_matrix()
    other = make_tracker(
        kinematic_state=turned @ backend.to_numpy(tracker.kinematic_state),
        orientation=turned @ backend.to_numpy(tracker.orientation),
        covariance=turned @ backend.to_numpy(tracker.covariance) @ turned.T,
    )
    points = np.array(
        [[2.1, 0.4, 0.9], [1.6, 0.1, -0.4], [0.6, 0.8, 0.5], [-0.3, -1.5, 0.9]]
    )
    noise = np.diag([0.04, 0.06, 0.02])
    tracker.update(points, meas_noise_cov=noise)
    for point in points:
        other.update(turned @ point, meas_noise_cov=turned @ noise @ turned.T)
    assert_allclose(
        backend.to_numpy(other.kinematic_state),
        turned @ backend.to_numpy(tracker.kinematic_state),
        atol=1e-12,
    )
    assert_allclose(
        backend.to_numpy(other.orientation),
        turned @ backend.to_numpy(tracker.orientation),
        atol=1e-12,
    )
    assert_allclose(
        backend.to_numpy(other.semi_axes),
        backend.to_numpy(tracker.semi_axes),
        atol=1e-12,
    )
    assert_allclose(
        backend.to_numpy(other.shape_covariance),
        backend.to_numpy(tracker.shape_covariance),
        atol=1e-12,
    )


def test_spherical_orientation_unobservable_and_prediction_drops_cross_blocks():
    initial = np.diag([0.08, 0.05, 0.1, 0.0, 0.0, 0.0])
    tracker = make_tracker(semi_axes=[2.0] * 3, shape_covariance=initial)
    orientation = backend.to_numpy(tracker.orientation)
    tracker.update([0.9, 1.1, -1.2], meas_noise_cov=0.05)
    assert_allclose(backend.to_numpy(tracker.orientation), orientation, atol=1e-14)
    assert_allclose(backend.to_numpy(tracker.shape_covariance), initial, atol=1e-14)

    crossed = np.eye(6) + 0.1
    tracker = make_tracker(shape_covariance=crossed)
    assert_allclose(
        backend.to_numpy(tracker.shape_covariance),
        block_diag(crossed[:3, :3], crossed[3:, 3:]),
    )
    increment = np.array([0.2, -0.3, 0.4])
    u = Rotation.from_rotvec(increment).as_matrix()
    tracker.predict(
        np.eye(3), rotation_increment=increment, shape_sys_noise=0.1 * crossed
    )
    expected = block_diag(
        u.T @ crossed[:3, :3] @ u, crossed[3:, 3:]
    ) + 0.1 * block_diag(crossed[:3, :3], crossed[3:, 3:])
    assert_allclose(backend.to_numpy(tracker.shape_covariance), expected, atol=1e-13)
    assert MemQkf3DTracker is MEMQKF3DTracker


@pytest.mark.skipif(
    backend.__backend_name__ != "numpy", reason="Seeded multi-scan regression on NumPy"
)
def test_recovers_spatial_triaxial_extent_and_center():
    rng = np.random.default_rng(37145)
    axes = np.array([3.5, 2.0, 0.9])
    frame = Rotation.from_rotvec([-0.5, 0.65, 0.35]).as_matrix()
    extent = (frame * axes**2) @ frame.T
    center = np.array([0.6, -0.2, 0.4])
    tracker = make_tracker(
        kinematic_state=center + [0.3, -0.1, 0.2],
        orientation=frame @ Rotation.from_rotvec([0.19, -0.13, 0.16]).as_matrix(),
        semi_axes=axes * [0.85, 1.1, 1.22],
        shape_covariance=[0.07] * 3 + [0.3] * 3,
    )
    initial_error = np.linalg.norm(tracker.get_point_estimate_extent() - extent)
    for scan in rng.multivariate_normal(
        center, 0.2 * extent + 0.03 * np.eye(3), size=(25, 24)
    ):
        tracker.update(scan, meas_noise_cov=0.03)
        assert np.linalg.eigvalsh(tracker.shape_covariance).min() >= -1e-12
        assert np.all(tracker.semi_axes > 0)
    assert np.linalg.norm(tracker.kinematic_state - center) < 0.16
    assert (
        np.linalg.norm(tracker.get_point_estimate_extent() - extent)
        < 0.4 * initial_error
    )
    assert Rotation.from_matrix(frame.T @ tracker.orientation).magnitude() < 0.12
