"""Regression tests for the Rao-Blackwellized spatial ellipsoid tracker."""

import unittest
from unittest.mock import patch

import numpy as np
from numpy import testing as npt
from scipy.spatial.transform import Rotation
from scipy.special import softmax
from scipy.stats import multivariate_normal

from pyrecest import backend
from pyrecest.filters import MEMRBPF3DTracker, MemRbpf3DTracker


def make_tracker(**kwargs):
    parameters = dict(
        kinematic_state=[0.2, -0.4, 0.1],
        covariance=[0.2, 0.3, 0.15],
        orientation=Rotation.from_rotvec([0.3, -0.5, 0.2]).as_matrix(),
        semi_axes=[3.8, 2.2, 1.1],
        shape_covariance=[0.1, 0.08, 0.07, 0.15, 0.09, 0.03],
        n_particles=40,
        resampling_threshold=0,
        rng=19,
    )
    parameters.update(kwargs)
    return MEMRBPF3DTracker(**parameters)


@unittest.skipUnless(backend.__backend_name__ == "numpy", "NumPy particle backend")
class TestMEMRBPF3DTracker(unittest.TestCase):
    def test_shape_noise_conditions_axes_on_sampled_rotations(self):
        tangent_cov = np.diag([0.08, 0.01, 0])  # rank deficient
        coupling = np.array([[0.2, 0.03, 0], [-0.1, 0.3, 0], [0.5, 0, 0]])
        residual = np.array([[0.1, -0.01, 0.03], [-0.01, 0.05, 0], [0.03, 0, 0.08]])
        cross = coupling @ tangent_cov
        covariance = np.block(
            [[tangent_cov, cross.T], [cross, residual + cross @ coupling.T]]
        )
        orientation = Rotation.from_rotvec([0.3, -0.6, 0.4]).as_matrix()
        axes = [3.1, 2.0, 1.5]
        tracker = make_tracker(
            orientation=orientation, semi_axes=axes, shape_covariance=covariance
        )
        tangent = Rotation.from_matrix(orientation.T @ tracker.rotations).as_rotvec()
        npt.assert_allclose(tracker.axis, axes + tangent @ coupling.T, atol=1e-12)
        npt.assert_allclose(
            tracker.axis_covariances,
            np.tile(residual, (tracker.n_particles, 1, 1)),
            atol=1e-14,
        )
        prior_axes = tracker.axis.copy()
        increment = [0.35, -0.2, 0.1]
        expected = tracker.rotations @ Rotation.from_rotvec(increment).as_matrix()
        tracker.predict(
            np.eye(3), shape_sys_noise=covariance, rotation_increment=increment
        )
        actual = Rotation.from_matrix(
            expected.swapaxes(1, 2) @ tracker.rotations
        ).as_rotvec()
        npt.assert_allclose(tracker.axis, prior_axes + actual @ coupling.T, atol=1e-12)
        npt.assert_allclose(
            tracker.axis_covariances,
            np.tile(2 * residual, (tracker.n_particles, 1, 1)),
            atol=1e-14,
        )

    def test_fixed_axes_weights_match_full_correlated_scan_density(self):
        rng = np.random.default_rng(4)
        orientation = Rotation.from_rotvec([0.37, 0.42, -0.1]).as_matrix()
        source = np.array([[0.2, -0.04, 0.03], [-0.04, 0.3, 0.05], [0.03, 0.05, 0.17]])
        sensor = np.array(
            [[0.08, -0.025, 0.01], [-0.025, 0.1, 0.02], [0.01, 0.02, 0.05]]
        )
        h = rng.normal(size=(3, 5))
        state = rng.normal(size=5)
        covariance = np.diag([0.1, 0.15, 0.05, 0.03, 0.2])
        tracker = make_tracker(
            kinematic_state=state,
            covariance=covariance,
            orientation=orientation,
            shape_covariance=[0.13, 0.20, 0.08, 0, 0, 0],
            n_particles=11,
            measurement_matrix=h,
            multiplicative_noise_cov=source,
        )
        points = rng.normal(size=(4, 3)) + h @ state
        center_cov = h @ covariance @ h.T
        log_density, spreads = [], []
        for rotation, axes in zip(tracker.rotations, tracker.axis):
            factor = rotation * axes
            spread = sensor + factor @ source @ factor.T
            spreads.append(spread)
            # Exact stacked Gaussian after analytically integrating the shared center.
            joint = np.kron(np.eye(len(points)), spread) + np.kron(
                np.ones((len(points), len(points))), center_cov
            )
            log_density.append(
                multivariate_normal.logpdf((points - h @ state).ravel(), cov=joint)
            )
        mean_noise = np.mean(spreads, axis=0) / len(points)
        gain = covariance @ h.T @ np.linalg.inv(center_cov + mean_noise)
        prior_axes = tracker.axis.copy()
        tracker.update(points, meas_noise_cov=sensor)
        npt.assert_allclose(tracker.weights, softmax(log_density), atol=1e-12)
        npt.assert_allclose(tracker.axis, prior_axes)
        npt.assert_allclose(
            tracker.kinematic_state,
            state + gain @ (points.mean(axis=0) - h @ state),
            atol=1e-12,
        )
        npt.assert_allclose(
            tracker.covariance, covariance - gain @ h @ covariance, atol=1e-12
        )

    def test_singleton_axis_recursion_matches_gaussian_product_moments(self):
        axes = np.array([3.3, 2.1, 1.2])
        axis_var = np.array([0.4, 0.3, 0.2])
        source_var = np.array([0.25, 0.12, 0.18])
        center_var = np.array([0.02, 0.04, 0.01])
        sensor_var = np.array([0.2, 0.1, 0.1])
        tracker = make_tracker(
            orientation=np.eye(3),
            kinematic_state=np.zeros(3),
            semi_axes=axes,
            covariance=center_var,
            shape_covariance=np.r_[np.zeros(3), axis_var],
            multiplicative_noise_cov=source_var,
            n_particles=1,
        )
        point = np.array([2.1, 0.01, -1.5])
        additive = center_var + sensor_var
        mean_sq = (axes**2 + axis_var) * source_var + additive
        var_axis_sq = 4 * axes**2 * axis_var + 2 * axis_var**2
        var_sq = 2 * mean_sq**2 + 3 * source_var**2 * var_axis_sq
        cross = 2 * axes * axis_var * source_var
        tracker.update(point, meas_noise_cov=sensor_var)
        npt.assert_allclose(
            tracker.axis[0], axes + cross / var_sq * (point**2 - mean_sq)
        )
        npt.assert_allclose(
            tracker.axis_covariances[0], np.diag(axis_var - cross**2 / var_sq)
        )
        self.assertEqual(tracker.effective_sample_size, 1)

    def test_seeded_resampling_copies_complete_particles_and_retries_repeat(self):
        first = make_tracker()
        second = make_tracker()
        npt.assert_array_equal(first.rotations, second.rotations)
        # A concentrated posterior must copy all three fields from the same ancestor.
        for tracker in (first, second):
            tracker._log_weights[:] = -np.inf
            tracker._log_weights[7] = 0.0
            tracker.axis_covariances[7] *= 2
        selected = [
            array[7].copy()
            for array in (first.rotations, first.axis, first.axis_covariances)
        ]
        for tracker in (first, second):
            tracker.resample()
            for array, value in zip(
                (tracker.rotations, tracker.axis, tracker.axis_covariances), selected
            ):
                npt.assert_array_equal(array, np.broadcast_to(value, array.shape))
            npt.assert_allclose(tracker.weights, 1 / tracker.n_particles)
        for tracker in (first, second):
            tracker.predict(np.eye(3), shape_sys_noise=0.002)
            tracker.update([1.0, -0.2, 0.3])
        npt.assert_array_equal(first.rotations, second.rotations)
        npt.assert_array_equal(first.axis, second.axis)
        npt.assert_array_equal(first.weights, second.weights)

    def test_geometry_averages_physical_extents_and_contour_is_on_surface(self):
        h = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
        tracker = make_tracker(
            kinematic_state=[0.3, -1, 0.2, 0.8],
            covariance=0.2,
            measurement_matrix=h,
            n_particles=2,
        )
        # Two equivalent proper principal frames differ by a pair of sign flips.
        tracker.rotations[1] = tracker.rotations[0] @ np.diag([-1, 1, -1])
        axes = tracker.axis[0]
        estimate = tracker.rotations[0] @ np.diag(axes**2) @ tracker.rotations[0].T
        npt.assert_allclose(tracker.extent, estimate, atol=1e-12)
        frame, estimated_axes = tracker.get_point_estimate_shape()
        npt.assert_allclose(frame @ np.diag(estimated_axes**2) @ frame.T, estimate)
        self.assertAlmostEqual(np.linalg.det(frame), 1)
        npt.assert_allclose(estimated_axes, axes)
        offsets = tracker.get_contour_points(41) - h @ tracker.kinematic_state
        npt.assert_allclose(
            np.einsum("ni,ij,nj->n", offsets, np.linalg.inv(estimate), offsets),
            1,
            atol=1e-12,
        )
        self.assertIs(MEMRBPF3DTracker, MemRbpf3DTracker)

    def test_history_empty_scans_and_invalid_operations_are_atomic(self):
        tracker = make_tracker(log_prior_estimates=True, log_posterior_extents=True)
        clean = make_tracker()
        for points in ([], np.empty((3, 0)), np.empty((0, 3))):
            tracker.update(points)
        with self.assertRaises(ValueError):
            tracker.predict(
                np.eye(3), rotation_increment=[0, np.nan, 0], shape_sys_noise=0.01
            )
        with self.assertRaises(ValueError):
            tracker.update(np.ones((7, 3)), meas_noise_cov=-np.eye(3))
        # Failure on the second transformed observation must discard partial updates.
        from pyrecest.filters import mem_rbpf_3d_tracker as module

        original = module.axis_squared_moments

        def fail_after_centroid(*args):
            if not np.array_equal(args[0], clean.axis):
                raise ValueError("simulated moment failure")
            return original(*args)

        with patch.object(
            module, "axis_squared_moments", side_effect=fail_after_centroid
        ):
            with self.assertRaisesRegex(ValueError, "simulated moment failure"):
                tracker.update(np.ones((3, 5)))
        for prop in (
            "kinematic_state",
            "covariance",
            "rotations",
            "axis",
            "axis_covariances",
            "weights",
        ):
            npt.assert_array_equal(getattr(tracker, prop), getattr(clean, prop))
        tracker.predict(np.eye(3), shape_sys_noise=0.02)
        clean.predict(np.eye(3), shape_sys_noise=0.02)
        npt.assert_array_equal(tracker.rotations, clean.rotations)
        self.assertEqual(tracker.prior_estimates_over_time.shape, (12, 1))
        square = np.array([[1, 2, 3], [0, -1, 2], [3, 1, -1]])
        tracker.update(square)
        clean.update(square)
        npt.assert_array_equal(tracker.weights, clean.weights)
        npt.assert_allclose(
            tracker.posterior_extents_over_time.reshape(3, 3), tracker.extent
        )

    def test_long_scan_has_stable_weights_and_triggers_ess_resampling(self):
        points = np.random.default_rng(6).normal(size=(3, 200)) * np.array(
            [[3], [1.5], [0.3]]
        )
        unchanged = make_tracker(shape_covariance=[0.2] * 3 + [0] * 3)
        resampled = make_tracker(
            shape_covariance=[0.2] * 3 + [0] * 3, resampling_threshold=40
        )
        unchanged.update(points)
        resampled.update(points)
        self.assertTrue(np.all(np.isfinite(unchanged.weights)))
        self.assertAlmostEqual(unchanged.weights.sum(), 1)
        self.assertLess(unchanged.effective_sample_size, unchanged.n_particles / 2)
        npt.assert_allclose(resampled.weights, 1 / resampled.n_particles)
        self.assertLess(
            len(np.unique(resampled.rotations, axis=0)), resampled.n_particles
        )

    def test_underflowed_particle_can_recover_on_later_evidence(self):
        tracker = make_tracker(
            n_particles=2,
            kinematic_state=np.zeros(3),
            covariance=0,
            shape_covariance=0,
            semi_axes=[4, 1, 2],
            orientation=np.eye(3),
        )
        tracker.rotations[1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        tracker.update([1000, 0, 0])
        npt.assert_array_equal(tracker.weights, [1, 0])
        tracker.update([0, 1000, 0])
        npt.assert_allclose(tracker.weights, [0.5, 0.5], atol=1e-8)

    def test_seeded_rotated_ellipsoid_converges(self):
        rng = np.random.default_rng(230)
        true_rotation = Rotation.from_rotvec([0.65, -0.4, 0.35]).as_matrix()
        true_extent = (
            true_rotation @ np.diag([4.0**2, 2.2**2, 1.1**2]) @ true_rotation.T
        )
        center = np.array([1.0, -0.6, 0.3])
        tracker = make_tracker(
            kinematic_state=[1.4, -0.8, 0.2],
            orientation=true_rotation
            @ Rotation.from_rotvec([0.17, -0.23, 0.15]).as_matrix(),
            semi_axes=[3.4, 1.9, 1.4],
            n_particles=120,
            resampling_threshold=60,
            shape_covariance=[0.06, 0.07, 0.05, 0.3, 0.2, 0.1],
        )
        initial = np.linalg.norm(tracker.extent - true_extent)
        for _ in range(25):
            tracker.predict(np.eye(3), shape_sys_noise=[0.0006] * 3 + [0.0001] * 3)
            tracker.update(
                rng.multivariate_normal(
                    center, true_extent / 5 + 0.01 * np.eye(3), size=30
                ),
                meas_noise_cov=0.01,
            )
        self.assertLess(np.linalg.norm(tracker.extent - true_extent), 0.5 * initial)
        self.assertLess(np.linalg.norm(tracker.kinematic_state - center), 0.2)
        self.assertTrue(np.all(tracker.axis > 0))
        self.assertTrue(np.all(np.linalg.eigvalsh(tracker.axis_covariances) >= -1e-12))
        npt.assert_allclose(np.linalg.det(tracker.rotations), 1, atol=1e-12)

    def test_constructor_validation(self):
        for name, value in [
            ("n_particles", 0),
            ("n_particles", True),
            ("n_particles", 2.1),
            ("resampling_threshold", np.nan),
            ("resampling_threshold", 41),
            ("minimum_axis_length", -1),
            ("orientation", -np.eye(3)),
            ("shape_covariance", -np.eye(6)),
            ("kinematic_state", [0, 1j, 2]),
            ("semi_axes", [1, -1, 2]),
            ("multiplicative_noise_cov", np.diag([0, 1, 2])),
        ]:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    make_tracker(**{name: value})


@unittest.skipIf(backend.__backend_name__ == "numpy", "Requires optional backend")
class TestMEMRBPF3DBackend(unittest.TestCase):
    def test_unsupported_backend(self):
        with self.assertRaisesRegex(NotImplementedError, "NumPy"):
            make_tracker()
