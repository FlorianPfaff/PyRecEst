import itertools
import unittest

import numpy as np
import numpy.testing as npt
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

# pylint: disable=no-name-in-module,no-member,protected-access,duplicate-code
from pyrecest.backend import array as backend_array
from pyrecest.backend import to_numpy
from pyrecest.filters import MEMEKFStar3DTracker


class TestMEMEKFStar3DTracker(unittest.TestCase):
    PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def setUp(self):
        self.orientation = Rotation.from_rotvec([0.43, -0.61, 0.27]).as_matrix()
        self.axes = np.array([4.7, 2.3, 1.1])
        self.shape_covariance = np.diag([0.025, 0.018, 0.032, 0.16, 0.09, 0.04])
        self.measurement_noise = np.array(
            [[0.06, 0.012, -0.005], [0.012, 0.09, 0.018], [-0.005, 0.018, 0.04]]
        )

    def make_tracker(self, **overrides):
        parameters = {
            "kinematic_state": np.array([0.4, -0.2, 0.3, 0.1, -0.3, 0.2]),
            "covariance": np.diag([0.3, 0.2, 0.25, 0.04, 0.03, 0.05]),
            "orientation": self.orientation,
            "semi_axes": self.axes,
            "shape_covariance": self.shape_covariance,
        }
        parameters.update(overrides)
        return MEMEKFStar3DTracker(**parameters)

    @classmethod
    def quadratic_products(cls, points):
        return np.array([points[..., i] * points[..., j] for i, j in cls.PAIRS]).T

    def test_body_extent_jacobians_match_independent_rotation_differences(self):
        tracker = self.make_tracker()
        analytical = tracker._extent_row_jacobians()
        numerical = np.empty((3, 3, 6))
        step = 1e-6
        for coordinate in range(6):
            delta = step * np.eye(6)[coordinate]
            positive = (
                self.orientation
                @ Rotation.from_rotvec(delta[:3]).as_matrix()
                @ np.diag(self.axes + delta[3:])
            )
            negative = (
                self.orientation
                @ Rotation.from_rotvec(-delta[:3]).as_matrix()
                @ np.diag(self.axes - delta[3:])
            )
            numerical[:, :, coordinate] = (positive - negative) / (2.0 * step)
        npt.assert_allclose(analytical, numerical, rtol=1e-8, atol=1e-9)

    def test_shape_pseudo_jacobian_matches_expected_scatter_differences(self):
        tracker = self.make_tracker()
        multiplicative_covariance = np.diag([0.18, 0.23, 0.27])
        numerical = np.empty((6, 6))
        step = 1e-6
        for coordinate in range(6):
            delta = step * np.eye(6)[coordinate]
            products = []
            for sign in (1.0, -1.0):
                extent = (
                    self.orientation
                    @ Rotation.from_rotvec(sign * delta[:3]).as_matrix()
                    @ np.diag(self.axes + sign * delta[3:])
                )
                scatter = extent @ multiplicative_covariance @ extent.T
                products.append(np.array([scatter[i, j] for i, j in self.PAIRS]))
            numerical[:, coordinate] = (products[0] - products[1]) / (2.0 * step)
        npt.assert_allclose(
            tracker._shape_pseudo_jacobian(backend_array(multiplicative_covariance)),
            numerical,
            rtol=1e-8,
            atol=1e-9,
        )

    def test_shape_noise_correction_matches_independent_joint_sigma_points(self):
        generator = np.random.default_rng(301)
        factor = generator.normal(size=(6, 6))
        shape_covariance = 0.003 * factor @ factor.T
        tracker = self.make_tracker(shape_covariance=shape_covariance)
        noise_covariance = np.diag([0.13, 0.2, 0.29])
        shape_factor = np.linalg.cholesky(shape_covariance)
        noise_factor = np.linalg.cholesky(noise_covariance)
        shape_points = np.concatenate(
            (np.sqrt(6.0) * shape_factor.T, -np.sqrt(6.0) * shape_factor.T)
        )
        noise_points = np.concatenate(
            (np.sqrt(3.0) * noise_factor.T, -np.sqrt(3.0) * noise_factor.T)
        )
        correction = np.zeros((3, 3))
        step = 1e-5
        for delta in shape_points:
            positive = (
                self.orientation
                @ Rotation.from_rotvec(step * delta[:3]).as_matrix()
                @ np.diag(self.axes + step * delta[3:])
            )
            negative = (
                self.orientation
                @ Rotation.from_rotvec(-step * delta[:3]).as_matrix()
                @ np.diag(self.axes - step * delta[3:])
            )
            extent_delta = (positive - negative) / (2.0 * step)
            for noise in noise_points:
                residual = extent_delta @ noise
                correction += np.outer(residual, residual)
        correction /= len(shape_points) * len(noise_points)
        npt.assert_allclose(
            tracker._shape_noise_covariance(backend_array(noise_covariance)),
            correction,
            rtol=1e-8,
            atol=1e-10,
        )

    def test_quadratic_covariance_matches_exact_gaussian_quadrature(self):
        covariance = np.array([[2.0, 0.4, -0.3], [0.4, 1.3, 0.2], [-0.3, 0.2, 0.8]])
        factor = np.linalg.cholesky(covariance)
        nodes = (-np.sqrt(3.0), 0.0, np.sqrt(3.0))
        weights = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
        expected = np.zeros(6)
        raw_second_moment = np.zeros((6, 6))
        for indices in itertools.product(range(3), repeat=3):
            point = factor @ np.array([nodes[index] for index in indices])
            weight = np.prod([weights[index] for index in indices])
            products = self.quadratic_products(point)
            expected += weight * products
            raw_second_moment += weight * np.outer(products, products)
        centered_covariance = raw_second_moment - np.outer(expected, expected)
        npt.assert_allclose(
            MEMEKFStar3DTracker._pseudo_measurement_covariance(covariance),
            centered_covariance,
            rtol=1e-13,
            atol=1e-13,
        )

    def test_right_reset_jacobian_matches_independent_group_differences(self):
        for correction in (np.array([0.42, -0.61, 0.35]), np.zeros(3)):
            inverse = Rotation.from_rotvec(-correction)
            numerical = np.empty((3, 3))
            step = 1e-6
            for coordinate in range(3):
                delta = step * np.eye(3)[coordinate]
                positive = (
                    inverse * Rotation.from_rotvec(correction + delta)
                ).as_rotvec()
                negative = (
                    inverse * Rotation.from_rotvec(correction - delta)
                ).as_rotvec()
                numerical[:, coordinate] = (positive - negative) / (2.0 * step)
            npt.assert_allclose(
                MEMEKFStar3DTracker._right_jacobian(correction),
                numerical,
                rtol=1e-8,
                atol=1e-9,
            )

    def test_update_retracts_rotation_and_resets_rotation_axis_cross_covariance(self):
        generator = np.random.default_rng(17)
        factor = generator.normal(size=(6, 6))
        prior_covariance = 0.004 * factor @ factor.T
        tracker = self.make_tracker(shape_covariance=prior_covariance)
        measurement = np.array([2.8, -0.9, 0.6])
        innovation = measurement - to_numpy(tracker.kinematic_state[:3])
        extent = to_numpy(tracker._extent_transform())
        multiplicative_covariance = 0.2 * np.eye(3)
        covariance = (
            to_numpy(tracker.covariance[:3, :3])
            + extent @ multiplicative_covariance @ extent.T
            + to_numpy(
                tracker._shape_noise_covariance(
                    backend_array(multiplicative_covariance)
                )
            )
            + self.measurement_noise
        )
        pseudo_covariance = to_numpy(tracker._pseudo_measurement_covariance(covariance))
        derivative = to_numpy(
            tracker._shape_pseudo_jacobian(backend_array(multiplicative_covariance))
        )
        gain = np.linalg.solve(pseudo_covariance, (prior_covariance @ derivative.T).T).T
        pseudo_mean = np.array([covariance[i, j] for i, j in self.PAIRS])
        correction = gain @ (self.quadratic_products(innovation) - pseudo_mean)
        before_reset = prior_covariance - gain @ pseudo_covariance @ gain.T

        # Independently differentiate the actual SO(3) change of reference.
        inverse = Rotation.from_rotvec(-correction[:3])
        reset_rotation = np.empty((3, 3))
        step = 1e-6
        for coordinate in range(3):
            delta = step * np.eye(3)[coordinate]
            positive = (
                inverse * Rotation.from_rotvec(correction[:3] + delta)
            ).as_rotvec()
            negative = (
                inverse * Rotation.from_rotvec(correction[:3] - delta)
            ).as_rotvec()
            reset_rotation[:, coordinate] = (positive - negative) / (2.0 * step)
        reset = block_diag(reset_rotation, np.eye(3))
        expected_covariance = reset @ before_reset @ reset.T
        self.assertGreater(np.linalg.norm(expected_covariance[:3, 3:]), 1e-4)
        self.assertGreater(np.linalg.norm(expected_covariance - before_reset), 1e-5)
        tracker.update(measurement, meas_noise_cov=self.measurement_noise)
        npt.assert_allclose(
            tracker.orientation,
            self.orientation @ Rotation.from_rotvec(correction[:3]).as_matrix(),
            atol=1e-12,
        )
        npt.assert_allclose(tracker.semi_axes, self.axes + correction[3:], atol=1e-12)
        npt.assert_allclose(tracker.shape_covariance, expected_covariance, atol=1e-10)

    def test_sphere_and_repeated_axes_have_no_unobservable_rotation_signal(self):
        sphere = self.make_tracker(semi_axes=np.array([2.0, 2.0, 2.0]))
        npt.assert_allclose(
            sphere._shape_pseudo_jacobian(backend_array(0.2 * np.eye(3)))[:, :3],
            np.zeros((6, 3)),
            atol=1e-14,
        )
        spheroid = self.make_tracker(semi_axes=np.array([4.0, 2.0, 2.0]))
        jacobian = to_numpy(
            spheroid._shape_pseudo_jacobian(backend_array(0.2 * np.eye(3)))
        )
        npt.assert_allclose(jacobian[:, 0], np.zeros(6), atol=1e-14)
        self.assertGreater(np.linalg.norm(jacobian[:, 1:3]), 1.0)

    def test_updates_are_equivariant_under_global_rotations(self):
        global_rotation = Rotation.from_rotvec([-0.53, 0.32, 0.47]).as_matrix()
        state_rotation = block_diag(global_rotation, global_rotation)
        generator = np.random.default_rng(420)
        factor = generator.normal(size=(6, 6))
        shape_covariance = 0.002 * factor @ factor.T
        tracker = self.make_tracker(shape_covariance=shape_covariance)
        rotated = self.make_tracker(
            kinematic_state=state_rotation @ to_numpy(tracker.kinematic_state),
            covariance=state_rotation @ to_numpy(tracker.covariance) @ state_rotation.T,
            orientation=global_rotation @ self.orientation,
            shape_covariance=shape_covariance,
        )
        measurements = np.array(
            [[1.4, -0.4, 0.8], [-0.3, 0.6, -0.2], [0.7, 1.2, 0.4], [-0.9, -0.2, 0.5]]
        )
        tracker.update(measurements, meas_noise_cov=self.measurement_noise)
        rotated.update(
            measurements @ global_rotation.T,
            meas_noise_cov=global_rotation @ self.measurement_noise @ global_rotation.T,
        )
        npt.assert_allclose(
            rotated.kinematic_state,
            state_rotation @ to_numpy(tracker.kinematic_state),
            atol=1e-10,
        )
        npt.assert_allclose(
            rotated.covariance,
            state_rotation @ to_numpy(tracker.covariance) @ state_rotation.T,
            atol=1e-10,
        )
        npt.assert_allclose(
            rotated.orientation,
            global_rotation @ to_numpy(tracker.orientation),
            atol=1e-10,
        )
        npt.assert_allclose(rotated.semi_axes, tracker.semi_axes, atol=1e-10)
        npt.assert_allclose(
            rotated.shape_covariance, tracker.shape_covariance, atol=1e-10
        )

    def test_recovers_fully_rotated_ellipsoid_and_preserves_valid_covariances(self):
        generator = np.random.default_rng(20261011)
        true_orientation = Rotation.from_rotvec([0.7, -0.45, 0.5]).as_matrix()
        true_axes = np.array([4.0, 2.5, 1.2])
        true_center = np.array([1.2, -0.7, 0.4])
        true_extent = true_orientation @ np.diag(true_axes**2) @ true_orientation.T
        initial_orientation = (
            true_orientation @ Rotation.from_rotvec([0.3, -0.2, 0.18]).as_matrix()
        )
        tracker = self.make_tracker(
            kinematic_state=np.concatenate(
                (true_center + [0.2, 0.1, -0.1], np.zeros(3))
            ),
            covariance=np.diag([0.5, 0.5, 0.5, 0.01, 0.01, 0.01]),
            orientation=initial_orientation,
            semi_axes=np.array([3.6, 2.1, 1.6]),
            shape_covariance=np.diag([0.09, 0.09, 0.09, 0.25, 0.25, 0.16]),
        )
        initial_error = np.linalg.norm(
            to_numpy(tracker.get_point_estimate_extent()) - true_extent
        )
        source_covariance = 0.2 * true_extent + 0.04 * np.eye(3)
        for _ in range(40):
            measurements = generator.multivariate_normal(
                true_center, source_covariance, size=25
            )
            tracker.update(measurements, meas_noise_cov=0.04 * np.eye(3))
            npt.assert_allclose(
                tracker.orientation.T @ tracker.orientation, np.eye(3), atol=1e-12
            )
            self.assertAlmostEqual(
                np.linalg.det(to_numpy(tracker.orientation)), 1.0, places=12
            )
            self.assertGreater(np.min(to_numpy(tracker.semi_axes)), 0.0)
            self.assertGreaterEqual(
                np.min(np.linalg.eigvalsh(to_numpy(tracker.covariance))), -1e-12
            )
            self.assertGreaterEqual(
                np.min(np.linalg.eigvalsh(to_numpy(tracker.shape_covariance))), -1e-12
            )
        extent_error = np.linalg.norm(
            to_numpy(tracker.get_point_estimate_extent()) - true_extent
        )
        self.assertLess(extent_error, 0.4 * initial_error)
        self.assertLess(extent_error / np.linalg.norm(true_extent), 0.12)
        self.assertLess(
            np.linalg.norm(to_numpy(tracker.kinematic_state[:3]) - true_center), 0.15
        )
        rotation_error = Rotation.from_matrix(
            true_orientation.T @ to_numpy(tracker.orientation)
        ).magnitude()
        self.assertLess(rotation_error, 0.15)

    def test_measurement_layouts_and_empty_update(self):
        measurements = np.array(
            [[0.8, -0.2, 0.4], [0.3, 0.7, -0.1], [-0.5, 0.2, 0.8], [1.0, -0.8, 0.1]]
        )
        rows = self.make_tracker()
        columns = self.make_tracker()
        rows.update(measurements, meas_noise_cov=self.measurement_noise)
        columns.update(measurements.T, meas_noise_cov=self.measurement_noise)
        npt.assert_allclose(rows.get_point_estimate(), columns.get_point_estimate())
        before = np.array(to_numpy(rows.get_point_estimate()), copy=True)
        before_covariance = np.array(to_numpy(rows.shape_covariance), copy=True)
        rows.update(np.empty((3, 0)), meas_noise_cov=self.measurement_noise)
        npt.assert_array_equal(rows.get_point_estimate(), before)
        npt.assert_array_equal(rows.shape_covariance, before_covariance)

    def test_prediction_rotates_body_covariance_and_records_inherited_history(self):
        generator = np.random.default_rng(11)
        factor = generator.normal(size=(6, 6))
        shape_covariance = 0.003 * factor @ factor.T
        tracker = self.make_tracker(
            shape_covariance=shape_covariance,
            log_prior_estimates=True,
            log_posterior_estimates=True,
            log_prior_extents=True,
            log_posterior_extents=True,
        )
        increment = np.array([0.1, -0.2, 0.3])
        rotation = Rotation.from_rotvec(increment).as_matrix()
        transport = block_diag(rotation.T, np.eye(3))
        tracker.predict_linear(np.eye(6), rotation_increment=increment)
        npt.assert_allclose(
            tracker.orientation, self.orientation @ rotation, atol=1e-12
        )
        npt.assert_allclose(
            tracker.shape_covariance,
            transport @ shape_covariance @ transport.T,
            atol=1e-12,
        )
        tracker.update(
            np.array([0.8, -0.1, 0.2]), meas_noise_cov=self.measurement_noise
        )
        self.assertEqual(tracker.prior_estimates_over_time.shape[1], 1)
        self.assertEqual(tracker.posterior_estimates_over_time.shape[1], 1)
        self.assertEqual(tracker.prior_extents_over_time.shape[1], 1)
        self.assertEqual(tracker.posterior_extents_over_time.shape[1], 1)

    def test_rejects_invalid_orientation_axes_and_covariances(self):
        invalid_parameters = (
            {"orientation": np.eye(2)},
            {"orientation": np.diag([1.0, 1.0, -1.0])},
            {"orientation": 1.1 * np.eye(3)},
            {"orientation": np.full((3, 3), np.nan)},
            {"semi_axes": np.array([1.0, 2.0])},
            {"semi_axes": np.array([1.0, 0.0, 2.0])},
            {"semi_axes": np.array([1.0, np.nan, 2.0])},
            {"shape_covariance": np.eye(3)},
            {"shape_covariance": np.diag([0.1, 0.1, 0.1, 0.1, 0.1, -0.1])},
        )
        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                    self.make_tracker(**parameters)


if __name__ == "__main__":
    unittest.main()
