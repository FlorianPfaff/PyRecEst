from __future__ import annotations

from typing import Any

# pylint: disable=no-name-in-module,no-member,too-many-arguments,too-many-locals,protected-access
import numpy as np

from pyrecest.backend import array, diag, linalg, mean, zeros

from .mem_qkf_tracker import MEMQKFTracker


class IteratedBatchMEMQKFTracker(MEMQKFTracker):
    """Iterated batch-extent MEM-QKF tracker for 2-D elliptical objects.

    ``MEMQKFTracker(update_mode="batch")`` performs a single centroid-based
    kinematic update, but its extent update remains sequential in the individual
    detections.  This subclass keeps the same centroid kinematic update and
    replaces the scan extent step by batch quadratic pseudo-measurement updates
    for orientation and semi-axis lengths.  The batch equations can be
    re-linearized several times without re-counting the scan: every iteration
    recomputes a posterior from the same prior and the current linearization
    point.

    For scans with only one detection, the default behavior is to delegate to
    :class:`MEMQKFTracker`, because a single point does not provide useful batch
    scatter statistics.

    Parameters added by this subclass
    ---------------------------------
    n_iterations : int, default=3
        Number of fixed-point re-linearization passes used for the batch extent
        update.  ``1`` gives a non-iterated batch update.
    damping : float, default=1.0
        Damping factor applied to each fixed-point posterior relative to the
        scan prior.  Values in ``(0, 1]`` are accepted.
    single_measurement_fallback : bool, default=True
        If true, one-point scans use the ordinary sequential MEM-QKF update.
    use_centroid_residuals : bool, default=True
        If true, extent residuals for multi-point scans are centered at the scan
        centroid after the kinematic centroid update.  If false, they are
        centered at ``H @ kinematic_state`` and include kinematic covariance in
        the additive shape covariance.
    minimum_batch_covariance_eigenvalue : float, default=1e-9
        Eigenvalue floor used by the batch pseudo-measurement covariances.
    """

    kinematic_state: Any
    covariance: Any
    shape_state: Any
    shape_covariance: Any

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        *args,
        n_iterations=3,
        damping=1.0,
        single_measurement_fallback=True,
        use_centroid_residuals=True,
        minimum_batch_covariance_eigenvalue=1e-9,
        **kwargs,
    ):
        kwargs.pop("update_mode", None)
        super().__init__(*args, update_mode="sequential", **kwargs)

        self.n_iterations = int(n_iterations)
        if self.n_iterations < 1:
            raise ValueError("n_iterations must be at least 1")

        self.damping = float(damping)
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")

        self.single_measurement_fallback = bool(single_measurement_fallback)
        self.use_centroid_residuals = bool(use_centroid_residuals)
        self.minimum_batch_covariance_eigenvalue = float(
            minimum_batch_covariance_eigenvalue
        )
        if self.minimum_batch_covariance_eigenvalue < 0.0:
            raise ValueError("minimum_batch_covariance_eigenvalue must be non-negative")

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        """Update from one or more target-originated 2-D points."""
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return

        if self.single_measurement_fallback and measurements.shape[1] == 1:
            super().update(
                measurements,
                meas_mat=meas_mat,
                meas_noise_cov=meas_noise_cov,
                multiplicative_noise_cov=multiplicative_noise_cov,
            )
            return

        measurement_matrix = self._get_measurement_matrix(meas_mat)
        meas_noise_cov = self._get_update_measurement_noise(meas_noise_cov)
        multiplicative_noise_cov = self._get_multiplicative_noise_cov(
            multiplicative_noise_cov
        )

        self._batch_kinematic_update(
            measurements,
            measurement_matrix,
            meas_noise_cov,
            multiplicative_noise_cov,
        )

        center_estimate, additive_shape_covariance = (
            self._batch_extent_center_and_additive_covariance(
                measurements,
                measurement_matrix,
            )
        )
        shifted_measurements = measurements - center_estimate.reshape(2, 1)
        shape_measurement_covariance = self._regularize_batch_covariance(
            meas_noise_cov + additive_shape_covariance
        )

        prior_orientation = self.shape_state[0]
        prior_orientation_variance = self._regularize_batch_variance(
            self.orientation_variance
        )
        prior_axes = self.shape_state[1:]
        prior_axis_covariance = self._regularize_batch_covariance(self.axis_covariance)

        orientation_linearization = prior_orientation
        axes_linearization = prior_axes
        posterior_orientation = prior_orientation
        posterior_orientation_variance = prior_orientation_variance
        posterior_axes = prior_axes
        posterior_axis_covariance = prior_axis_covariance

        for _ in range(self.n_iterations):
            full_orientation, posterior_orientation_variance = (
                self._batch_orientation_posterior(
                    shifted_measurements=shifted_measurements,
                    prior_orientation=prior_orientation,
                    prior_orientation_variance=prior_orientation_variance,
                    orientation_linearization=orientation_linearization,
                    axes_linearization=axes_linearization,
                    multiplicative_noise_cov=multiplicative_noise_cov,
                    shape_measurement_covariance=shape_measurement_covariance,
                )
            )
            posterior_orientation = prior_orientation + self.damping * self._axial_delta(
                prior_orientation,
                full_orientation,
            )

            full_axes, posterior_axis_covariance = self._batch_axis_posterior(
                shifted_measurements=shifted_measurements,
                prior_axes=prior_axes,
                prior_axis_covariance=prior_axis_covariance,
                axes_linearization=axes_linearization,
                orientation_linearization=posterior_orientation,
                multiplicative_noise_cov=multiplicative_noise_cov,
                shape_measurement_covariance=shape_measurement_covariance,
            )
            posterior_axes = prior_axes + self.damping * (full_axes - prior_axes)
            posterior_axes, posterior_axis_covariance = (
                self._canonicalize_axes_and_axis_covariance(
                    posterior_axes,
                    posterior_axis_covariance,
                )
            )
            posterior_axis_covariance = self._regularize_batch_covariance(
                posterior_axis_covariance
            )
            orientation_linearization = posterior_orientation
            axes_linearization = posterior_axes

        self.shape_state = array(
            [posterior_orientation, posterior_axes[0], posterior_axes[1]]
        )
        self.shape_covariance = self._decoupled_shape_covariance(
            self._regularize_batch_variance(posterior_orientation_variance),
            posterior_axis_covariance,
        )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _batch_extent_center_and_additive_covariance(
        self,
        measurements,
        measurement_matrix,
    ):
        if measurements.shape[1] == 1:
            center_estimate = measurement_matrix @ self.kinematic_state
            additive_shape_covariance = (
                measurement_matrix @ self.covariance @ measurement_matrix.T
                + self.axis_covariance
            )
            return center_estimate, additive_shape_covariance

        if self.use_centroid_residuals:
            return mean(measurements, axis=1), zeros((2, 2))

        center_estimate = measurement_matrix @ self.kinematic_state
        additive_shape_covariance = measurement_matrix @ self.covariance @ measurement_matrix.T
        return center_estimate, additive_shape_covariance

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _batch_orientation_posterior(
        self,
        *,
        shifted_measurements,
        prior_orientation,
        prior_orientation_variance,
        orientation_linearization,
        axes_linearization,
        multiplicative_noise_cov,
        shape_measurement_covariance,
    ):
        n_measurements = shifted_measurements.shape[1]
        semi_axis_1, semi_axis_2 = axes_linearization
        orientation_variance = self._regularize_batch_variance(
            prior_orientation_variance
        )

        extent_transform = self._rotation(orientation_linearization) @ diag(
            array([semi_axis_1, semi_axis_2])
        )
        first_extent_row = extent_transform[0, :]
        second_extent_row = extent_transform[1, :]
        first_orientation_jacobian = array(
            [
                -semi_axis_1 * np.sin(orientation_linearization),
                -semi_axis_2 * np.cos(orientation_linearization),
            ]
        )
        second_orientation_jacobian = array(
            [
                semi_axis_1 * np.cos(orientation_linearization),
                -semi_axis_2 * np.sin(orientation_linearization),
            ]
        )

        extent_covariance = extent_transform @ multiplicative_noise_cov @ extent_transform.T
        orientation_covariance = orientation_variance * array(
            [
                [
                    first_orientation_jacobian
                    @ multiplicative_noise_cov
                    @ first_orientation_jacobian,
                    first_orientation_jacobian
                    @ multiplicative_noise_cov
                    @ second_orientation_jacobian,
                ],
                [
                    second_orientation_jacobian
                    @ multiplicative_noise_cov
                    @ first_orientation_jacobian,
                    second_orientation_jacobian
                    @ multiplicative_noise_cov
                    @ second_orientation_jacobian,
                ],
            ]
        )
        quadratic_covariance = self._regularize_batch_covariance(
            shape_measurement_covariance + extent_covariance + orientation_covariance
        )

        pseudo_measurement = array(
            [
                mean(shifted_measurements[0, :] ** 2),
                mean(shifted_measurements[1, :] ** 2),
                mean(shifted_measurements[0, :] * shifted_measurements[1, :]),
            ]
        )
        expected_pseudo_measurement = array(
            [
                quadratic_covariance[0, 0],
                quadratic_covariance[1, 1],
                quadratic_covariance[0, 1],
            ]
        )
        pseudo_covariance = self._regularize_batch_covariance(
            self._quadratic_pseudo_covariance(quadratic_covariance) / n_measurements
        )

        orientation_sensitivity = array(
            [
                2.0
                * first_extent_row
                @ multiplicative_noise_cov
                @ first_orientation_jacobian,
                2.0
                * second_extent_row
                @ multiplicative_noise_cov
                @ second_orientation_jacobian,
                first_extent_row
                @ multiplicative_noise_cov
                @ second_orientation_jacobian
                + second_extent_row
                @ multiplicative_noise_cov
                @ first_orientation_jacobian,
            ]
        )
        orientation_cross_covariance = (
            orientation_variance * orientation_sensitivity.reshape(1, 3)
        )
        orientation_gain = self._gain_from_cross_covariance(
            orientation_cross_covariance,
            pseudo_covariance,
        )
        prior_minus_linearization = self._axial_delta(
            orientation_linearization,
            prior_orientation,
        )
        innovation = (
            pseudo_measurement
            - expected_pseudo_measurement
            - orientation_sensitivity * prior_minus_linearization
        )
        posterior_orientation = prior_orientation + (orientation_gain @ innovation)[0]
        posterior_orientation = prior_orientation + self._axial_delta(
            prior_orientation,
            posterior_orientation,
        )
        posterior_orientation_variance = (
            orientation_variance
            - (orientation_gain @ orientation_cross_covariance.T)[0, 0]
        )
        return posterior_orientation, posterior_orientation_variance

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _batch_axis_posterior(
        self,
        *,
        shifted_measurements,
        prior_axes,
        prior_axis_covariance,
        axes_linearization,
        orientation_linearization,
        multiplicative_noise_cov,
        shape_measurement_covariance,
    ):
        n_measurements = shifted_measurements.shape[1]
        semi_axes = axes_linearization
        axis_covariance = self._regularize_batch_covariance(prior_axis_covariance)

        rotation_to_axis_frame = self._rotation(-orientation_linearization)
        rotated_measurement_covariance = self._regularize_batch_covariance(
            rotation_to_axis_frame
            @ shape_measurement_covariance
            @ rotation_to_axis_frame.T
        )
        rotated_measurements = rotation_to_axis_frame @ shifted_measurements
        pseudo_measurement = array(
            [
                mean(rotated_measurements[0, :] ** 2),
                mean(rotated_measurements[1, :] ** 2),
            ]
        )

        multiplicative_variance_1 = multiplicative_noise_cov[0, 0]
        multiplicative_variance_2 = multiplicative_noise_cov[1, 1]
        expected_pseudo_measurement = array(
            [
                rotated_measurement_covariance[0, 0]
                + multiplicative_variance_1
                * (axis_covariance[0, 0] + semi_axes[0] ** 2),
                rotated_measurement_covariance[1, 1]
                + multiplicative_variance_2
                * (axis_covariance[1, 1] + semi_axes[1] ** 2),
            ]
        )
        pseudo_covariance = self._regularize_batch_covariance(
            array(
                [
                    [
                        2.0 * expected_pseudo_measurement[0] ** 2,
                        2.0 * rotated_measurement_covariance[0, 1] ** 2,
                    ],
                    [
                        2.0 * rotated_measurement_covariance[1, 0] ** 2,
                        2.0 * expected_pseudo_measurement[1] ** 2,
                    ],
                ]
            )
            / n_measurements
        )
        axis_cross_covariance = array(
            [
                [
                    2.0
                    * multiplicative_variance_1
                    * semi_axes[0]
                    * axis_covariance[0, 0],
                    0.0,
                ],
                [
                    0.0,
                    2.0
                    * multiplicative_variance_2
                    * semi_axes[1]
                    * axis_covariance[1, 1],
                ],
            ]
        )
        axis_sensitivity = diag(
            array(
                [
                    2.0 * multiplicative_variance_1 * semi_axes[0],
                    2.0 * multiplicative_variance_2 * semi_axes[1],
                ]
            )
        )
        axis_gain = self._gain_from_cross_covariance(
            axis_cross_covariance,
            pseudo_covariance,
        )
        innovation = (
            pseudo_measurement
            - expected_pseudo_measurement
            - axis_sensitivity @ (prior_axes - semi_axes)
        )
        posterior_axes = prior_axes + axis_gain @ innovation
        posterior_axis_covariance = axis_covariance - axis_gain @ axis_cross_covariance.T
        return posterior_axes, self._symmetrize(posterior_axis_covariance)

    @staticmethod
    def _quadratic_pseudo_covariance(covariance):
        covariance = array(covariance)
        c_xx = covariance[0, 0]
        c_xy = covariance[0, 1]
        c_yy = covariance[1, 1]
        return array(
            [
                [2.0 * c_xx**2, 2.0 * c_xy**2, 2.0 * c_xx * c_xy],
                [2.0 * c_xy**2, 2.0 * c_yy**2, 2.0 * c_yy * c_xy],
                [
                    2.0 * c_xx * c_xy,
                    2.0 * c_yy * c_xy,
                    c_xx * c_yy + c_xy**2,
                ],
            ]
        )

    def _regularize_batch_variance(self, variance):
        return max(float(variance), self.minimum_batch_covariance_eigenvalue)

    def _regularize_batch_covariance(self, covariance):
        covariance = self._symmetrize(array(covariance))
        eigenvalues, eigenvectors = linalg.eigh(covariance)
        eigenvalues = np.maximum(
            np.asarray(eigenvalues),
            self.minimum_batch_covariance_eigenvalue,
        )
        return self._symmetrize((eigenvectors * array(eigenvalues)) @ eigenvectors.T)

    @staticmethod
    def _axial_delta(reference, angle):
        return ((angle - reference + np.pi / 2.0) % np.pi) - np.pi / 2.0


IteratedBatchMemQkfTracker = IteratedBatchMEMQKFTracker
