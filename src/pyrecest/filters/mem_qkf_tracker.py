from __future__ import annotations

from typing import Any

# pylint: disable=no-name-in-module,no-member,duplicate-code,too-many-locals
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import (
    array,
    cos,
    diag,
    eye,
    kron,
    linalg,
    maximum,
    mean,
    sin,
    where,
    zeros,
)

from .mem_ekf_tracker import MEMEKFTracker


class MEMQKFTracker(MEMEKFTracker):
    """Sequential decoupled MEM-QKF tracker for one 2-D elliptical object.

    This filter ports the MEM-QKF update used in the quadratic extended-object
    tracking benchmark to PyRecEst's MEM tracker convention. The state is split
    into a Euclidean kinematic state and a shape state
    ``[orientation, semi_axis_1, semi_axis_2]``. This differs from the reference
    benchmark helper that reports full ``length`` and ``width`` by doubling its
    internal semi-axis states.

    By default, the update is sequential in target-originated point
    measurements. For each point, it performs a kinematic Kalman update, a
    scalar quadratic orientation update, and a two-dimensional quadratic
    semi-axis update. With ``update_mode="batch"``, the kinematics are updated
    once from the measurement-set centroid, while the extent update remains
    sequential in the individual points. The reference MEM-QKF equations keep
    the orientation covariance separate from the two axis-length covariances.
    Consequently this implementation only retains the block-diagonal terms of
    ``shape_covariance``:
    ``shape_covariance[0, 0]`` and ``shape_covariance[1:, 1:]``. Cross-covariance
    terms between orientation and axes are ignored and reset to zero after
    initialization, prediction, and update.

    Parameters added by this subclass
    ---------------------------------
    default_meas_noise_cov : array-like, optional
        Measurement-noise covariance used by :meth:`update` when
        ``meas_noise_cov`` is omitted. This is useful when porting code from
        reference implementations that store ``R`` in the tracker instance.
    update_mode : {"sequential", "batch"}, default="sequential"
        Selects whether every measurement performs a kinematic update
        (``"sequential"``) or whether one centroid-based kinematic update is
        performed before the sequential extent update (``"batch"``).
    minimum_axis_length : float, default=1e-9
        Lower bound applied to semi-axis lengths after updates.
    minimum_covariance_eigenvalue : float, default=0.0
        Lower bound used when projecting updated covariance matrices back to the
        positive-semidefinite cone.
    """

    kinematic_state: Any
    covariance: Any
    shape_state: Any
    shape_covariance: Any

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        kinematic_state,
        covariance,
        shape_state,
        shape_covariance,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        covariance_regularization=0.0,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
        default_meas_noise_cov=None,
        update_mode="sequential",
        minimum_axis_length=1e-9,
        minimum_covariance_eigenvalue=0.0,
    ):
        super().__init__(
            kinematic_state,
            covariance,
            shape_state,
            shape_covariance,
            measurement_matrix=measurement_matrix,
            multiplicative_noise_cov=multiplicative_noise_cov,
            covariance_regularization=covariance_regularization,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.minimum_axis_length = float(minimum_axis_length)
        if self.minimum_axis_length <= 0.0:
            raise ValueError("minimum_axis_length must be positive")
        self.minimum_covariance_eigenvalue = float(minimum_covariance_eigenvalue)
        if self.minimum_covariance_eigenvalue < 0.0:
            raise ValueError("minimum_covariance_eigenvalue must be non-negative")

        self.default_meas_noise_cov = None
        self.set_default_measurement_noise_cov(default_meas_noise_cov)
        self.update_mode = self._validate_update_mode(update_mode)
        self.shape_covariance = self._decoupled_shape_covariance(
            self._regularize_variance(self.shape_covariance[0, 0]),
            self._project_symmetric_covariance(self.shape_covariance[1:, 1:]),
        )
        orientation = self.shape_state[0]
        axes, axis_covariance = self._canonicalize_axes_and_axis_covariance(
            self.shape_state[1:],
            self.axis_covariance,
        )
        self.shape_state = array([orientation, axes[0], axes[1]])
        self.shape_covariance = self._decoupled_shape_covariance(
            self.orientation_variance,
            axis_covariance,
        )

    @staticmethod
    def _validate_update_mode(update_mode):
        update_mode = str(update_mode).lower()
        if update_mode not in {"sequential", "batch"}:
            raise ValueError("update_mode must be 'sequential' or 'batch'")
        return update_mode

    @staticmethod
    def _rotation(angle):
        return array(
            [
                [cos(angle), -sin(angle)],
                [sin(angle), cos(angle)],
            ]
        )

    @staticmethod
    def _vectorize_columns(matrix):
        """Return ``vec(matrix)`` with column-major ordering."""
        return matrix.T.reshape(-1)

    @classmethod
    def _decoupled_shape_covariance(cls, orientation_variance, axis_covariance):
        axis_covariance = cls._symmetrize(array(axis_covariance))
        if axis_covariance.shape != (2, 2):
            raise ValueError("axis_covariance must have shape (2, 2)")
        return cls._symmetrize(
            linalg.block_diag(array([[orientation_variance]]), axis_covariance)
        )

    def _regularize_variance(self, variance):
        if float(variance) < self.minimum_covariance_eigenvalue:
            return array(self.minimum_covariance_eigenvalue)
        return variance

    def _project_symmetric_covariance(self, covariance):
        covariance = self._symmetrize(covariance)
        eigenvalues, eigenvectors = linalg.eigh(covariance)
        if float(eigenvalues[0]) >= self.minimum_covariance_eigenvalue:
            return covariance
        eigenvalues = maximum(eigenvalues, self.minimum_covariance_eigenvalue)
        return self._symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)

    def _canonicalize_axes_and_axis_covariance(self, axes, axis_covariance):
        axes = array(axes)
        signs = where(axes < 0.0, -1.0, 1.0)
        sign_matrix = diag(signs)
        axis_covariance = sign_matrix @ axis_covariance @ sign_matrix.T
        axes = maximum(backend_abs(axes), self.minimum_axis_length)
        return axes, self._project_symmetric_covariance(axis_covariance)

    def _regularize_covariance(self, covariance):
        covariance = self._symmetrize(covariance)
        if self.covariance_regularization > 0.0:
            covariance = covariance + self.covariance_regularization * eye(
                covariance.shape[0]
            )
        return self._project_symmetric_covariance(covariance)

    @staticmethod
    def _gain_from_cross_covariance(cross_covariance, innovation_covariance):
        """Return ``cross_covariance @ inv(innovation_covariance)`` stably."""
        return linalg.solve(innovation_covariance.T, cross_covariance.T).T

    @property
    def orientation_variance(self):
        return self.shape_covariance[0, 0]

    @property
    def axis_covariance(self):
        return self.shape_covariance[1:, 1:]

    def set_default_measurement_noise_cov(self, meas_noise_cov):
        """Set the measurement-noise covariance used by default in updates."""
        if meas_noise_cov is None:
            self.default_meas_noise_cov = None
            return
        self.default_meas_noise_cov = self._get_measurement_noise(meas_noise_cov)

    def set_R(self, meas_noise_cov):
        """Compatibility alias for setting the default measurement covariance."""
        self.set_default_measurement_noise_cov(meas_noise_cov)

    def _get_update_measurement_noise(self, meas_noise_cov):
        if meas_noise_cov is None and self.default_meas_noise_cov is not None:
            return self.default_meas_noise_cov
        return self._get_measurement_noise(meas_noise_cov)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_system_matrix=None,
        shape_sys_noise=None,
    ):
        """Predict one step and keep the MEM-QKF shape covariance decoupled."""
        super().predict_linear(
            system_matrix,
            sys_noise=sys_noise,
            inputs=inputs,
            shape_system_matrix=shape_system_matrix,
            shape_sys_noise=shape_sys_noise,
        )
        axes, axis_covariance = self._canonicalize_axes_and_axis_covariance(
            self.shape_state[1:],
            self.shape_covariance[1:, 1:],
        )
        self.shape_state = array([self.shape_state[0], axes[0], axes[1]])
        self.shape_covariance = self._decoupled_shape_covariance(
            self._regularize_variance(self.shape_covariance[0, 0]),
            axis_covariance,
        )

    def predict(self, *args, **kwargs):
        """Alias for :meth:`predict_linear` to match existing EOT tracker APIs."""
        self.predict_linear(*args, **kwargs)

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        """Sequentially update from one or more 2-D target-originated points."""
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return

        measurement_matrix = self._get_measurement_matrix(meas_mat)
        meas_noise_cov = self._get_update_measurement_noise(meas_noise_cov)
        multiplicative_noise_cov = self._get_multiplicative_noise_cov(
            multiplicative_noise_cov
        )

        if measurements.shape[1] == 1:
            center_estimate = measurement_matrix @ self.kinematic_state
            additive_measurement_noise = (
                measurement_matrix @ self.covariance @ measurement_matrix.T
                + self.axis_covariance
            )
        else:
            center_estimate = mean(measurements, axis=1)
            additive_measurement_noise = zeros(
                (self.measurement_dim, self.measurement_dim)
            )
        shape_measurement_covariance = self._symmetrize(
            meas_noise_cov + additive_measurement_noise
        )

        if self.update_mode == "batch":
            self._batch_kinematic_update(
                measurements,
                measurement_matrix,
                meas_noise_cov,
                multiplicative_noise_cov,
            )
            update_kinematics = False
        else:
            update_kinematics = True

        for measurement_index in range(measurements.shape[1]):
            self._update_single_measurement_qkf(
                measurements[:, measurement_index],
                center_estimate,
                measurement_matrix,
                meas_noise_cov,
                multiplicative_noise_cov,
                shape_measurement_covariance,
                update_kinematics=update_kinematics,
            )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def _update_single_measurement_qkf(
        self,
        measurement,
        center_estimate,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
        shape_measurement_covariance,
        update_kinematics=True,
    ):
        orientation = self.shape_state[0]
        semi_axes = self.shape_state[1:]
        orientation_variance = self.orientation_variance
        axis_covariance = self.axis_covariance

        kinematic_state = self.kinematic_state
        covariance = self.covariance
        if update_kinematics:
            kinematic_state, covariance = self._kinematic_update(
                measurement,
                measurement_matrix,
                meas_noise_cov,
                multiplicative_noise_cov,
            )
        orientation, orientation_variance = self._orientation_update(
            measurement,
            center_estimate,
            orientation,
            semi_axes,
            orientation_variance,
            multiplicative_noise_cov,
            shape_measurement_covariance,
        )
        semi_axes, axis_covariance = self._axis_update(
            measurement,
            center_estimate,
            self.shape_state[0],
            semi_axes,
            axis_covariance,
            multiplicative_noise_cov,
            shape_measurement_covariance,
        )
        semi_axes, axis_covariance = self._canonicalize_axes_and_axis_covariance(
            semi_axes,
            axis_covariance,
        )

        self.kinematic_state = kinematic_state
        self.covariance = self._project_symmetric_covariance(covariance)
        self.shape_state = array([orientation, semi_axes[0], semi_axes[1]])
        self.shape_covariance = self._decoupled_shape_covariance(
            self._regularize_variance(orientation_variance),
            axis_covariance,
        )

    def _kinematic_update(
        self,
        measurement,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
    ):
        extent_transform = self._extent_transform()
        innovation_covariance = self._regularize_covariance(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + extent_transform @ multiplicative_noise_cov @ extent_transform.T
            + meas_noise_cov
        )
        kinematic_cross_covariance = self.covariance @ measurement_matrix.T
        kinematic_gain = self._gain_from_cross_covariance(
            kinematic_cross_covariance,
            innovation_covariance,
        )
        innovation = measurement - measurement_matrix @ self.kinematic_state
        kinematic_state = self.kinematic_state + kinematic_gain @ innovation
        covariance = self.covariance - (
            kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )
        return kinematic_state, covariance

    def _batch_kinematic_update(
        self,
        measurements,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
    ):
        n_measurements = measurements.shape[1]
        centroid = mean(measurements, axis=1)
        extent_transform = self._extent_transform()
        centroid_covariance = self._symmetrize(
            (
                extent_transform @ multiplicative_noise_cov @ extent_transform.T
                + meas_noise_cov
            )
            / n_measurements
        )
        innovation_covariance = self._regularize_covariance(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + centroid_covariance
        )
        kinematic_cross_covariance = self.covariance @ measurement_matrix.T
        kinematic_gain = self._gain_from_cross_covariance(
            kinematic_cross_covariance,
            innovation_covariance,
        )
        innovation = centroid - measurement_matrix @ self.kinematic_state
        self.kinematic_state = self.kinematic_state + kinematic_gain @ innovation
        self.covariance = self._project_symmetric_covariance(
            self.covariance
            - kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )

    # pylint: disable=too-many-arguments
    def _orientation_update(
        self,
        measurement,
        center_estimate,
        orientation,
        semi_axes,
        orientation_variance,
        multiplicative_noise_cov,
        shape_measurement_covariance,
    ):
        shifted_measurement = measurement - center_estimate
        semi_axis_1, semi_axis_2 = semi_axes
        extent_transform = self._rotation(orientation) @ array(
            [[semi_axis_1, 0.0], [0.0, semi_axis_2]]
        )
        first_extent_row = extent_transform[0, :]
        second_extent_row = extent_transform[1, :]
        first_orientation_jacobian = array(
            [-semi_axis_1 * sin(orientation), -semi_axis_2 * cos(orientation)]
        )
        second_orientation_jacobian = array(
            [semi_axis_1 * cos(orientation), -semi_axis_2 * sin(orientation)]
        )

        extent_covariance = (
            extent_transform @ multiplicative_noise_cov @ extent_transform.T
        )
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
        quadratic_covariance = self._symmetrize(
            shape_measurement_covariance + extent_covariance + orientation_covariance
        )

        selection_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        transpose_selection_matrix = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        pseudo_measurement = array(
            [
                shifted_measurement[0] ** 2,
                shifted_measurement[1] ** 2,
                shifted_measurement[0] * shifted_measurement[1],
            ]
        )
        expected_pseudo_measurement = selection_matrix @ self._vectorize_columns(
            quadratic_covariance
        )
        pseudo_covariance = self._regularize_covariance(
            selection_matrix
            @ kron(quadratic_covariance, quadratic_covariance)
            @ (selection_matrix + transpose_selection_matrix).T
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
        innovation = pseudo_measurement - expected_pseudo_measurement
        posterior_orientation = orientation + (orientation_gain @ innovation)[0]
        posterior_orientation_variance = (
            orientation_variance
            - (orientation_gain @ orientation_cross_covariance.T)[0, 0]
        )
        return posterior_orientation, posterior_orientation_variance

    # pylint: disable=too-many-arguments
    def _axis_update(
        self,
        measurement,
        center_estimate,
        orientation,
        semi_axes,
        axis_covariance,
        multiplicative_noise_cov,
        shape_measurement_covariance,
    ):
        shifted_measurement = measurement - center_estimate
        rotation_to_axis_frame = self._rotation(-orientation)
        rotated_measurement_covariance = (
            rotation_to_axis_frame
            @ shape_measurement_covariance
            @ rotation_to_axis_frame.T
        )
        rotated_measurement = rotation_to_axis_frame @ shifted_measurement
        pseudo_measurement = rotated_measurement**2

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
        pseudo_covariance = self._regularize_covariance(
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
        axis_gain = self._gain_from_cross_covariance(
            axis_cross_covariance,
            pseudo_covariance,
        )
        posterior_axes = semi_axes + axis_gain @ (
            pseudo_measurement - expected_pseudo_measurement
        )
        posterior_axis_covariance = axis_covariance - (
            axis_gain @ axis_cross_covariance.T
        )
        return posterior_axes, self._symmetrize(posterior_axis_covariance)


MemQkfTracker = MEMQKFTracker
