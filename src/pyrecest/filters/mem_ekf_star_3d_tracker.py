"""MEM-EKF* moment approximation for a spatial ellipsoid with SO(3) orientation."""

from __future__ import annotations

from math import isfinite

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all as backend_all,
    array,
    concatenate,
    diag,
    eye,
    linalg,
    maximum,
    stack,
    trace,
    zeros,
)
from pyrecest.exceptions import ShapeError

from ._spatial_mem_tracker import _SpatialMEMTracker


class MEMEKFStar3DTracker(_SpatialMEMTracker):
    """Sequential MEM-EKF* tracker for one fully oriented 3-D ellipsoid.

    Measurements obey ``z = H x + R diag(a) h + v``. ``orientation`` is a
    proper 3-by-3 rotation from the object's principal-axis frame to the world
    frame, and ``semi_axes`` contains three positive semi-axis lengths. The
    6-by-6 ``shape_covariance`` describes ``[delta_theta, delta_a]``, with
    right/body rotation errors: ``R_true = R exp(skew(delta_theta))``.
    Rotation/axis cross-covariances are retained. Orientation corrections use
    the SO(3) exponential and reset the covariance into the new tangent frame.

    This is a 3-D generalization of the MEM-EKF* moment approximation, not an
    exact Bayesian update. The initial orientation should be locally close to
    the desired ellipsoid frame. Axes retain their labels and are never sorted.
    Rotation about equal axes is unobservable for isotropic multiplicative
    noise. The default ``multiplicative_noise_cov = I / 5`` matches the second
    moment of uniform sources throughout a filled 3-D ellipsoid; fourth moments
    in the update use the MEM-EKF* Gaussian approximation.

    Covariances accept a scalar, a diagonal vector, or a square matrix.
    ``measurement_matrix`` defaults to selecting the first three kinematic
    coordinates. ``minimum_axis_length`` clips corrected axes to a positive
    lower bound. ``covariance_regularization`` adds a nonnegative diagonal term
    to the measurement and quadratic innovation covariances.
    """

    _quadratic_pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

    def __init__(
        self,
        kinematic_state,
        covariance,
        orientation,
        semi_axes,
        shape_covariance,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        covariance_regularization=0.0,
        minimum_axis_length=1e-9,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.kinematic_state = self._as_real_array(kinematic_state, "kinematic_state")
        if self.kinematic_state.ndim != 1 or self.kinematic_state.shape[0] < 3:
            raise ShapeError(
                "kinematic_state must be a vector containing a 3-D position"
            )
        self.covariance = self._as_covariance_matrix(
            covariance, self.kinematic_state.shape[0], "covariance"
        )
        self.orientation = self._as_rotation(orientation, "orientation")
        self.semi_axes = self._as_vector(semi_axes, 3, "semi_axes")
        if not bool(backend_all(self.semi_axes > 0.0)):
            raise ValueError("semi_axes must be positive")
        self.shape_covariance = self._as_covariance_matrix(
            shape_covariance, 6, "shape_covariance"
        )
        self.measurement_matrix = (
            None
            if measurement_matrix is None
            else self._get_measurement_matrix(measurement_matrix)
        )
        self.multiplicative_noise_cov = self._as_multiplicative_noise(
            0.2 * eye(3)
            if multiplicative_noise_cov is None
            else multiplicative_noise_cov
        )
        self.covariance_regularization = float(covariance_regularization)
        if (
            not isfinite(self.covariance_regularization)
            or self.covariance_regularization < 0.0
        ):
            raise ValueError(
                "covariance_regularization must be finite and non-negative"
            )
        self.minimum_axis_length = float(minimum_axis_length)
        if not isfinite(self.minimum_axis_length) or self.minimum_axis_length <= 0.0:
            raise ValueError("minimum_axis_length must be finite and positive")

    def _extent_transform(self):
        return self.orientation @ diag(self.semi_axes)

    def _extent_row_jacobians(self):
        axes = diag(self.semi_axes)
        return stack(
            [
                concatenate([axes @ self._skew(row), diag(row)], axis=1)
                for row in self.orientation
            ]
        )

    def _shape_noise_covariance(self, multiplicative_noise_cov):
        jacobians = self._extent_row_jacobians()
        return self._symmetrize(
            array(
                [
                    [
                        trace(
                            self.shape_covariance
                            @ jacobians[j].T
                            @ multiplicative_noise_cov
                            @ jacobians[i]
                        )
                        for j in range(3)
                    ]
                    for i in range(3)
                ]
            )
        )

    def _shape_pseudo_jacobian(self, multiplicative_noise_cov):
        extent = self._extent_transform()
        jacobians = self._extent_row_jacobians()
        return stack(
            [
                extent[i] @ multiplicative_noise_cov @ jacobians[j]
                + extent[j] @ multiplicative_noise_cov @ jacobians[i]
                for i, j in self._quadratic_pairs
            ]
        )

    @classmethod
    def _pseudo_measurement_covariance(cls, innovation_covariance):
        return array(
            [
                [
                    innovation_covariance[i, k] * innovation_covariance[j, ell]
                    + innovation_covariance[i, ell] * innovation_covariance[j, k]
                    for k, ell in cls._quadratic_pairs
                ]
                for i, j in cls._quadratic_pairs
            ]
        )

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        """Process target-originated points sequentially; a 3-by-3 set uses columns."""
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return
        measurement_matrix = self._get_measurement_matrix(meas_mat)
        measurement_noise = (
            zeros((3, 3))
            if meas_noise_cov is None
            else self._as_covariance_matrix(meas_noise_cov, 3, "meas_noise_cov")
        )
        multiplicative_noise = (
            self.multiplicative_noise_cov
            if multiplicative_noise_cov is None
            else self._as_multiplicative_noise(multiplicative_noise_cov)
        )
        for index in range(measurements.shape[1]):
            self._update_single_measurement(
                measurements[:, index],
                measurement_matrix,
                measurement_noise,
                multiplicative_noise,
            )
        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _update_single_measurement(
        self, measurement, measurement_matrix, measurement_noise, multiplicative_noise
    ):
        extent = self._extent_transform()
        pseudo_jacobian = self._shape_pseudo_jacobian(multiplicative_noise)
        innovation = measurement - measurement_matrix @ self.kinematic_state
        innovation_covariance = self._symmetrize(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + extent @ multiplicative_noise @ extent.T
            + self._shape_noise_covariance(multiplicative_noise)
            + measurement_noise
            + self.covariance_regularization * eye(3)
        )
        kinematic_cross_covariance = self.covariance @ measurement_matrix.T
        kinematic_gain = linalg.solve(
            innovation_covariance, kinematic_cross_covariance.T
        ).T
        kinematic_state = self.kinematic_state + kinematic_gain @ innovation
        covariance = self._project_covariance(
            self.covariance - kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )

        pseudo_measurement = array(
            [innovation[i] * innovation[j] for i, j in self._quadratic_pairs]
        )
        pseudo_mean = array(
            [innovation_covariance[i, j] for i, j in self._quadratic_pairs]
        )
        pseudo_covariance = self._pseudo_measurement_covariance(innovation_covariance)
        pseudo_covariance = pseudo_covariance + self.covariance_regularization * eye(6)
        shape_cross_covariance = self.shape_covariance @ pseudo_jacobian.T
        shape_gain = linalg.solve(pseudo_covariance, shape_cross_covariance.T).T
        correction = shape_gain @ (pseudo_measurement - pseudo_mean)
        shape_covariance = self._project_covariance(
            self.shape_covariance - shape_gain @ pseudo_covariance @ shape_gain.T
        )
        reset = linalg.block_diag(self._right_jacobian(correction[:3]), eye(3))
        orientation = self.orientation @ self._rotation_exp(correction[:3])
        semi_axes = maximum(self.semi_axes + correction[3:], self.minimum_axis_length)
        shape_covariance = self._project_covariance(reset @ shape_covariance @ reset.T)
        for value in (
            kinematic_state,
            covariance,
            orientation,
            semi_axes,
            shape_covariance,
        ):
            self._require_finite(value, "posterior state")
        self.kinematic_state = kinematic_state
        self.covariance = covariance
        self.orientation = orientation
        self.semi_axes = semi_axes
        self.shape_covariance = shape_covariance

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_sys_noise=None,
        rotation_increment=None,
    ):
        """Predict kinematics and a constant-axis shape with optional body rotation.

        ``rotation_increment`` is a deterministic rotation vector applied on the
        right. Existing body error covariance is transported by the inverse
        increment's adjoint. ``shape_sys_noise`` is then added in the predicted
        body frame. Omitted increments/noises give constant-shape prediction.
        All inputs are validated before the tracker state changes.
        """
        state_dim = self.kinematic_state.shape[0]
        system_matrix = self._as_real_array(system_matrix, "system_matrix")
        if system_matrix.shape != (state_dim, state_dim):
            raise ShapeError(
                "system_matrix must have shape (kinematic_dim, kinematic_dim)"
            )
        process_noise = (
            zeros((state_dim, state_dim))
            if sys_noise is None
            else self._as_covariance_matrix(sys_noise, state_dim, "sys_noise")
        )
        control = (
            zeros(state_dim)
            if inputs is None
            else self._as_vector(inputs, state_dim, "inputs")
        )
        shape_noise = (
            zeros((6, 6))
            if shape_sys_noise is None
            else self._as_covariance_matrix(shape_sys_noise, 6, "shape_sys_noise")
        )
        increment = (
            eye(3)
            if rotation_increment is None
            else self._rotation_exp(rotation_increment)
        )
        transport = linalg.block_diag(increment.T, eye(3))
        kinematic_state = system_matrix @ self.kinematic_state + control
        covariance = self._project_covariance(
            system_matrix @ self.covariance @ system_matrix.T + process_noise
        )
        orientation = self.orientation @ increment
        shape_covariance = self._project_covariance(
            transport @ self.shape_covariance @ transport.T + shape_noise
        )
        for value in (kinematic_state, covariance, orientation, shape_covariance):
            self._require_finite(value, "predicted state")
        self.kinematic_state = kinematic_state
        self.covariance = covariance
        self.orientation = orientation
        self.shape_covariance = shape_covariance
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()


MemEkfStar3DTracker = MEMEKFStar3DTracker
