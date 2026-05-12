from __future__ import annotations

# pylint: disable=no-name-in-module,no-member
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
from pyrecest.backend import (
    array,
    concatenate,
    copy,
    cos,
    diag,
    diagonal,
    exp,
    eye,
    linalg,
    linspace,
    mean,
    pi,
    sin,
    sqrt,
    stack,
    trace,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class VBRMTracker(AbstractExtendedObjectTracker):
    """Variational-Bayes random-matrix tracker for a 2-D elliptical object.

    The kinematic state is Euclidean.  The extent is represented by an
    orientation estimate and two inverse-gamma random variables for the
    principal semi-axis variances.  Public shape estimates use the MEM-style
    vector ``[orientation, semi_axis_1, semi_axis_2]``.

    This implementation follows the update equations of Tuncer and Özkan,
    "Random Matrix Based Extended Target Tracking With Orientation: A New Model
    and Inference," IEEE Transactions on Signal Processing, 2021.
    """

    measurement_dim = 2

    def __init__(
        self,
        kinematic_state,
        covariance,
        shape_state,
        orientation_variance,
        inverse_gamma_shape=10.0,
        measurement_noise_cov=None,
        measurement_matrix=None,
        num_iterations=5,
        forgetting_factor=1.0,
        extent_scale=0.25,
        covariance_regularization=0.0,
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
        self.kinematic_state = array(kinematic_state)
        if self.kinematic_state.ndim != 1:
            raise ValueError("kinematic_state must be one-dimensional")
        if self.kinematic_state.shape[0] < self.measurement_dim:
            raise ValueError("kinematic_state must contain at least a 2-D position")

        self.covariance = self._as_covariance_matrix(
            covariance,
            self.kinematic_state.shape[0],
            "covariance",
        )
        shape_state = array(shape_state)
        self._validate_shape_state(shape_state)
        self.orientation = shape_state[0]
        self.orientation_variance = self._as_positive_scalar(
            orientation_variance,
            "orientation_variance",
        )

        self.alpha = self._as_positive_vector(
            inverse_gamma_shape,
            self.measurement_dim,
            "inverse_gamma_shape",
        )
        if float(self.alpha[0]) <= 1.0 or float(self.alpha[1]) <= 1.0:
            raise ValueError("inverse_gamma_shape entries must be greater than 1")
        self.beta = shape_state[1:] ** 2 * (self.alpha - 1.0)

        self.measurement_matrix = None
        if measurement_matrix is not None:
            self.measurement_matrix = array(measurement_matrix)
            self._validate_measurement_matrix(self.measurement_matrix)

        self.measurement_noise_cov = None
        if measurement_noise_cov is not None:
            self.measurement_noise_cov = self._as_covariance_matrix(
                measurement_noise_cov,
                self.measurement_dim,
                "measurement_noise_cov",
            )

        self.num_iterations = int(num_iterations)
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        self.forgetting_factor = float(forgetting_factor)
        if self.forgetting_factor <= 0.0:
            raise ValueError("forgetting_factor must be positive")
        self.extent_scale = float(extent_scale)
        if self.extent_scale <= 0.0:
            raise ValueError("extent_scale must be positive")
        self.covariance_regularization = float(covariance_regularization)
        if self.covariance_regularization < 0.0:
            raise ValueError("covariance_regularization must be non-negative")

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _validate_positive_definite(matrix, name):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
        linalg.cholesky(matrix)

    @classmethod
    def _as_covariance_matrix(cls, value, dim, name, require_positive_definite=True):
        matrix = array(value)
        if matrix.ndim == 0:
            matrix = matrix * eye(dim)
        elif matrix.ndim == 1:
            if matrix.shape[0] != dim:
                raise ValueError(f"{name} vector must have length {dim}")
            matrix = diag(matrix)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        matrix = cls._symmetrize(matrix)
        if require_positive_definite:
            cls._validate_positive_definite(matrix, name)
        return matrix

    @staticmethod
    def _as_positive_scalar(value, name):
        scalar = array(value)
        if scalar.ndim == 1 and scalar.shape == (1,):
            scalar = scalar[0]
        if scalar.shape != ():
            raise ValueError(f"{name} must be scalar")
        if float(scalar) <= 0.0:
            raise ValueError(f"{name} must be positive")
        return scalar

    @staticmethod
    def _as_nonnegative_scalar(value, name):
        scalar = array(value)
        if scalar.ndim == 1 and scalar.shape == (1,):
            scalar = scalar[0]
        if scalar.shape != ():
            raise ValueError(f"{name} must be scalar")
        if float(scalar) < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return scalar

    @staticmethod
    def _as_positive_vector(value, dim, name):
        vector = array(value)
        if vector.ndim == 0:
            vector = vector * array([1.0] * dim)
        if vector.shape != (dim,):
            raise ValueError(f"{name} must be scalar or have shape ({dim},)")
        for index in range(dim):
            if float(vector[index]) <= 0.0:
                raise ValueError(f"{name} entries must be positive")
        return vector

    @staticmethod
    def _validate_shape_state(shape_state):
        if shape_state.shape != (3,):
            raise ValueError("shape_state must have shape (3,)")
        if float(shape_state[1]) <= 0.0 or float(shape_state[2]) <= 0.0:
            raise ValueError("shape semi-axis lengths must be positive")

    def _validate_measurement_matrix(self, measurement_matrix):
        expected_shape = (self.measurement_dim, self.kinematic_state.shape[0])
        if measurement_matrix.shape != expected_shape:
            raise ValueError(
                "measurement_matrix must have shape "
                f"{expected_shape}, got {measurement_matrix.shape}"
            )

    def _get_measurement_matrix(self, measurement_matrix=None):
        if measurement_matrix is not None:
            measurement_matrix = array(measurement_matrix)
            self._validate_measurement_matrix(measurement_matrix)
            return measurement_matrix
        if self.measurement_matrix is not None:
            return self.measurement_matrix
        return eye(self.measurement_dim, self.kinematic_state.shape[0])

    def _get_measurement_noise(self, meas_noise_cov=None):
        if meas_noise_cov is not None:
            return self._as_covariance_matrix(
                meas_noise_cov,
                self.measurement_dim,
                "meas_noise_cov",
            )
        if self.measurement_noise_cov is not None:
            return self.measurement_noise_cov
        raise ValueError(
            "VBRMTracker requires a positive-definite measurement noise "
            "covariance; pass measurement_noise_cov at construction or "
            "meas_noise_cov to update()."
        )

    def _normalize_measurements(self, measurements):
        measurements = array(measurements)
        if measurements.ndim == 1:
            if measurements.shape[0] != self.measurement_dim:
                raise ValueError("A single measurement must be two-dimensional")
            return measurements.reshape((self.measurement_dim, 1))
        if measurements.ndim != 2:
            raise ValueError("measurements must be a vector or a two-dimensional array")
        if measurements.shape[0] == self.measurement_dim:
            return measurements
        if measurements.shape[1] == self.measurement_dim:
            return measurements.T
        raise ValueError(
            "measurements must have shape (2, n_measurements) or (n_measurements, 2)"
        )

    @staticmethod
    def _rotation_matrix(theta):
        return array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

    @staticmethod
    def _rotation_derivative(theta):
        return array([[-sin(theta), -cos(theta)], [cos(theta), -sin(theta)]])

    @staticmethod
    def _compute_lxl(theta_bar, theta_variance, matrix):
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]
        exp_term = exp(-2.0 * theta_variance)
        cos_term = cos(2.0 * theta_bar) * exp_term
        sin_term = sin(2.0 * theta_bar) * exp_term
        return 0.5 * array(
            [
                [
                    a * (1.0 + cos_term) + d * (1.0 - cos_term) - (c + b) * sin_term,
                    b * (1.0 + cos_term) - c * (1.0 - cos_term) + (a - d) * sin_term,
                ],
                [
                    c * (1.0 + cos_term) - b * (1.0 - cos_term) + (a - d) * sin_term,
                    d * (1.0 + cos_term) + a * (1.0 - cos_term) + (c + b) * sin_term,
                ],
            ]
        )

    def _expected_scaled_extent(self, alpha, beta):
        return diag(self.extent_scale * beta / (alpha - 1.0))

    def _expected_oriented_scaled_extent_inverse(
        self, theta, theta_variance, alpha, beta
    ):
        extent_inverse = diag(alpha / (self.extent_scale * beta))
        rotation_matrix = self._rotation_matrix(theta)
        exp_term = exp(-2.0 * theta_variance)
        isotropic_part = (1.0 - exp_term) * 0.5 * trace(extent_inverse) * eye(2)
        anisotropic_part = (
            exp_term * rotation_matrix @ extent_inverse @ rotation_matrix.T
        )
        return self._symmetrize(isotropic_part + anisotropic_part)

    def _semi_axis_lengths(self):
        return sqrt(self.beta / (self.alpha - 1.0))

    def _extent_transform(self):
        return self._rotation_matrix(self.orientation) @ diag(self._semi_axis_lengths())

    @property
    def extent(self):
        return self.get_point_estimate_extent()

    def get_point_estimate(self):
        return concatenate([self.kinematic_state, self.get_point_estimate_shape()])

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_shape(self, full_axes=False):
        axes = self._semi_axis_lengths()
        if full_axes:
            axes = 2.0 * axes
        return concatenate([array([self.orientation]), axes])

    def get_point_estimate_extent(self, flatten_matrix=False):
        extent_transform = self._extent_transform()
        extent = self._symmetrize(extent_transform @ extent_transform.T)
        if flatten_matrix:
            return extent.flatten()
        return extent

    def get_inverse_gamma_parameters(self):
        """Return copies of the internal inverse-gamma shape and scale vectors."""
        return copy(self.alpha), copy(self.beta)

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        orientation_system_matrix=1.0,
        orientation_sys_noise=0.0,
        forgetting_factor=None,
    ):
        system_matrix = array(system_matrix)
        state_dim = self.kinematic_state.shape[0]
        if system_matrix.shape != (state_dim, state_dim):
            raise ValueError(
                "system_matrix shape must match the kinematic state dimension"
            )
        if sys_noise is None:
            sys_noise = zeros((state_dim, state_dim))
        else:
            sys_noise = self._as_covariance_matrix(
                sys_noise,
                state_dim,
                "sys_noise",
                require_positive_definite=False,
            )

        self.kinematic_state = system_matrix @ self.kinematic_state
        if inputs is not None:
            self.kinematic_state = self.kinematic_state + array(inputs)
        self.covariance = self._symmetrize(
            system_matrix @ self.covariance @ system_matrix.T + sys_noise
        )

        orientation_system_matrix = float(orientation_system_matrix)
        orientation_sys_noise = self._as_nonnegative_scalar(
            orientation_sys_noise,
            "orientation_sys_noise",
        )
        self.orientation = orientation_system_matrix * self.orientation
        self.orientation_variance = (
            orientation_system_matrix**2 * self.orientation_variance
            + orientation_sys_noise
        )

        gamma = (
            self.forgetting_factor
            if forgetting_factor is None
            else float(forgetting_factor)
        )
        if gamma <= 0.0:
            raise ValueError("forgetting_factor must be positive")
        self.alpha = gamma * self.alpha
        self.beta = gamma * self.beta
        if float(self.alpha[0]) <= 1.0 or float(self.alpha[1]) <= 1.0:
            raise ValueError(
                "The prediction made inverse-gamma alpha <= 1, so the extent "
                "mean is undefined. Increase inverse_gamma_shape or use a larger "
                "forgetting_factor."
            )

        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict(self, *args, **kwargs):
        """Alias for :meth:`predict_linear` to match existing tracker APIs."""
        self.predict_linear(*args, **kwargs)

    def predict_constant_velocity(
        self,
        time_delta=1.0,
        sys_noise=None,
        orientation_sys_noise=0.0,
        forgetting_factor=None,
    ):
        if self.kinematic_state.shape[0] != 4:
            raise ValueError(
                "predict_constant_velocity expects a 4-D [x, y, vx, vy] state"
            )
        system_matrix = array(
            [
                [1.0, 0.0, time_delta, 0.0],
                [0.0, 1.0, 0.0, time_delta],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.predict_linear(
            system_matrix,
            sys_noise=sys_noise,
            orientation_sys_noise=orientation_sys_noise,
            forgetting_factor=forgetting_factor,
        )

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        num_iterations=None,
    ):
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return

        measurement_matrix = self._get_measurement_matrix(meas_mat)
        meas_noise_cov = self._get_measurement_noise(meas_noise_cov)
        num_iterations = (
            self.num_iterations if num_iterations is None else int(num_iterations)
        )
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

        self._update_vbrm(
            measurements.T,
            measurement_matrix,
            meas_noise_cov,
            num_iterations,
        )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _update_vbrm(
        self, measurements, measurement_matrix, meas_noise_cov, num_iterations
    ):
        x_prior = self.kinematic_state
        p_prior = self.covariance
        alpha_prior = self.alpha
        beta_prior = self.beta
        theta_prior = self.orientation
        theta_variance_prior = self.orientation_variance

        measurement_count = measurements.shape[0]
        p_prior_inv = linalg.inv(p_prior)
        meas_noise_inv = linalg.inv(meas_noise_cov)

        x_iterations = [x_prior]
        p_iterations = [p_prior]
        alpha_iterations = [alpha_prior]
        beta_iterations = [beta_prior]
        theta_iterations = [theta_prior]
        theta_variance_iterations = [theta_variance_prior]
        z_iterations = [measurements]
        sigma_iterations = [self._expected_scaled_extent(alpha_prior, beta_prior)]

        for _ in range(num_iterations):
            z_current = z_iterations[-1].reshape(
                (measurement_count, self.measurement_dim)
            )
            z_mean = mean(z_current, axis=0)
            exp_oriented_extent_inv = self._expected_oriented_scaled_extent_inverse(
                theta_iterations[-1],
                theta_variance_iterations[-1],
                alpha_iterations[-1],
                beta_iterations[-1],
            )

            information = (
                p_prior_inv
                + measurement_count
                * measurement_matrix.T
                @ exp_oriented_extent_inv
                @ measurement_matrix
            )
            if self.covariance_regularization > 0.0:
                information = information + self.covariance_regularization * eye(
                    information.shape[0]
                )
            rhs = (
                p_prior_inv @ x_prior
                + measurement_count
                * measurement_matrix.T
                @ exp_oriented_extent_inv
                @ z_mean
            )
            p_next = self._symmetrize(linalg.inv(information))
            x_next = p_next @ rhs

            alpha_current = alpha_iterations[-1]
            beta_current = beta_iterations[-1]
            s_x_inv_mean = diag(alpha_current / (self.extent_scale * beta_current))
            theta_current = theta_iterations[-1]
            theta_variance_current = theta_variance_iterations[-1]
            rotation_matrix = self._rotation_matrix(theta_current)
            rotation_derivative = self._rotation_derivative(theta_current)

            delta = 0.0
            capital_delta = 0.0
            innovation_bars = []
            x_current = x_iterations[-1]
            p_current = p_iterations[-1]
            sigma_current = sigma_iterations[-1]
            predicted_measurement_current = measurement_matrix @ x_current

            for measurement_index in range(measurement_count):
                innovation = (
                    z_current[measurement_index] - predicted_measurement_current
                )
                innovation = innovation.reshape((self.measurement_dim, 1))
                innovation_bar = (
                    measurement_matrix @ p_current @ measurement_matrix.T
                    + sigma_current
                    + innovation @ innovation.T
                )
                innovation_bars.append(innovation_bar)
                theta_information = trace(
                    s_x_inv_mean
                    @ rotation_derivative.T
                    @ innovation_bar
                    @ rotation_derivative
                )
                delta = delta + theta_information * theta_current
                delta = delta - trace(
                    s_x_inv_mean
                    @ rotation_matrix.T
                    @ innovation_bar
                    @ rotation_derivative
                )
                capital_delta = capital_delta + theta_information

            innovation_bar_sum = sum(innovation_bars, zeros((2, 2)))
            theta_variance_next = 1.0 / (1.0 / theta_variance_prior + capital_delta)
            theta_next = theta_variance_next * (
                theta_prior / theta_variance_prior + delta
            )

            alpha_next = alpha_prior + 0.5 * measurement_count
            lxl_mean = self._compute_lxl(
                -theta_current,
                theta_variance_current,
                (0.5 / self.extent_scale) * innovation_bar_sum,
            )
            beta_next = beta_prior + diagonal(lxl_mean)

            exp_oriented_extent_inv = self._expected_oriented_scaled_extent_inverse(
                theta_current,
                theta_variance_current,
                alpha_current,
                beta_current,
            )
            sigma_next = self._symmetrize(
                linalg.inv(exp_oriented_extent_inv + meas_noise_inv)
            )
            z_next = stack(
                [
                    sigma_next
                    @ (
                        exp_oriented_extent_inv @ predicted_measurement_current
                        + meas_noise_inv @ measurements[measurement_index]
                    )
                    for measurement_index in range(measurement_count)
                ]
            )

            x_iterations.append(x_next)
            p_iterations.append(p_next)
            alpha_iterations.append(alpha_next)
            beta_iterations.append(beta_next)
            theta_iterations.append(theta_next)
            theta_variance_iterations.append(theta_variance_next)
            sigma_iterations.append(sigma_next)
            z_iterations.append(z_next)

        self.kinematic_state = x_iterations[-1]
        self.covariance = p_iterations[-1]
        self.alpha = alpha_iterations[-1]
        self.beta = beta_iterations[-1]
        self.orientation = theta_iterations[-1]
        self.orientation_variance = theta_variance_iterations[-1]

    def get_contour_points(self, n, scaling_factor=1.0):
        if n <= 0:
            raise ValueError("n must be positive")
        measurement_matrix = self._get_measurement_matrix()
        position_estimate = measurement_matrix @ self.kinematic_state
        angles = linspace(0.0, 2.0 * pi, n, endpoint=False)
        unit_circle = array([cos(angles), sin(angles)])
        contour_points = (
            position_estimate[:, None]
            + scaling_factor * self._extent_transform() @ unit_circle
        )
        return contour_points.T


VbrmTracker = VBRMTracker
