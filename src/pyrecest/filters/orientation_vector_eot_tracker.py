from __future__ import annotations

# pylint: disable=no-name-in-module,no-member
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals,duplicate-code
from pyrecest.backend import (
    array,
    concatenate,
    cos,
    diag,
    eye,
    linalg,
    linspace,
    mean,
    maximum,
    pi,
    sin,
    sqrt,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class OrientationVectorEOTTracker(AbstractExtendedObjectTracker):
    """Orientation-vector variational-Bayes tracker for a 2-D ellipse.

    This tracker implements a PyRecEst-compatible EOT-OV style filter inspired
    by Wen, Zheng, and Zeng, "Extended Object Tracking Using an Orientation
    Vector Based on Constrained Filtering," Remote Sensing, 2025.  The state is
    split into

    * Euclidean kinematics ``x`` with Gaussian covariance ``P``;
    * an orientation vector ``epsilon = [cos(theta), sin(theta)]`` with a local
      Gaussian covariance projected onto the unit-circle constraint; and
    * two inverse-gamma random variables for the squared semi-axis lengths.

    The public shape estimate follows the convention of the existing MEM
    trackers: ``[orientation, semi_axis_1, semi_axis_2]``.  The default variant
    uses the paper's velocity-heading constraint.  Disable it with
    ``use_heading_constraint=False`` or instantiate :class:`EOTOV0Tracker`.

    The implementation deliberately exposes the same high-level methods as the
    existing extended-object trackers: ``predict_linear``,
    ``predict_constant_velocity``, ``update``, ``get_point_estimate``,
    ``get_point_estimate_kinematics``, ``get_point_estimate_shape``,
    ``get_point_estimate_extent``, and ``get_contour_points``.
    """

    measurement_dim = 2

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        kinematic_state,
        covariance,
        shape_state,
        orientation_vector_covariance=None,
        inverse_gamma_shape=10.0,
        inverse_gamma_scale=None,
        measurement_noise_cov=None,
        measurement_matrix=None,
        velocity_indices=(2, 3),
        num_iterations=10,
        forgetting_factor=0.99,
        extent_scale=0.25,
        orientation_sys_noise_default=0.0,
        heading_noise_variance=0.0324,
        use_heading_constraint=True,
        speed_threshold=1e-9,
        minimum_orientation_vector_variance=1e-12,
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
        orientation = shape_state[0]
        self.orientation_vector = self._normalize_orientation_vector(
            array([cos(orientation), sin(orientation)])
        )
        self.minimum_orientation_vector_variance = float(
            minimum_orientation_vector_variance
        )
        if self.minimum_orientation_vector_variance <= 0.0:
            raise ValueError("minimum_orientation_vector_variance must be positive")

        if orientation_vector_covariance is None:
            tangent = array([-sin(orientation), cos(orientation)])
            orientation_vector_covariance = 0.1 * self._outer(tangent)
            orientation_vector_covariance = orientation_vector_covariance + (
                self.minimum_orientation_vector_variance * eye(2)
            )
        self.orientation_vector_covariance = self._as_covariance_matrix(
            orientation_vector_covariance,
            self.measurement_dim,
            "orientation_vector_covariance",
        )

        self.alpha = self._as_positive_vector(
            inverse_gamma_shape,
            self.measurement_dim,
            "inverse_gamma_shape",
        )
        if float(self.alpha[0]) <= 1.0 or float(self.alpha[1]) <= 1.0:
            raise ValueError("inverse_gamma_shape entries must be greater than 1")
        if inverse_gamma_scale is None:
            self.beta = shape_state[1:] ** 2 * (self.alpha - 1.0)
        else:
            self.beta = self._as_positive_vector(
                inverse_gamma_scale,
                self.measurement_dim,
                "inverse_gamma_scale",
            )

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
                require_positive_definite=False,
            )

        self.velocity_indices = self._normalize_velocity_indices(velocity_indices)
        self.num_iterations = int(num_iterations)
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        self.forgetting_factor = float(forgetting_factor)
        if self.forgetting_factor <= 0.0:
            raise ValueError("forgetting_factor must be positive")
        self.extent_scale = float(extent_scale)
        if self.extent_scale <= 0.0:
            raise ValueError("extent_scale must be positive")
        self.orientation_sys_noise_default = self._as_nonnegative_scalar(
            orientation_sys_noise_default,
            "orientation_sys_noise_default",
        )
        self.heading_noise_variance = self._as_nonnegative_scalar(
            heading_noise_variance,
            "heading_noise_variance",
        )
        self.use_heading_constraint = bool(use_heading_constraint)
        self.speed_threshold = float(speed_threshold)
        if self.speed_threshold < 0.0:
            raise ValueError("speed_threshold must be non-negative")
        self.covariance_regularization = float(covariance_regularization)
        if self.covariance_regularization < 0.0:
            raise ValueError("covariance_regularization must be non-negative")

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _outer(vector):
        return vector[:, None] @ vector[None, :]

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
                require_positive_definite=False,
            )
        if self.measurement_noise_cov is not None:
            return self.measurement_noise_cov
        return zeros((self.measurement_dim, self.measurement_dim))

    def _normalize_velocity_indices(self, velocity_indices):
        if len(velocity_indices) != 2:
            raise ValueError("velocity_indices must contain exactly two entries")
        state_dim = self.kinematic_state.shape[0]
        normalized = []
        for index in velocity_indices:
            index = int(index)
            if index < 0:
                index += state_dim
            if index < 0 or index >= state_dim:
                raise ValueError(
                    "velocity index out of bounds for kinematic state dimension "
                    f"{state_dim}: {velocity_indices}"
                )
            normalized.append(index)
        if normalized[0] == normalized[1]:
            raise ValueError("velocity_indices must refer to two distinct states")
        return tuple(normalized)

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

    def _normalize_orientation_vector(self, orientation_vector):
        norm = sqrt(orientation_vector @ orientation_vector.T)
        if float(norm) <= 0.0:
            raise ValueError("orientation_vector must have non-zero norm")
        return orientation_vector / norm

    def _project_orientation_covariance(self, orientation_vector, covariance):
        tangent = array([-orientation_vector[1], orientation_vector[0]])
        tangent_variance = maximum(
            tangent @ covariance @ tangent.T,
            self.minimum_orientation_vector_variance,
        )
        projected = tangent_variance * self._outer(tangent)
        projected = projected + self.minimum_orientation_vector_variance * eye(2)
        return self._symmetrize(projected)

    def _project_orientation_gaussian(self, orientation_vector, covariance):
        orientation_vector = self._normalize_orientation_vector(orientation_vector)
        covariance = self._project_orientation_covariance(orientation_vector, covariance)
        return orientation_vector, covariance

    @staticmethod
    def _rotation_matrix_from_vector(orientation_vector):
        return array(
            [
                [orientation_vector[0], -orientation_vector[1]],
                [orientation_vector[1], orientation_vector[0]],
            ]
        )

    def _semi_axis_lengths(self):
        return sqrt(self.beta / (self.alpha - 1.0))

    def _extent_transform(self):
        return self._rotation_matrix_from_vector(self.orientation_vector) @ diag(
            self._semi_axis_lengths()
        )

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
        orientation = linalg.norm(self.orientation_vector)
        # atan2 is not consistently exported by all PyRecEst backends in older
        # revisions.  The arccos-free expression below preserves a scalar angle
        # through NumPy-compatible backends via the Python math fallback.
        from math import atan2  # pylint: disable=import-outside-toplevel

        orientation = atan2(float(self.orientation_vector[1]), float(self.orientation_vector[0]))
        return concatenate([array([orientation]), axes])

    def get_point_estimate_extent(self, flatten_matrix=False):
        extent_transform = self._extent_transform()
        extent = self._symmetrize(extent_transform @ extent_transform.T)
        if flatten_matrix:
            return extent.flatten()
        return extent

    def get_inverse_gamma_parameters(self):
        """Return the inverse-gamma shape and scale parameters."""
        return self.alpha.copy(), self.beta.copy()

    def get_orientation_vector_state(self):
        """Return the orientation-vector mean and covariance."""
        return self.orientation_vector.copy(), self.orientation_vector_covariance.copy()

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        orientation_sys_noise=None,
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

        if orientation_sys_noise is None:
            orientation_sys_noise = self.orientation_sys_noise_default
        orientation_sys_noise = self._as_nonnegative_scalar(
            orientation_sys_noise,
            "orientation_sys_noise",
        )
        self.orientation_vector_covariance = self._symmetrize(
            self.orientation_vector_covariance + orientation_sys_noise * eye(2)
        )
        (
            self.orientation_vector,
            self.orientation_vector_covariance,
        ) = self._project_orientation_gaussian(
            self.orientation_vector,
            self.orientation_vector_covariance,
        )

        gamma = self.forgetting_factor if forgetting_factor is None else float(forgetting_factor)
        if gamma <= 0.0:
            raise ValueError("forgetting_factor must be positive")
        self.alpha = gamma * self.alpha
        self.beta = gamma * self.beta
        if float(self.alpha[0]) <= 1.0 or float(self.alpha[1]) <= 1.0:
            raise ValueError(
                "The prediction made inverse-gamma alpha <= 1; increase "
                "inverse_gamma_shape or use a larger forgetting_factor."
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
        orientation_sys_noise=None,
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

    def _velocity_vector(self):
        velocity_x_index, velocity_y_index = self.velocity_indices
        return array(
            [self.kinematic_state[velocity_x_index], self.kinematic_state[velocity_y_index]]
        )

    def _fuse_orientation_vector(self, measurement, measurement_covariance):
        measurement = self._normalize_orientation_vector(array(measurement))
        measurement_covariance = self._as_covariance_matrix(
            measurement_covariance,
            2,
            "orientation measurement covariance",
            require_positive_definite=False,
        )
        prior_cov = self._symmetrize(
            self.orientation_vector_covariance
            + self.minimum_orientation_vector_variance * eye(2)
        )
        innovation_cov = self._symmetrize(prior_cov + measurement_covariance)
        if self.covariance_regularization > 0.0:
            innovation_cov = innovation_cov + self.covariance_regularization * eye(2)
        gain = linalg.solve(innovation_cov.T, prior_cov.T).T
        orientation_unprojected = self.orientation_vector + gain @ (
            measurement - self.orientation_vector
        )
        covariance_unprojected = self._symmetrize(
            prior_cov - gain @ innovation_cov @ gain.T
        )
        (
            self.orientation_vector,
            self.orientation_vector_covariance,
        ) = self._project_orientation_gaussian(
            orientation_unprojected,
            covariance_unprojected,
        )

    def _update_orientation_from_heading(self):
        if not self.use_heading_constraint:
            return False
        velocity = self._velocity_vector()
        speed_squared = velocity @ velocity.T
        if float(speed_squared) <= self.speed_threshold**2:
            return False
        speed = sqrt(speed_squared)
        heading = velocity / speed
        velocity_covariance = zeros((2, 2))
        velocity_x_index, velocity_y_index = self.velocity_indices
        velocity_covariance[0, 0] = self.covariance[velocity_x_index, velocity_x_index]
        velocity_covariance[0, 1] = self.covariance[velocity_x_index, velocity_y_index]
        velocity_covariance[1, 0] = self.covariance[velocity_y_index, velocity_x_index]
        velocity_covariance[1, 1] = self.covariance[velocity_y_index, velocity_y_index]
        tangent = array([-heading[1], heading[0]])
        heading_variance = tangent @ velocity_covariance @ tangent.T / maximum(
            speed_squared,
            self.speed_threshold**2,
        )
        heading_variance = maximum(heading_variance + self.heading_noise_variance, 0.0)
        measurement_covariance = heading_variance * self._outer(tangent)
        measurement_covariance = measurement_covariance + (
            self.minimum_orientation_vector_variance * eye(2)
        )
        self._fuse_orientation_vector(heading, measurement_covariance)
        return True

    def _update_orientation_from_measurement_cloud(
        self,
        measurements,
        measurement_matrix,
    ):
        if measurements.shape[1] < 2:
            return False
        predicted = measurement_matrix @ self.kinematic_state
        residuals = measurements - predicted[:, None]
        scatter = residuals @ residuals.T / measurements.shape[1]
        eigvals, eigvecs = linalg.eigh(self._symmetrize(scatter))
        axis = eigvecs[:, 1]
        if float(axis @ self.orientation_vector.T) < 0.0:
            axis = -axis
        anisotropy = maximum(eigvals[1] - eigvals[0], 0.0)
        variance = 1.0 / maximum(measurements.shape[1] * anisotropy, 1.0)
        tangent = array([-axis[1], axis[0]])
        measurement_covariance = variance * self._outer(tangent)
        measurement_covariance = measurement_covariance + (
            self.minimum_orientation_vector_variance * eye(2)
        )
        self._fuse_orientation_vector(axis, measurement_covariance)
        return True

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        num_iterations=None,
        use_heading_constraint=None,
    ):
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return
        measurement_matrix = self._get_measurement_matrix(meas_mat)
        meas_noise_cov = self._get_measurement_noise(meas_noise_cov)
        num_iterations = self.num_iterations if num_iterations is None else int(num_iterations)
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        old_use_heading_constraint = self.use_heading_constraint
        if use_heading_constraint is not None:
            self.use_heading_constraint = bool(use_heading_constraint)
        try:
            for _ in range(num_iterations):
                self._update_once(measurements, measurement_matrix, meas_noise_cov)
        finally:
            self.use_heading_constraint = old_use_heading_constraint

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _update_once(self, measurements, measurement_matrix, meas_noise_cov):
        measurement_count = measurements.shape[1]
        expected_delta = self.beta / (self.alpha - 1.0)
        orientation_matrix = self._rotation_matrix_from_vector(self.orientation_vector)
        scaled_axes = sqrt(self.extent_scale * expected_delta)
        spread_matrix = orientation_matrix @ diag(scaled_axes)
        spatial_spread = self._symmetrize(spread_matrix @ spread_matrix.T)
        innovation_cov = self._symmetrize(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + (spatial_spread + meas_noise_cov) / measurement_count
        )
        if self.covariance_regularization > 0.0:
            innovation_cov = innovation_cov + self.covariance_regularization * eye(2)
        z_bar = mean(measurements, axis=1)
        predicted = measurement_matrix @ self.kinematic_state
        kinematic_cross_cov = self.covariance @ measurement_matrix.T
        gain = linalg.solve(innovation_cov.T, kinematic_cross_cov.T).T
        self.kinematic_state = self.kinematic_state + gain @ (z_bar - predicted)
        self.covariance = self._symmetrize(
            self.covariance - gain @ innovation_cov @ gain.T
        )

        self._update_orientation_from_measurement_cloud(measurements, measurement_matrix)
        self._update_orientation_from_heading()

        predicted = measurement_matrix @ self.kinematic_state
        residuals = measurements - predicted[:, None]
        body_residuals = self._rotation_matrix_from_vector(self.orientation_vector).T @ residuals
        hph = measurement_matrix @ self.covariance @ measurement_matrix.T
        noise_body = self._rotation_matrix_from_vector(self.orientation_vector).T @ (
            hph + meas_noise_cov
        ) @ self._rotation_matrix_from_vector(self.orientation_vector)
        scatter = body_residuals @ body_residuals.T
        scatter = scatter + measurement_count * noise_body
        self.alpha = self.alpha + 0.5 * measurement_count
        self.beta = self.beta + 0.5 * diag(scatter) / self.extent_scale

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


class EOTOV0Tracker(OrientationVectorEOTTracker):
    """EOT-OV0 ablation without the velocity-heading pseudo-measurement."""

    def __init__(self, *args, **kwargs):
        kwargs["use_heading_constraint"] = False
        super().__init__(*args, **kwargs)


EOTOVTracker = OrientationVectorEOTTracker
OrientationVectorEOT0Tracker = EOTOV0Tracker
