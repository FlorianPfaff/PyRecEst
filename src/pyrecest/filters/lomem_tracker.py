from __future__ import annotations

# pylint: disable=no-name-in-module,no-member
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import (
    arctan2,
    array,
    concatenate,
    cos,
    diag,
    eye,
    linalg,
    linspace,
    maximum,
    mean,
    pi,
    sin,
    sqrt,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class LOMEMTracker(AbstractExtendedObjectTracker):
    """Lambda:Omicron MEM tracker for a 2-D maneuvering elliptical object.

    The state convention is ``[x, y, speed, heading, semi_axis_lambda,
    semi_axis_omicron]``.  The heading defines the Lambda direction and the
    orthogonal Omicron direction.  The measurement update follows the L:OMEM
    point-object-reduction idea from Tesori, Battistelli, Chisci, and Farina,
    "L:OMEM - A fast filter to track maneuvering extended objects", FUSION
    2023: a full scan of target-originated points is reduced to one augmented
    point-object measurement containing centroid, orientation, and axes.

    This class intentionally keeps the first implementation compact: the
    prediction is a unicycle-style maneuvering model and the reduced
    measurement is applied with an EKF correction.
    """

    measurement_dim = 2
    state_dim = 6

    def __init__(
        self,
        state,
        covariance,
        measurement_noise_cov=None,
        extent_scale=0.25,
        orientation_measurement_variance=0.25,
        axis_measurement_variance_scale=0.25,
        minimum_axis_length=1e-9,
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
        self.state = array(state)
        if self.state.shape != (self.state_dim,):
            raise ValueError("state must have shape (6,)")
        self.covariance = self._as_covariance_matrix(
            covariance,
            self.state_dim,
            "covariance",
            require_positive_definite=True,
        )

        self.measurement_noise_cov = None
        if measurement_noise_cov is not None:
            self.measurement_noise_cov = self._as_covariance_matrix(
                measurement_noise_cov,
                self.measurement_dim,
                "measurement_noise_cov",
            )

        self.extent_scale = float(extent_scale)
        if self.extent_scale <= 0.0:
            raise ValueError("extent_scale must be positive")
        self.orientation_measurement_variance = self._as_positive_scalar(
            orientation_measurement_variance,
            "orientation_measurement_variance",
        )
        self.axis_measurement_variance_scale = self._as_positive_scalar(
            axis_measurement_variance_scale,
            "axis_measurement_variance_scale",
        )
        self.minimum_axis_length = float(minimum_axis_length)
        if self.minimum_axis_length <= 0.0:
            raise ValueError("minimum_axis_length must be positive")
        self.covariance_regularization = float(covariance_regularization)
        if self.covariance_regularization < 0.0:
            raise ValueError("covariance_regularization must be non-negative")
        self._canonicalize_axes()

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
    def _wrap_angle(angle):
        return (float(angle) + float(pi)) % (2.0 * float(pi)) - float(pi)

    @staticmethod
    def _wrap_half_turn(angle):
        return (float(angle) + 0.5 * float(pi)) % float(pi) - 0.5 * float(pi)

    @staticmethod
    def _rotation_matrix(theta):
        return array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

    def _regularized_covariance(self, covariance):
        covariance = self._symmetrize(covariance)
        if self.covariance_regularization > 0.0:
            covariance = covariance + self.covariance_regularization * eye(
                covariance.shape[0]
            )
        return covariance

    def _canonicalize_axes(self):
        self.state[3] = self._wrap_angle(self.state[3])
        self.state[4] = maximum(backend_abs(self.state[4]), self.minimum_axis_length)
        self.state[5] = maximum(backend_abs(self.state[5]), self.minimum_axis_length)

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

    def _measurement_scatter(self, measurements, center):
        measurement_count = measurements.shape[1]
        if measurement_count <= 1:
            return zeros((self.measurement_dim, self.measurement_dim))
        centered = measurements - center.reshape((self.measurement_dim, 1))
        return self._symmetrize(centered @ centered.T / (measurement_count - 1))

    def reduce_measurements(self, measurements, meas_noise_cov=None):
        """Reduce a scan to one L:OMEM augmented point-object measurement.

        Returns ``(z, R)``.  For a single detection, ``z`` only contains the
        centroid and ``R`` is 2x2.  For two or more detections, ``z`` contains
        ``[center_x, center_y, heading, semi_axis_lambda, semi_axis_omicron]``.
        """
        measurements = self._normalize_measurements(measurements)
        measurement_count = measurements.shape[1]
        if measurement_count == 0:
            return None, None

        meas_noise_cov = self._get_measurement_noise(meas_noise_cov)
        center = mean(measurements, axis=1)
        scatter = self._measurement_scatter(measurements, center)
        center_covariance = self._regularized_covariance(
            (scatter + meas_noise_cov) / measurement_count
        )
        if measurement_count == 1:
            return center, center_covariance

        eigenvalues, eigenvectors = linalg.eigh(scatter)
        major_variance = maximum(
            eigenvalues[1], self.extent_scale * self.minimum_axis_length**2
        )
        minor_variance = maximum(
            eigenvalues[0], self.extent_scale * self.minimum_axis_length**2
        )
        heading = arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
        axes = sqrt(array([major_variance, minor_variance]) / self.extent_scale)
        z = concatenate([center, array([heading]), axes])

        shape_variance = array(
            [
                float(self.orientation_measurement_variance)
                / max(measurement_count - 1, 1),
                float(self.axis_measurement_variance_scale)
                * float(axes[0]) ** 2
                / max(measurement_count, 1),
                float(self.axis_measurement_variance_scale)
                * float(axes[1]) ** 2
                / max(measurement_count, 1),
            ]
        )
        reduction_covariance = zeros((5, 5))
        reduction_covariance[:2, :2] = center_covariance
        reduction_covariance[2:, 2:] = diag(shape_variance)
        return z, self._regularized_covariance(reduction_covariance)

    def predict_unicycle(
        self,
        time_delta=1.0,
        sys_noise=None,
        longitudinal_acceleration=0.0,
        turn_rate=0.0,
    ):
        """Predict with a Lambda:Omicron unicycle-style maneuvering model."""
        time_delta = float(time_delta)
        speed = self.state[2]
        heading = self.state[3]
        longitudinal_acceleration = float(longitudinal_acceleration)
        turn_rate = float(turn_rate)

        next_state = array(self.state)
        next_state[0] = self.state[0] + time_delta * speed * cos(heading)
        next_state[1] = self.state[1] + time_delta * speed * sin(heading)
        next_state[2] = self.state[2] + time_delta * longitudinal_acceleration
        next_state[3] = self._wrap_angle(self.state[3] + time_delta * turn_rate)

        jacobian = eye(self.state_dim)
        jacobian[0, 2] = time_delta * cos(heading)
        jacobian[0, 3] = -time_delta * speed * sin(heading)
        jacobian[1, 2] = time_delta * sin(heading)
        jacobian[1, 3] = time_delta * speed * cos(heading)

        if sys_noise is None:
            sys_noise = zeros((self.state_dim, self.state_dim))
        else:
            sys_noise = self._as_covariance_matrix(
                sys_noise,
                self.state_dim,
                "sys_noise",
                require_positive_definite=False,
            )

        self.state = next_state
        self.covariance = self._regularized_covariance(
            jacobian @ self.covariance @ jacobian.T + sys_noise
        )
        self._canonicalize_axes()

        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict_linear(self, system_matrix, sys_noise=None, inputs=None):
        """Predict with a user-supplied linear model for the six-state vector."""
        system_matrix = array(system_matrix)
        if system_matrix.shape != (self.state_dim, self.state_dim):
            raise ValueError("system_matrix must have shape (6, 6)")
        if sys_noise is None:
            sys_noise = zeros((self.state_dim, self.state_dim))
        else:
            sys_noise = self._as_covariance_matrix(
                sys_noise,
                self.state_dim,
                "sys_noise",
                require_positive_definite=False,
            )
        self.state = system_matrix @ self.state
        if inputs is not None:
            self.state = self.state + array(inputs)
        self.covariance = self._regularized_covariance(
            system_matrix @ self.covariance @ system_matrix.T + sys_noise
        )
        self._canonicalize_axes()

    def predict(self, *args, **kwargs):
        """Alias for :meth:`predict_unicycle`."""
        self.predict_unicycle(*args, **kwargs)

    @staticmethod
    def _gain_from_cross_covariance(cross_covariance, innovation_covariance):
        return linalg.solve(innovation_covariance.T, cross_covariance.T).T

    def update(self, measurements, meas_noise_cov=None, reduction_covariance=None):
        z, derived_covariance = self.reduce_measurements(
            measurements,
            meas_noise_cov=meas_noise_cov,
        )
        if z is None:
            return
        if reduction_covariance is None:
            reduction_covariance = derived_covariance
        else:
            reduction_covariance = array(reduction_covariance)

        if z.shape[0] == self.measurement_dim:
            measurement_matrix = zeros((self.measurement_dim, self.state_dim))
            measurement_matrix[0, 0] = 1.0
            measurement_matrix[1, 1] = 1.0
            innovation = z - measurement_matrix @ self.state
        else:
            measurement_matrix = zeros((5, self.state_dim))
            measurement_matrix[0, 0] = 1.0
            measurement_matrix[1, 1] = 1.0
            measurement_matrix[2, 3] = 1.0
            measurement_matrix[3, 4] = 1.0
            measurement_matrix[4, 5] = 1.0
            innovation = z - measurement_matrix @ self.state
            innovation = array(
                [
                    innovation[0],
                    innovation[1],
                    self._wrap_half_turn(innovation[2]),
                    innovation[3],
                    innovation[4],
                ]
            )

        innovation_covariance = self._regularized_covariance(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + reduction_covariance
        )
        cross_covariance = self.covariance @ measurement_matrix.T
        gain = self._gain_from_cross_covariance(
            cross_covariance,
            innovation_covariance,
        )
        self.state = self.state + gain @ innovation
        self.covariance = self._regularized_covariance(
            self.covariance - gain @ innovation_covariance @ gain.T
        )
        self._canonicalize_axes()

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    @property
    def lambda_direction(self):
        return array([cos(self.state[3]), sin(self.state[3])])

    @property
    def omicron_direction(self):
        return array([-sin(self.state[3]), cos(self.state[3])])

    @property
    def extent(self):
        return self.get_point_estimate_extent()

    def get_point_estimate(self):
        return self.state

    def get_point_estimate_kinematics(self):
        return self.state[:4]

    def get_point_estimate_shape(self, full_axes=False):
        axes = self.state[4:6]
        if full_axes:
            axes = 2.0 * axes
        return concatenate([array([self.state[3]]), axes])

    def get_point_estimate_extent(self, flatten_matrix=False):
        rotation_matrix = self._rotation_matrix(self.state[3])
        axes = diag(self.state[4:6])
        extent = self._symmetrize(rotation_matrix @ axes @ axes @ rotation_matrix.T)
        if flatten_matrix:
            return extent.flatten()
        return extent

    def get_contour_points(self, n, scaling_factor=1.0):
        if n <= 0:
            raise ValueError("n must be positive")
        angles = linspace(0.0, 2.0 * pi, n, endpoint=False)
        unit_circle = array([cos(angles), sin(angles)])
        transform = self._rotation_matrix(self.state[3]) @ diag(self.state[4:6])
        contour_points = self.state[:2].reshape((2, 1)) + scaling_factor * (
            transform @ unit_circle
        )
        return contour_points.T


LomemTracker = LOMEMTracker
