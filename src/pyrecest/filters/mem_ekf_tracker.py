from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,too-many-instance-attributes,duplicate-code
from pyrecest.backend import (
    array,
    concatenate,
    cos,
    diag,
    eye,
    linalg,
    linspace,
    pi,
    sin,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class MEMEKFTracker(AbstractExtendedObjectTracker):
    """Multiplicative-error-model EKF for one 2-D elliptical extended object.

    The shape state is ``[orientation, semi_axis_1, semi_axis_2]``. The
    measurement model is ``z = H x + S(p) h + v``, where ``S(p)`` rotates and
    scales the unit multiplicative error ``h`` into the ellipse and ``v`` is
    additive measurement noise.
    """

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
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.measurement_dim = 2
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
        self.shape_state = array(shape_state)
        self._validate_shape_state(self.shape_state)
        self.shape_covariance = self._as_covariance_matrix(
            shape_covariance,
            3,
            "shape_covariance",
        )

        if multiplicative_noise_cov is None:
            multiplicative_noise_cov = 0.25 * eye(self.measurement_dim)
        self.multiplicative_noise_cov = self._as_covariance_matrix(
            multiplicative_noise_cov,
            self.measurement_dim,
            "multiplicative_noise_cov",
        )
        self._validate_diagonal_multiplicative_noise(self.multiplicative_noise_cov)

        self.measurement_matrix = None
        if measurement_matrix is not None:
            self.measurement_matrix = array(measurement_matrix)
            self._validate_measurement_matrix(self.measurement_matrix)

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
    def _validate_shape_state(shape_state):
        if shape_state.shape != (3,):
            raise ValueError("shape_state must have shape (3,)")
        if float(shape_state[1]) <= 0.0 or float(shape_state[2]) <= 0.0:
            raise ValueError("shape semi-axis lengths must be positive")

    @staticmethod
    def _validate_diagonal_multiplicative_noise(multiplicative_noise_cov):
        if (
            abs(float(multiplicative_noise_cov[0, 1])) > 1e-12
            or abs(float(multiplicative_noise_cov[1, 0])) > 1e-12
        ):
            raise ValueError("multiplicative_noise_cov must be diagonal")

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

    def _get_measurement_noise(self, meas_noise_cov):
        if meas_noise_cov is None:
            return zeros((self.measurement_dim, self.measurement_dim))
        return self._as_covariance_matrix(
            meas_noise_cov,
            self.measurement_dim,
            "meas_noise_cov",
            require_positive_definite=False,
        )

    def _get_multiplicative_noise_cov(self, multiplicative_noise_cov=None):
        if multiplicative_noise_cov is None:
            return self.multiplicative_noise_cov
        multiplicative_noise_cov = self._as_covariance_matrix(
            multiplicative_noise_cov,
            self.measurement_dim,
            "multiplicative_noise_cov",
        )
        self._validate_diagonal_multiplicative_noise(multiplicative_noise_cov)
        return multiplicative_noise_cov

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
            "measurements must have shape (2, n_measurements) or " "(n_measurements, 2)"
        )

    def _extent_transform(self):
        orientation, semi_axis_1, semi_axis_2 = self.shape_state
        rotation_matrix = array(
            [
                [cos(orientation), -sin(orientation)],
                [sin(orientation), cos(orientation)],
            ]
        )
        return rotation_matrix @ diag(array([semi_axis_1, semi_axis_2]))

    def _shape_pseudo_jacobian(self, multiplicative_noise_cov):
        orientation, semi_axis_1, semi_axis_2 = self.shape_state
        variance_1 = multiplicative_noise_cov[0, 0]
        variance_2 = multiplicative_noise_cov[1, 1]
        scale_difference = semi_axis_1**2 * variance_1 - semi_axis_2**2 * variance_2
        base_matrix = array(
            [
                [-sin(2.0 * orientation), cos(orientation) ** 2, sin(orientation) ** 2],
                [
                    cos(2.0 * orientation),
                    sin(2.0 * orientation),
                    -sin(2.0 * orientation),
                ],
                [sin(2.0 * orientation), sin(orientation) ** 2, cos(orientation) ** 2],
            ]
        )
        scales = diag(
            array(
                [
                    scale_difference,
                    2.0 * semi_axis_1 * variance_1,
                    2.0 * semi_axis_2 * variance_2,
                ]
            )
        )
        return base_matrix @ scales

    @property
    def extent(self):
        return self.get_point_estimate_extent()

    def get_point_estimate(self):
        return concatenate([self.kinematic_state, self.shape_state])

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_shape(self):
        return self.shape_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        extent_transform = self._extent_transform()
        extent = self._symmetrize(extent_transform @ extent_transform.T)
        if flatten_matrix:
            return extent.flatten()
        return extent

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_system_matrix=None,
        shape_sys_noise=None,
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

        if shape_system_matrix is None:
            shape_system_matrix = eye(3)
        else:
            shape_system_matrix = array(shape_system_matrix)
            if shape_system_matrix.shape != (3, 3):
                raise ValueError("shape_system_matrix must have shape (3, 3)")
        if shape_sys_noise is None:
            shape_sys_noise = zeros((3, 3))
        else:
            shape_sys_noise = self._as_covariance_matrix(
                shape_sys_noise,
                3,
                "shape_sys_noise",
                require_positive_definite=False,
            )

        self.shape_state = shape_system_matrix @ self.shape_state
        self._validate_shape_state(self.shape_state)
        self.shape_covariance = self._symmetrize(
            shape_system_matrix @ self.shape_covariance @ shape_system_matrix.T
            + shape_sys_noise
        )

        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

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
        """Sequentially update from one or more 2-D target-originated measurements."""
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[1] == 0:
            return
        measurement_matrix = self._get_measurement_matrix(meas_mat)
        meas_noise_cov = self._get_measurement_noise(meas_noise_cov)
        multiplicative_noise_cov = self._get_multiplicative_noise_cov(
            multiplicative_noise_cov
        )

        for measurement_index in range(measurements.shape[1]):
            self._update_single_measurement(
                measurements[:, measurement_index],
                measurement_matrix,
                meas_noise_cov,
                multiplicative_noise_cov,
            )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    # pylint: disable=too-many-locals
    def _update_single_measurement(
        self,
        measurement,
        measurement_matrix,
        meas_noise_cov,
        multiplicative_noise_cov,
    ):
        extent_transform = self._extent_transform()
        shape_pseudo_jacobian = self._shape_pseudo_jacobian(multiplicative_noise_cov)

        predicted_measurement = measurement_matrix @ self.kinematic_state
        innovation = measurement - predicted_measurement
        innovation_covariance = self._symmetrize(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + extent_transform @ multiplicative_noise_cov @ extent_transform.T
            + meas_noise_cov
        )
        if self.covariance_regularization > 0.0:
            innovation_covariance = innovation_covariance + (
                self.covariance_regularization * eye(self.measurement_dim)
            )

        kinematic_cross_covariance = self.covariance @ measurement_matrix.T
        kinematic_gain = linalg.solve(
            innovation_covariance.T,
            kinematic_cross_covariance.T,
        ).T
        self.kinematic_state = self.kinematic_state + kinematic_gain @ innovation
        self.covariance = self._symmetrize(
            self.covariance - kinematic_gain @ innovation_covariance @ kinematic_gain.T
        )

        shifted_measurement = innovation
        pseudo_measurement = array(
            [
                shifted_measurement[0] ** 2,
                shifted_measurement[0] * shifted_measurement[1],
                shifted_measurement[1] ** 2,
            ]
        )
        sigma_11 = innovation_covariance[0, 0]
        sigma_12 = innovation_covariance[0, 1]
        sigma_22 = innovation_covariance[1, 1]
        pseudo_mean = array([sigma_11, sigma_12, sigma_22])
        pseudo_covariance = array(
            [
                [
                    3.0 * sigma_11**2,
                    3.0 * sigma_11 * sigma_12,
                    sigma_11 * sigma_22 + 2.0 * sigma_12**2,
                ],
                [
                    3.0 * sigma_11 * sigma_12,
                    sigma_11 * sigma_22 + 2.0 * sigma_12**2,
                    3.0 * sigma_22 * sigma_12,
                ],
                [
                    sigma_11 * sigma_22 + 2.0 * sigma_12**2,
                    3.0 * sigma_22 * sigma_12,
                    3.0 * sigma_22**2,
                ],
            ]
        )
        if self.covariance_regularization > 0.0:
            pseudo_covariance = pseudo_covariance + (
                self.covariance_regularization * eye(3)
            )

        shape_cross_covariance = self.shape_covariance @ shape_pseudo_jacobian.T
        shape_gain = linalg.solve(pseudo_covariance.T, shape_cross_covariance.T).T
        self.shape_state = self.shape_state + shape_gain @ (
            pseudo_measurement - pseudo_mean
        )
        self.shape_covariance = self._symmetrize(
            self.shape_covariance - shape_gain @ pseudo_covariance @ shape_gain.T
        )

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


MemEkfTracker = MEMEKFTracker
