from __future__ import annotations

# pylint: disable=no-name-in-module,no-member,redefined-builtin,duplicate-code
from pyrecest.backend import (
    array,
    concatenate,
    cos,
    eye,
    gammaln,
    linalg,
    linspace,
    log,
    mean,
    pi,
    sin,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class GGIWTracker(
    AbstractExtendedObjectTracker
):  # pylint: disable=too-many-instance-attributes
    """Gamma-Gaussian-inverse-Wishart tracker for one extended object.

    The posterior is represented by a Gaussian kinematic state, an inverse-Wishart
    extent model, and a Gamma model for the expected number of detections per
    scan. The public extent estimate follows the same ellipse convention used in
    several extended-object tracking references: ``E[X] = V / (nu - 2d - 2)``,
    where ``V`` is the inverse-Wishart scale matrix and ``d`` is the measurement
    dimension.

    The update uses a centroid Kalman step for the kinematics, a scatter update
    for the extent sufficient statistics, and a conjugate Gamma-Poisson count
    update. This keeps the implementation compatible with PyRecEst's existing
    full covariance state representation instead of assuming a separable
    kinematic/extent covariance.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        kinematic_state,
        covariance,
        extent,
        extent_degrees_of_freedom,
        gamma_shape,
        gamma_rate,
        measurement_matrix=None,
        extent_is_scale=False,
        extent_innovation_weight=1.0,
        subtract_measurement_noise_from_scatter=False,
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
        self.covariance = self._as_covariance_matrix(
            covariance,
            self.kinematic_state.shape[0],
            "covariance",
        )

        extent = array(extent)
        self.measurement_dim = extent.shape[0]
        self.extent_degrees_of_freedom = float(extent_degrees_of_freedom)
        self.gamma_shape = float(gamma_shape)
        self.gamma_rate = float(gamma_rate)
        self.extent_innovation_weight = float(extent_innovation_weight)
        self.subtract_measurement_noise_from_scatter = bool(
            subtract_measurement_noise_from_scatter
        )
        self.latest_log_likelihood = None

        if self.gamma_shape <= 0.0 or self.gamma_rate <= 0.0:
            raise ValueError("gamma_shape and gamma_rate must be positive")
        if self.extent_innovation_weight < 0.0:
            raise ValueError("extent_innovation_weight must be non-negative")

        self.measurement_matrix = None
        if measurement_matrix is not None:
            self.measurement_matrix = array(measurement_matrix)
            self._validate_measurement_matrix(self.measurement_matrix)

        self.extent_scale = self._symmetrize(extent)
        if not extent_is_scale:
            self.extent_scale = self.extent_scale * self._extent_mean_denominator()
        self._validate_positive_definite(self.extent_scale, "extent_scale")

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _validate_positive_definite(matrix, name):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
        linalg.cholesky(matrix)

    @classmethod
    def _as_covariance_matrix(cls, value, dim, name):
        matrix = array(value)
        if matrix.ndim == 0:
            matrix = matrix * eye(dim)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        matrix = cls._symmetrize(matrix)
        cls._validate_positive_definite(matrix, name)
        return matrix

    def _extent_mean_denominator(self, degrees_of_freedom=None):
        if degrees_of_freedom is None:
            degrees_of_freedom = self.extent_degrees_of_freedom
        denominator = float(degrees_of_freedom) - 2.0 * self.measurement_dim - 2.0
        if denominator <= 0.0:
            raise ValueError(
                "extent_degrees_of_freedom must be larger than 2 * measurement_dim + 2"
            )
        return denominator

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
        if self.kinematic_state.shape[0] < self.measurement_dim:
            raise ValueError(
                "Cannot infer a position measurement matrix because the kinematic state "
                "dimension is smaller than the measurement dimension"
            )
        return eye(self.measurement_dim, self.kinematic_state.shape[0])

    def _normalize_measurements(self, measurements):
        measurements = array(measurements)
        if measurements.ndim == 1:
            if measurements.shape[0] != self.measurement_dim:
                raise ValueError(
                    "A single measurement vector must match the measurement dimension"
                )
            return measurements.reshape((self.measurement_dim, 1))
        if measurements.ndim != 2:
            raise ValueError("measurements must be a vector or a two-dimensional array")
        if measurements.shape[0] == self.measurement_dim:
            return measurements
        if measurements.shape[1] == self.measurement_dim:
            return measurements.T
        raise ValueError(
            "measurements must have shape (measurement_dim, n_measurements) or "
            "(n_measurements, measurement_dim)"
        )

    def _get_measurement_noise(self, meas_noise_cov):
        if meas_noise_cov is None:
            return zeros((self.measurement_dim, self.measurement_dim))
        return self._as_covariance_matrix(
            meas_noise_cov,
            self.measurement_dim,
            "meas_noise_cov",
        )

    def get_measurement_rate_estimate(self):
        """Return the posterior mean of the Poisson measurement rate."""
        return self.gamma_shape / self.gamma_rate

    @property
    def extent(self):
        """Return the posterior mean extent matrix."""
        return self.get_point_estimate_extent()

    def get_point_estimate(self):
        return concatenate(
            [
                self.kinematic_state,
                self.get_point_estimate_extent(flatten_matrix=True),
                array([self.get_measurement_rate_estimate()]),
            ]
        )

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        extent = self._symmetrize(self.extent_scale / self._extent_mean_denominator())
        if flatten_matrix:
            return extent.flatten()
        return extent

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def predict_linear(
        self,
        system_matrix,
        sys_noise,
        inputs=None,
        extent_forgetting_factor=1.0,
        measurement_rate_forgetting_factor=1.0,
    ):
        """Predict with a linear kinematic model and optional information decay."""
        if not 0.0 < extent_forgetting_factor <= 1.0:
            raise ValueError("extent_forgetting_factor must be in (0, 1]")
        if not 0.0 < measurement_rate_forgetting_factor <= 1.0:
            raise ValueError("measurement_rate_forgetting_factor must be in (0, 1]")

        system_matrix = array(system_matrix)
        if system_matrix.shape != (
            self.kinematic_state.shape[0],
            self.kinematic_state.shape[0],
        ):
            raise ValueError(
                "system_matrix shape must match the kinematic state dimension"
            )
        sys_noise = self._as_covariance_matrix(
            sys_noise,
            self.kinematic_state.shape[0],
            "sys_noise",
        )

        self.kinematic_state = system_matrix @ self.kinematic_state
        if inputs is not None:
            self.kinematic_state = self.kinematic_state + array(inputs)
        self.covariance = self._symmetrize(
            system_matrix @ self.covariance @ system_matrix.T + sys_noise
        )

        extent_mean = self.get_point_estimate_extent()
        offset = 2.0 * self.measurement_dim + 2.0
        extent_information = self.extent_degrees_of_freedom - offset
        self.extent_degrees_of_freedom = (
            offset + extent_forgetting_factor * extent_information
        )
        self.extent_scale = extent_mean * self._extent_mean_denominator()

        self.gamma_shape *= measurement_rate_forgetting_factor
        self.gamma_rate *= measurement_rate_forgetting_factor

        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict(self, *args, **kwargs):
        """Alias for :meth:`predict_linear` to match existing EOT tracker APIs."""
        self.predict_linear(*args, **kwargs)

    def _gamma_poisson_log_predictive(self, count, gamma_shape, gamma_rate):
        count = array(count)
        gamma_shape = array(gamma_shape)
        gamma_rate = array(gamma_rate)
        return (
            gammaln(gamma_shape + count)
            - gammaln(gamma_shape)
            - gammaln(count + 1.0)
            + gamma_shape * log(gamma_rate / (gamma_rate + 1.0))
            + count * log(1.0 / (gamma_rate + 1.0))
        )

    def _gaussian_log_likelihood(self, innovation, covariance):
        determinant = linalg.det(covariance)
        if float(determinant) <= 0.0:
            return float("-inf")
        mahalanobis_distance = innovation.T @ linalg.solve(covariance, innovation)
        return -0.5 * (
            innovation.shape[0] * log(array(2.0 * pi))
            + log(determinant)
            + mahalanobis_distance
        )

    # pylint: disable=too-many-locals
    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        extent_innovation_weight=None,
    ):
        """Update from all detections generated by the target in one scan."""
        measurements = self._normalize_measurements(measurements)
        n_measurements = measurements.shape[1]
        if n_measurements == 0:
            prior_shape = self.gamma_shape
            prior_rate = self.gamma_rate
            self.gamma_rate += 1.0
            self.latest_log_likelihood = self._gamma_poisson_log_predictive(
                0,
                prior_shape,
                prior_rate,
            )
            return

        measurement_matrix = self._get_measurement_matrix(meas_mat)
        meas_noise_cov = self._get_measurement_noise(meas_noise_cov)
        if extent_innovation_weight is None:
            extent_innovation_weight = self.extent_innovation_weight
        if extent_innovation_weight < 0.0:
            raise ValueError("extent_innovation_weight must be non-negative")

        prior_shape = self.gamma_shape
        prior_rate = self.gamma_rate
        extent_prior = self.get_point_estimate_extent()
        measurement_mean = mean(measurements, axis=1)
        demeaned_measurements = measurements - measurement_mean[:, None]
        measurement_scatter = demeaned_measurements @ demeaned_measurements.T
        if self.subtract_measurement_noise_from_scatter and n_measurements > 1:
            measurement_scatter = (
                measurement_scatter - (n_measurements - 1) * meas_noise_cov
            )

        predicted_measurement_mean = measurement_matrix @ self.kinematic_state
        innovation = measurement_mean - predicted_measurement_mean
        centroid_covariance = self._symmetrize(
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + (extent_prior + meas_noise_cov) / n_measurements
        )
        cross_covariance = self.covariance @ measurement_matrix.T
        kalman_gain = linalg.solve(centroid_covariance.T, cross_covariance.T).T

        self.kinematic_state = self.kinematic_state + kalman_gain @ innovation
        self.covariance = self._symmetrize(
            self.covariance - kalman_gain @ centroid_covariance @ kalman_gain.T
        )

        innovation_outer = innovation[:, None] @ innovation[None, :]
        innovation_scale = (
            extent_innovation_weight * n_measurements / (n_measurements + 1.0)
        )
        self.extent_scale = self._symmetrize(
            self.extent_scale
            + measurement_scatter
            + innovation_scale * innovation_outer
        )
        self.extent_degrees_of_freedom += n_measurements

        self.gamma_shape += n_measurements
        self.gamma_rate += 1.0

        self.latest_log_likelihood = self._gamma_poisson_log_predictive(
            n_measurements,
            prior_shape,
            prior_rate,
        ) + self._gaussian_log_likelihood(innovation, centroid_covariance)

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def get_contour_points(self, n, scaling_factor=1.0):
        measurement_matrix = self._get_measurement_matrix()
        position_estimate = measurement_matrix @ self.kinematic_state
        angles = linspace(0.0, 2.0 * pi, n, endpoint=False)
        unit_circle = array([cos(angles), sin(angles)])
        extent_transform = linalg.cholesky(self.get_point_estimate_extent())
        contour_points = (
            position_estimate[:, None] + scaling_factor * extent_transform @ unit_circle
        )
        return contour_points.T
