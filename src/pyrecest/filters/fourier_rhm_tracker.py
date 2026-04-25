from __future__ import annotations

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin,duplicate-code
from pyrecest.backend import (
    arctan2,
    array,
    concatenate,
    cos,
    diag,
    eye,
    linalg,
    linspace,
    pi,
    reshape,
    sin,
    stack,
    zeros,
    zeros_like,
)

from ._ukf import MerweScaledSigmaPoints
from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


def _pol2cart(phi, radius=1.0):
    return radius * stack((cos(phi), sin(phi)))


class FourierRHMTracker(
    AbstractExtendedObjectTracker
):  # pylint: disable=too-many-instance-attributes
    """Star-convex Random Hypersurface Model with Fourier coefficients.

    The extent is represented by a radial function
    ``r(phi) = b0 / 2 + sum_k a_k cos(k phi) + c_k sin(k phi)``.
    Measurements are processed with the squared RHM pseudo-measurement from
    Baum and Hanebeck's star-convex RHM, using an augmented UKF over the shape
    and position state, the random scale variable, and additive measurement
    noise.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        n_harmonics,
        fourier_coefficients=None,
        kinematic_state=None,
        covariance=None,
        coefficient_covariance=0.02,
        kinematic_covariance=0.3,
        initial_radius=1.0,
        scale_mean=0.7,
        scale_variance=0.08,
        ukf_alpha=1.0,
        ukf_beta=0.0,
        ukf_kappa=0.0,
        covariance_regularization=1e-9,
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
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError(
                "FourierRHMTracker is not supported on the JAX backend"
            )

        self.n_harmonics = int(n_harmonics)
        if self.n_harmonics < 0:
            raise ValueError("n_harmonics must be non-negative")
        self.n_fourier_coefficients = 2 * self.n_harmonics + 1
        self.state_dim = self.n_fourier_coefficients + 2

        if fourier_coefficients is None:
            fourier_coefficients = zeros(self.n_fourier_coefficients)
            fourier_coefficients[0] = 2.0 * float(initial_radius)
        self.fourier_coefficients = self._as_vector(
            fourier_coefficients,
            self.n_fourier_coefficients,
            "fourier_coefficients",
        )

        if kinematic_state is None:
            kinematic_state = zeros(2)
        self.kinematic_state = self._as_vector(kinematic_state, 2, "kinematic_state")

        if covariance is None:
            covariance = linalg.block_diag(
                self._as_square_matrix(
                    coefficient_covariance,
                    self.n_fourier_coefficients,
                    "coefficient_covariance",
                ),
                self._as_square_matrix(
                    kinematic_covariance,
                    2,
                    "kinematic_covariance",
                ),
            )
        self.covariance = self._as_square_matrix(
            covariance, self.state_dim, "covariance"
        )
        self._validate_positive_definite(
            self.covariance + covariance_regularization * eye(self.state_dim),
            "covariance",
        )

        self.scale_mean = float(scale_mean)
        self.scale_variance = float(scale_variance)
        if self.scale_variance < 0.0:
            raise ValueError("scale_variance must be non-negative")

        self.ukf_alpha = float(ukf_alpha)
        self.ukf_beta = float(ukf_beta)
        self.ukf_kappa = float(ukf_kappa)
        self.covariance_regularization = float(covariance_regularization)
        if self.covariance_regularization < 0.0:
            raise ValueError("covariance_regularization must be non-negative")

        self.latest_pseudo_measurement = None
        self.latest_innovation_covariance = None

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _validate_positive_definite(matrix, name):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
        linalg.cholesky(matrix)

    @staticmethod
    def _as_vector(value, dim, name):
        vector = array(value).reshape(-1)
        if vector.shape != (dim,):
            raise ValueError(f"{name} must have shape ({dim},)")
        return vector

    @classmethod
    def _as_square_matrix(cls, value, dim, name):
        matrix = array(value)
        if matrix.ndim == 0:
            matrix = matrix * eye(dim)
        elif matrix.ndim == 1:
            if matrix.shape != (dim,):
                raise ValueError(f"{name} vector must have shape ({dim},)")
            matrix = diag(matrix)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        return cls._symmetrize(matrix)

    @staticmethod
    def _normalize_measurements(measurements):
        measurements = array(measurements)
        if measurements.ndim == 1:
            if measurements.shape[0] != 2:
                raise ValueError("A single measurement vector must have shape (2,)")
            return reshape(measurements, (2, 1))
        if measurements.ndim != 2:
            raise ValueError("measurements must be a vector or a two-dimensional array")
        if measurements.shape[0] == 2:
            return measurements
        if measurements.shape[1] == 2:
            return measurements.T
        raise ValueError("measurements must have shape (2, n) or (n, 2)")

    def _state_vector(self):
        return concatenate([self.fourier_coefficients, self.kinematic_state])

    def _set_state_vector(self, state):
        state = self._as_vector(state, self.state_dim, "state")
        self.fourier_coefficients = state[: self.n_fourier_coefficients]
        self.kinematic_state = state[self.n_fourier_coefficients :]

    def fourier_basis(self, phi):
        """Return the Fourier basis vector or basis matrix for angle ``phi``."""
        phi = array(phi)
        basis = [0.5 + zeros_like(phi)]
        for harmonic in range(1, self.n_harmonics + 1):
            basis.extend([cos(harmonic * phi), sin(harmonic * phi)])
        return array(basis)

    def evaluate_radius(self, phi):
        """Evaluate the star-convex radial function at one or more angles."""
        basis = self.fourier_basis(phi)
        if basis.ndim == 1:
            return basis @ self.fourier_coefficients
        return self.fourier_coefficients @ basis

    def get_point_estimate(self):
        return self._state_vector()

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        if flatten_matrix:
            return self.fourier_coefficients.flatten()
        return self.fourier_coefficients

    def get_extents_on_grid(self, n=100):
        angles = linspace(0.0, 2.0 * pi, n, endpoint=False)
        return self.evaluate_radius(angles)

    def get_contour_points(self, n=100):
        angles = linspace(0.0, 2.0 * pi, n, endpoint=False)
        radii = self.evaluate_radius(angles)
        contour_points = self.kinematic_state[:, None] + _pol2cart(angles, radii)
        return contour_points.T

    def predict_identity(self, sys_noise=None):
        if sys_noise is None:
            sys_noise = zeros((self.state_dim, self.state_dim))
        self.covariance = self._symmetrize(
            self.covariance
            + self._as_square_matrix(sys_noise, self.state_dim, "sys_noise")
        )
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict_linear(self, system_matrix, sys_noise=None, inputs=None):
        system_matrix = array(system_matrix)
        if system_matrix.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"system_matrix must have shape ({self.state_dim}, {self.state_dim})"
            )
        state = system_matrix @ self._state_vector()
        if inputs is not None:
            state = state + self._as_vector(inputs, self.state_dim, "inputs")
        self._set_state_vector(state)
        if sys_noise is None:
            sys_noise = zeros((self.state_dim, self.state_dim))
        self.covariance = self._symmetrize(
            system_matrix @ self.covariance @ system_matrix.T
            + self._as_square_matrix(sys_noise, self.state_dim, "sys_noise")
        )
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict(self, *args, **kwargs):
        if args or kwargs:
            self.predict_linear(*args, **kwargs)
        else:
            self.predict_identity()

    def _pseudo_measurement(self, state, noise, measurement):
        coefficients = state[: self.n_fourier_coefficients]
        center = state[self.n_fourier_coefficients :]
        scale = noise[0]
        measurement_noise = noise[1:]

        direction = measurement - center
        theta = arctan2(direction[1], direction[0]) % (2.0 * pi)
        basis = self.fourier_basis(theta)
        radius = basis @ coefficients
        unit_direction = _pol2cart(theta)

        squared_range = linalg.norm(center - measurement) ** 2
        squared_source_range = (
            scale**2 * radius**2
            + 2.0 * scale * radius * (unit_direction @ measurement_noise)
            + linalg.norm(measurement_noise) ** 2
        )
        return squared_range - squared_source_range

    # pylint: disable=too-many-locals
    def _update_single(self, measurement, meas_noise_cov, scale_mean, scale_variance):
        state = self._state_vector()
        noise_mean = array([scale_mean, 0.0, 0.0])
        noise_covariance = linalg.block_diag(
            array([[scale_variance]]),
            self._as_square_matrix(meas_noise_cov, 2, "meas_noise_cov"),
        )
        augmented_mean = concatenate([state, noise_mean])
        augmented_covariance = linalg.block_diag(self.covariance, noise_covariance)
        augmented_dim = augmented_mean.shape[0]
        augmented_covariance = self._symmetrize(
            augmented_covariance + self.covariance_regularization * eye(augmented_dim)
        )

        sigma_points = MerweScaledSigmaPoints(
            augmented_dim,
            alpha=self.ukf_alpha,
            beta=self.ukf_beta,
            kappa=self.ukf_kappa,
        )
        sigmas = sigma_points.sigma_points(augmented_mean, augmented_covariance)
        pseudo_sigmas = zeros(sigmas.shape[0])
        for sigma_index in range(sigmas.shape[0]):
            pseudo_sigmas[sigma_index] = self._pseudo_measurement(
                sigmas[sigma_index, : self.state_dim],
                sigmas[sigma_index, self.state_dim :],
                measurement,
            )

        predicted_pseudo = 0.0
        for sigma_index in range(sigmas.shape[0]):
            predicted_pseudo = (
                predicted_pseudo
                + sigma_points.Wm[sigma_index] * pseudo_sigmas[sigma_index]
            )

        innovation_variance = self.covariance_regularization
        cross_covariance = zeros(self.state_dim)
        for sigma_index in range(sigmas.shape[0]):
            pseudo_delta = pseudo_sigmas[sigma_index] - predicted_pseudo
            state_delta = sigmas[sigma_index, : self.state_dim] - state
            innovation_variance = (
                innovation_variance + sigma_points.Wc[sigma_index] * pseudo_delta**2
            )
            cross_covariance = (
                cross_covariance
                + sigma_points.Wc[sigma_index] * state_delta * pseudo_delta
            )

        kalman_gain = cross_covariance / innovation_variance
        posterior_state = state - kalman_gain * predicted_pseudo
        posterior_covariance = self.covariance - innovation_variance * (
            kalman_gain[:, None] @ kalman_gain[None, :]
        )

        self._set_state_vector(posterior_state)
        self.covariance = self._symmetrize(posterior_covariance)
        self.latest_pseudo_measurement = predicted_pseudo
        self.latest_innovation_covariance = innovation_variance

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def update(
        self,
        measurements,
        meas_noise_cov,
        scale_mean=None,
        scale_variance=None,
    ):
        """Update from one or more 2-D measurements.

        ``measurements`` may be a single vector, a ``(2, n)`` matrix, or a
        ``(n, 2)`` matrix. Each column/row is processed as one RHM contour or
        interior-source observation.
        """
        if scale_mean is None:
            scale_mean = self.scale_mean
        if scale_variance is None:
            scale_variance = self.scale_variance
        if scale_variance < 0.0:
            raise ValueError("scale_variance must be non-negative")

        measurements = self._normalize_measurements(measurements)
        for measurement_index in range(measurements.shape[1]):
            self._update_single(
                measurements[:, measurement_index],
                meas_noise_cov,
                float(scale_mean),
                float(scale_variance),
            )

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()
