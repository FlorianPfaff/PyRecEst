from __future__ import annotations

import numpy as np
import pyrecest.backend

# pylint: disable=no-member,no-name-in-module,too-many-lines
from pyrecest.backend import (
    array,
    concatenate,
    diag,
    eye,
    imag,
    linalg,
    linspace,
    pi,
    real,
    reshape,
    sqrt,
    stack,
    zeros,
)
from pyrecest.distributions.hypersphere_subset.abstract_sphere_subset_distribution import (
    AbstractSphereSubsetDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_real import (
    SphericalHarmonicsDistributionReal,
)
from pyrecest.sampling.sigma_points import MerweScaledSigmaPoints

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class SphericalHarmonicsEOTTracker(
    AbstractExtendedObjectTracker
):  # pylint: disable=too-many-instance-attributes
    """3-D star-convex EOT tracker with spherical-harmonic extent coefficients.

    The state is ``[cx, cy, cz, c_00, c_1,-1, c_1,0, c_1,1, ...]``.  The
    coefficients parameterize an unnormalized radial extent function
    ``r(u) = sum_lm c_lm Y_lm(u)`` rather than a probability density.  This
    matches the spherical-harmonics tracker in the ICRA 2017 MATLAB code.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    def __init__(
        self,
        order,
        coefficients=None,
        center=None,
        covariance=None,
        coefficient_covariance=0.02,
        kinematic_covariance=0.3,
        initial_radius=1.0,
        ukf_alpha=1.0,
        ukf_beta=2.0,
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
        if pyrecest.backend.__backend_name__ != "numpy":  # pylint: disable=no-member
            raise NotImplementedError(
                "SphericalHarmonicsEOTTracker is currently supported only on the "
                "numpy backend"
            )

        self.order = int(order)
        if self.order < 0:
            raise ValueError("order must be non-negative")
        self.n_coefficients = (self.order + 1) ** 2
        self.state_dim = 3 + self.n_coefficients

        if coefficients is None:
            coefficients = zeros(self.n_coefficients)
            coefficients[0] = float(initial_radius) * sqrt(4.0 * pi)
        self.coefficients = self._as_vector(
            coefficients,
            self.n_coefficients,
            "coefficients",
        )

        if center is None:
            center = zeros(3)
        self.center = self._as_vector(center, 3, "center")

        if covariance is None:
            covariance = linalg.block_diag(
                self._as_square_matrix(kinematic_covariance, 3, "kinematic_covariance"),
                self._as_square_matrix(
                    coefficient_covariance,
                    self.n_coefficients,
                    "coefficient_covariance",
                ),
            )
        self.covariance = self._as_square_matrix(
            covariance,
            self.state_dim,
            "covariance",
        )
        self._validate_positive_definite(
            self.covariance + covariance_regularization * eye(self.state_dim),
            "covariance",
        )

        self.ukf_alpha = float(ukf_alpha)
        self.ukf_beta = float(ukf_beta)
        self.ukf_kappa = float(ukf_kappa)
        self.covariance_regularization = float(covariance_regularization)
        if self.covariance_regularization < 0.0:
            raise ValueError("covariance_regularization must be non-negative")

        self.latest_innovation_covariance = None
        self.latest_predicted_measurement = None

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
            if measurements.shape[0] != 3:
                raise ValueError("A single measurement vector must have shape (3,)")
            return reshape(measurements, (3, 1))
        if measurements.ndim != 2:
            raise ValueError("measurements must be a vector or a two-dimensional array")
        if measurements.shape[0] == 3:
            return measurements
        if measurements.shape[1] == 3:
            return measurements.T
        raise ValueError("measurements must have shape (3, n) or (n, 3)")

    @staticmethod
    def _flatten_measurements(measurements):
        return measurements.T.reshape(-1)

    @staticmethod
    def _unit_directions(vectors):
        norms = linalg.norm(vectors, axis=0)
        safe_norms = array(norms)
        near_zero = safe_norms <= 1e-12
        safe_norms[near_zero] = 1.0
        directions = vectors / safe_norms
        if any(near_zero):
            directions[:, near_zero] = array([[1.0], [0.0], [0.0]])
        return directions

    @staticmethod
    def coefficients_to_matrix(coefficients):
        """Convert packed real SH coefficients to PyRecEst's coefficient matrix."""
        coefficients = array(coefficients).reshape(-1)
        order_float = sqrt(coefficients.shape[0]) - 1
        order = int(round(float(order_float)))
        if (order + 1) ** 2 != coefficients.shape[0]:
            raise ValueError("Number of coefficients must be a square")

        coeff_mat = zeros((order + 1, 2 * order + 1))
        index = 0
        for degree in range(order + 1):
            count = 2 * degree + 1
            coeff_mat[degree, :count] = coefficients[index : index + count]
            index += count
        return coeff_mat

    @staticmethod
    def matrix_to_coefficients(coeff_mat):
        """Pack PyRecEst's real SH coefficient matrix degree by degree."""
        coeff_mat = array(coeff_mat)
        coefficients = []
        for degree in range(coeff_mat.shape[0]):
            coefficients.extend(coeff_mat[degree, : 2 * degree + 1])
        return array(coefficients)

    @staticmethod
    def _real_coeff_mat_to_complex(real_coeff_mat):
        real_coeff_mat = array(real_coeff_mat)
        complex_coeff_mat = zeros(real_coeff_mat.shape, dtype=complex)
        complex_coeff_mat[0, 0] = real_coeff_mat[0, 0]

        for degree in range(1, real_coeff_mat.shape[0]):
            for order in range(-degree, degree + 1):
                if order < 0:
                    complex_coeff_mat[degree, degree + order] = (
                        1j * real_coeff_mat[degree, degree + order]
                        + real_coeff_mat[degree, degree - order]
                    ) / sqrt(2.0)
                elif order > 0:
                    complex_coeff_mat[degree, degree + order] = (
                        (-1) ** order
                        * (
                            -1j * real_coeff_mat[degree, degree - order]
                            + real_coeff_mat[degree, degree + order]
                        )
                        / sqrt(2.0)
                    )
                else:
                    complex_coeff_mat[degree, degree] = real_coeff_mat[degree, degree]
        return complex_coeff_mat

    @staticmethod
    def _complex_coeff_mat_to_real(complex_coeff_mat):
        complex_coeff_mat = array(complex_coeff_mat)
        real_coeff_mat = zeros(complex_coeff_mat.shape)
        real_coeff_mat[0, 0] = real(complex_coeff_mat[0, 0])

        for degree in range(1, complex_coeff_mat.shape[0]):
            for order in range(-degree, degree + 1):
                if order < 0:
                    real_coeff_mat[degree, degree + order] = (
                        (-1) ** order
                        * sqrt(2.0)
                        * (-1 if (-order) % 2 else 1)
                        * imag(complex_coeff_mat[degree, degree + order])
                    )
                elif order > 0:
                    real_coeff_mat[degree, degree + order] = (
                        sqrt(2.0)
                        * (-1 if order % 2 else 1)
                        * real(complex_coeff_mat[degree, degree + order])
                    )
                else:
                    real_coeff_mat[degree, degree] = real(
                        complex_coeff_mat[degree, degree]
                    )
        return real_coeff_mat

    @staticmethod
    def rotate_coefficients(coefficients, alpha, beta=0.0, gamma=0.0):
        """Rotate packed real SH coefficients by ZYZ Euler angles in radians."""
        if alpha == 0.0 and beta == 0.0 and gamma == 0.0:
            return array(coefficients).reshape(-1)

        coeff_mat_real = SphericalHarmonicsEOTTracker.coefficients_to_matrix(
            coefficients
        )
        degree = coeff_mat_real.shape[0] - 1
        coeff_mat_complex = SphericalHarmonicsEOTTracker._real_coeff_mat_to_complex(
            coeff_mat_real
        )
        clm = SphericalHarmonicsDistributionComplex._coeff_mat_to_pysh(  # pylint: disable=protected-access
            coeff_mat_complex, degree
        )
        clm_rot = clm.rotate(
            alpha * 180.0 / pi,
            beta * 180.0 / pi,
            gamma * 180.0 / pi,
            degrees=True,
            body=True,
        )
        coeff_mat_complex_rot = SphericalHarmonicsDistributionComplex._pysh_to_coeff_mat(  # pylint: disable=protected-access
            clm_rot, degree
        )
        coeff_mat_real_rot = SphericalHarmonicsEOTTracker._complex_coeff_mat_to_real(
            coeff_mat_complex_rot
        )
        return SphericalHarmonicsEOTTracker.matrix_to_coefficients(coeff_mat_real_rot)

    @staticmethod
    def evaluate_radius_from_coefficients(coefficients, directions):
        """Evaluate the raw radial SH extent for Cartesian unit directions."""
        directions = SphericalHarmonicsEOTTracker._normalize_measurements(directions)
        directions = SphericalHarmonicsEOTTracker._unit_directions(directions)
        coeff_mat = SphericalHarmonicsEOTTracker.coefficients_to_matrix(coefficients)
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            directions[0],
            directions[1],
            directions[2],
        )

        radii = zeros(directions.shape[1])
        for degree in range(coeff_mat.shape[0]):
            for order in range(-degree, degree + 1):
                basis_values = SphericalHarmonicsDistributionReal.real_spherical_harmonic_basis_function(
                    degree,
                    order,
                    theta,
                    phi,
                )
                radii += coeff_mat[degree, degree + order] * basis_values
        return radii

    def _state_vector(self):
        return concatenate([self.center, self.coefficients])

    def _set_state_vector(self, state):
        state = self._as_vector(state, self.state_dim, "state")
        self.center = state[:3]
        self.coefficients = state[3:]

    def evaluate_radius(self, directions):
        """Evaluate the current radial extent for Cartesian directions."""
        return self.evaluate_radius_from_coefficients(self.coefficients, directions)

    def surface_points_for_directions(self, directions, center=None, coefficients=None):
        """Return object surface points for Cartesian rays from *center*."""
        directions = self._normalize_measurements(directions)
        unit_directions = self._unit_directions(directions)
        if center is None:
            center = self.center
        if coefficients is None:
            coefficients = self.coefficients
        radii = self.evaluate_radius_from_coefficients(coefficients, unit_directions)
        return center.reshape(3, 1) + unit_directions * radii

    def measurement_function(self, state, measurements):
        """MATLAB-equivalent stacked point measurement equation."""
        measurements = self._normalize_measurements(measurements)
        state = self._as_vector(state, self.state_dim, "state")
        center = state[:3]
        coefficients = state[3:]
        local_measurements = measurements - center.reshape(3, 1)
        predicted_points = self.surface_points_for_directions(
            local_measurements,
            center,
            coefficients,
        )
        return self._flatten_measurements(predicted_points)

    def get_point_estimate(self):
        return self._state_vector()

    def get_point_estimate_kinematics(self):
        return self.center

    def get_point_estimate_extent(self, flatten_matrix=False):
        if flatten_matrix:
            return self.coefficients.flatten()
        return self.coefficients

    def get_extents_on_grid(self, n=100):
        azimuth = linspace(0.0, 2.0 * pi, n, endpoint=False)
        elevation = linspace(-0.5 * pi, 0.5 * pi, n)
        az_grid, el_grid = np.meshgrid(azimuth, elevation)
        x = np.cos(el_grid) * np.cos(az_grid)
        y = np.cos(el_grid) * np.sin(az_grid)
        z = np.sin(el_grid)
        directions = stack(
            [array(x).reshape(-1), array(y).reshape(-1), array(z).reshape(-1)]
        )
        return reshape(self.evaluate_radius(directions), az_grid.shape)

    def get_contour_points(self, n=100):
        azimuth = linspace(0.0, 2.0 * pi, n, endpoint=False)
        elevation = linspace(-0.5 * pi, 0.5 * pi, n)
        az_grid, el_grid = np.meshgrid(azimuth, elevation)
        x = np.cos(el_grid) * np.cos(az_grid)
        y = np.cos(el_grid) * np.sin(az_grid)
        z = np.sin(el_grid)
        directions = stack(
            [array(x).reshape(-1), array(y).reshape(-1), array(z).reshape(-1)]
        )
        return self.surface_points_for_directions(directions).T

    def _sigma_points(self, mean, covariance):
        covariance = self._symmetrize(
            covariance + self.covariance_regularization * eye(mean.shape[0])
        )
        points = MerweScaledSigmaPoints(
            mean.shape[0],
            alpha=self.ukf_alpha,
            beta=self.ukf_beta,
            kappa=self.ukf_kappa,
        )
        return points, points.sigma_points(mean, covariance)

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

    def predict_nonlinear(self, transition_function, sys_noise=None):
        state = self._state_vector()
        points, sigmas = self._sigma_points(state, self.covariance)
        propagated = zeros(sigmas.shape)
        for sigma_index in range(sigmas.shape[0]):
            propagated[sigma_index] = self._as_vector(
                transition_function(sigmas[sigma_index]),
                self.state_dim,
                "transition result",
            )

        predicted_state = zeros(self.state_dim)
        for sigma_index in range(sigmas.shape[0]):
            predicted_state += points.Wm[sigma_index] * propagated[sigma_index]

        predicted_covariance = zeros((self.state_dim, self.state_dim))
        for sigma_index in range(sigmas.shape[0]):
            delta = propagated[sigma_index] - predicted_state
            predicted_covariance += points.Wc[sigma_index] * (
                delta[:, None] @ delta[None, :]
            )
        if sys_noise is None:
            sys_noise = zeros((self.state_dim, self.state_dim))
        predicted_covariance += self._as_square_matrix(
            sys_noise,
            self.state_dim,
            "sys_noise",
        )

        self._set_state_vector(predicted_state)
        self.covariance = self._symmetrize(predicted_covariance)
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict_rotation(self, alpha, beta=0.0, gamma=0.0, sys_noise=None):
        def transition(state):
            return concatenate(
                [
                    state[:3],
                    self.rotate_coefficients(state[3:], alpha, beta, gamma),
                ]
            )

        self.predict_nonlinear(transition, sys_noise)

    def predict(self, *args, **kwargs):
        if not args and not kwargs:
            return self.predict_identity()
        if not args and set(kwargs) <= {"sys_noise"}:
            return self.predict_identity(**kwargs)
        if "rotation_angles" in kwargs:
            rotation_angles = kwargs.pop("rotation_angles")
            return self.predict_rotation(*rotation_angles, **kwargs)
        return self.predict_linear(*args, **kwargs)

    # pylint: disable=too-many-locals
    def update(self, measurements, meas_noise_cov):
        """Update from one or more 3-D point measurements.

        The measurement equation is the one used in the MATLAB
        ``SphericalHarmonicsAdditiveMeasmodel``: each observed point defines a
        bearing from the hypothesized center, and the predicted measurement is
        the surface point at the current SH radius along that bearing.
        """
        measurements = self._normalize_measurements(measurements)
        measurement = self._flatten_measurements(measurements)
        measurement_dim = measurement.shape[0]
        meas_noise_cov = self._as_square_matrix(
            meas_noise_cov,
            measurement_dim,
            "meas_noise_cov",
        )

        state = self._state_vector()
        points, sigmas = self._sigma_points(state, self.covariance)
        meas_sigmas = zeros((sigmas.shape[0], measurement_dim))
        for sigma_index in range(sigmas.shape[0]):
            meas_sigmas[sigma_index] = self.measurement_function(
                sigmas[sigma_index],
                measurements,
            )

        predicted_measurement = zeros(measurement_dim)
        for sigma_index in range(sigmas.shape[0]):
            predicted_measurement += points.Wm[sigma_index] * meas_sigmas[sigma_index]

        innovation_covariance = self._symmetrize(
            meas_noise_cov + self.covariance_regularization * eye(measurement_dim)
        )
        cross_covariance = zeros((self.state_dim, measurement_dim))
        for sigma_index in range(sigmas.shape[0]):
            meas_delta = meas_sigmas[sigma_index] - predicted_measurement
            state_delta = sigmas[sigma_index] - state
            innovation_covariance += points.Wc[sigma_index] * (
                meas_delta[:, None] @ meas_delta[None, :]
            )
            cross_covariance += points.Wc[sigma_index] * (
                state_delta[:, None] @ meas_delta[None, :]
            )

        kalman_gain = linalg.solve(innovation_covariance, cross_covariance.T).T
        posterior_state = state + kalman_gain @ (measurement - predicted_measurement)
        posterior_covariance = (
            self.covariance - kalman_gain @ innovation_covariance @ kalman_gain.T
        )

        self._set_state_vector(posterior_state)
        self.covariance = self._symmetrize(posterior_covariance)
        self.latest_predicted_measurement = predicted_measurement
        self.latest_innovation_covariance = innovation_covariance

        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()


SphericalHarmonicsExtendedObjectTracker = SphericalHarmonicsEOTTracker
