from __future__ import annotations

from numbers import Integral

from pyrecest import backend
from pyrecest.backend import any as backend_any
from pyrecest.backend import (
    arange,
    arctan2,
    array,
    concatenate,
    copy,
    cos,
    cumsum,
    diag,
    einsum,
    exp,
    eye,
    floor,
    full,
    isfinite,
    linalg,
    linspace,
    log,
)
from pyrecest.backend import max as backend_max
from pyrecest.backend import (
    maximum,
    mean,
    ones,
    pi,
    random,
    searchsorted,
    sin,
    sqrt,
    stack,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    transpose,
    where,
    zeros,
)

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker

# pylint: disable=no-name-in-module,no-member,duplicate-code
# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-positional-arguments,too-many-locals


class MEMRBPFTracker(AbstractExtendedObjectTracker):
    """Rao-Blackwellized MEM tracker for a single 2-D elliptical target.

    The orientation is represented by weighted particles. For every
    orientation particle, a conditional linear-Gaussian state stores the two
    semi-axis lengths. The kinematic state is updated with a Kalman update
    using the mean measurement of each scan; the extent part is updated with
    the pseudo-measurement recursion from the MEM-RBPF reference code.
    """

    def __init__(
        self,
        kinematic_state,
        covariance,
        shape_state,
        shape_covariance,
        meas_noise_cov=None,
        system_matrix=None,
        sys_noise=None,
        shape_sys_noise=None,
        n_particles=100,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        resampling_mode="systematic",
        resampling_threshold=None,
        rng=None,
        time_step_length=1.0,
        covariance_regularization=0.0,
        axis_floor=None,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        self._raise_if_backend_unsupported()
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        self.n_particles = int(n_particles)
        self._seed_backend_random(rng)
        self.resampling_mode = str(resampling_mode).lower()
        self.resampling_threshold = resampling_threshold
        self.covariance_regularization = float(covariance_regularization)
        self.axis_floor = axis_floor
        self.measurement_dim = 2

        self.kinematic_state = array(kinematic_state)
        if self.kinematic_state.ndim != 1 or self.kinematic_state.shape[0] < 2:
            raise ValueError(
                "kinematic_state must be a vector with at least two entries"
            )
        self.state_dim = self.kinematic_state.shape[0]
        self.covariance = self._as_covariance(covariance, self.state_dim, "covariance")

        shape_state = array(shape_state)
        self._validate_shape_state(shape_state)
        self.shape_covariance = self._as_covariance(
            shape_covariance, 3, "shape_covariance"
        )

        if meas_noise_cov is None:
            meas_noise_cov = zeros((2, 2))
        self.meas_noise_cov = self._as_covariance(
            meas_noise_cov, 2, "meas_noise_cov", require_pd=False
        )
        if multiplicative_noise_cov is None:
            multiplicative_noise_cov = 0.25 * eye(2)
        self.multiplicative_noise_cov = self._as_covariance(
            multiplicative_noise_cov, 2, "multiplicative_noise_cov"
        )
        self._check_isotropic_multiplicative_noise(self.multiplicative_noise_cov)
        self.multiplicative_noise_variance = float(self.multiplicative_noise_cov[0, 0])

        if measurement_matrix is None:
            measurement_matrix = eye(2, self.state_dim)
        self.measurement_matrix = array(measurement_matrix)
        self._validate_measurement_matrix(self.measurement_matrix)

        if system_matrix is None:
            system_matrix = self._default_system_matrix(time_step_length)
        self.system_matrix = array(system_matrix)
        self._validate_system_matrix(self.system_matrix)
        if sys_noise is None:
            sys_noise = zeros((self.state_dim, self.state_dim))
        self.sys_noise = self._as_covariance(
            sys_noise, self.state_dim, "sys_noise", require_pd=False
        )
        if shape_sys_noise is None:
            shape_sys_noise = zeros((3, 3))
        self.shape_sys_noise = self._as_covariance(
            shape_sys_noise, 3, "shape_sys_noise", require_pd=False
        )
        self.orientation_process_variance = float(self.shape_sys_noise[0, 0])
        self.axis_sys_noise = self.shape_sys_noise[1:, 1:]

        theta_std = sqrt(maximum(self.shape_covariance[0, 0], 0.0))
        self.theta = array(
            random.normal(
                loc=float(shape_state[0]),
                scale=float(theta_std),
                size=(self.n_particles,),
            )
            % (2.0 * pi)
        )
        self.axis = ones((self.n_particles, 1)) * shape_state[1:].reshape((1, 2))
        self.axis_covariances = (
            ones((self.n_particles, 1, 1)) * self.shape_covariance[1:, 1:]
        )
        self.weights = full((self.n_particles,), 1.0 / self.n_particles)

    @staticmethod
    def _raise_if_backend_unsupported():
        if backend.__backend_name__ == "jax":
            raise NotImplementedError(
                "MEMRBPFTracker is not supported on the JAX backend because "
                "the filter mutates per-particle state during RBPF updates."
            )

    @staticmethod
    def _seed_backend_random(rng):
        if rng is None:
            return
        if isinstance(rng, Integral):
            random.seed(int(rng))
            return
        raise NotImplementedError(
            "MEMRBPFTracker uses pyrecest.backend.random internally. Pass an "
            "integer seed as rng, or seed pyrecest.backend.random before "
            "constructing the tracker."
        )

    @classmethod
    def from_original_parameters(
        cls,
        m_init,
        p_init,
        p_kinematic_init,
        p_shape_init,
        r,
        q_kinematic,
        q_shape,
        n_particles=100,
        resampling_var=None,
        **kwargs,
    ):
        """Build from the argument names used in the MEM-QKF clone."""
        cls._raise_if_backend_unsupported()
        q_shape = copy(array(q_shape))
        if resampling_var is not None:
            q_shape[0, 0] = float(resampling_var)
        return cls(
            kinematic_state=m_init,
            covariance=p_kinematic_init,
            shape_state=p_init,
            shape_covariance=p_shape_init,
            meas_noise_cov=r,
            sys_noise=q_kinematic,
            shape_sys_noise=q_shape,
            n_particles=n_particles,
            **kwargs,
        )

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _symmetrize_stack(matrices):
        axes = tuple(range(matrices.ndim - 2)) + (matrices.ndim - 1, matrices.ndim - 2)
        return 0.5 * (matrices + transpose(matrices, axes))

    @classmethod
    def _as_covariance(cls, value, dim, name, require_pd=True):
        matrix = array(value)
        if matrix.ndim == 0:
            matrix = matrix * eye(dim)
        elif matrix.ndim == 1:
            if matrix.shape[0] != dim:
                raise ValueError(f"{name} must have length {dim}")
            matrix = diag(matrix)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        matrix = cls._symmetrize(matrix)
        if require_pd:
            linalg.cholesky(matrix)
        return matrix

    @staticmethod
    def _validate_shape_state(shape_state):
        if shape_state.shape != (3,):
            raise ValueError("shape_state must have shape (3,)")
        if float(shape_state[1]) <= 0.0 or float(shape_state[2]) <= 0.0:
            raise ValueError("shape semi-axis lengths must be positive")

    @staticmethod
    def _check_isotropic_multiplicative_noise(noise):
        if abs(float(noise[0, 1])) > 1e-12 or abs(float(noise[1, 0])) > 1e-12:
            raise ValueError("multiplicative_noise_cov must be diagonal")
        if abs(float(noise[0, 0] - noise[1, 1])) > 1e-12:
            raise ValueError("multiplicative_noise_cov must be isotropic")

    def _validate_measurement_matrix(self, measurement_matrix):
        if measurement_matrix.shape != (2, self.state_dim):
            raise ValueError("measurement_matrix must have shape (2, state_dim)")

    def _validate_system_matrix(self, system_matrix):
        if system_matrix.shape != (self.state_dim, self.state_dim):
            raise ValueError("system_matrix must have shape (state_dim, state_dim)")

    def _default_system_matrix(self, dt):
        if self.state_dim != 4:
            return eye(self.state_dim)
        return array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _rotation(theta):
        ctheta = cos(theta)
        stheta = sin(theta)
        first_row = stack([ctheta, -stheta], axis=-1)
        second_row = stack([stheta, ctheta], axis=-1)
        return stack([first_row, second_row], axis=-2)

    def _normalize_measurements(self, measurements):
        measurements = array(measurements)
        if measurements.ndim == 1:
            if measurements.shape[0] != 2:
                raise ValueError("a single measurement must be two-dimensional")
            return measurements.reshape((1, 2))
        if measurements.ndim != 2:
            raise ValueError("measurements must be one- or two-dimensional")
        if measurements.shape[1] == 2:
            return measurements
        if measurements.shape[0] == 2:
            return measurements.T
        raise ValueError("measurements must have shape (n, 2) or (2, n)")

    def predict_linear(
        self,
        system_matrix=None,
        sys_noise=None,
        inputs=None,
        shape_system_matrix=None,
        shape_sys_noise=None,
    ):
        if system_matrix is not None:
            self.system_matrix = array(system_matrix)
            self._validate_system_matrix(self.system_matrix)
        if sys_noise is not None:
            self.sys_noise = self._as_covariance(
                sys_noise, self.state_dim, "sys_noise", require_pd=False
            )
        self.kinematic_state = self.system_matrix @ self.kinematic_state
        if inputs is not None:
            self.kinematic_state = self.kinematic_state + array(inputs)
        self.covariance = self._symmetrize(
            self.system_matrix @ self.covariance @ self.system_matrix.T + self.sys_noise
        )

        axis_matrix = eye(2)
        if shape_system_matrix is not None:
            shape_system_matrix = array(shape_system_matrix)
            if shape_system_matrix.shape == (3, 3):
                axis_matrix = shape_system_matrix[1:, 1:]
            elif shape_system_matrix.shape == (2, 2):
                axis_matrix = shape_system_matrix
            else:
                raise ValueError("shape_system_matrix must be 3x3 or 2x2")
        if shape_sys_noise is not None:
            shape_sys_noise = array(shape_sys_noise)
            if shape_sys_noise.shape == (3, 3):
                self.orientation_process_variance = float(shape_sys_noise[0, 0])
                self.axis_sys_noise = shape_sys_noise[1:, 1:]
            elif shape_sys_noise.shape == (2, 2):
                self.axis_sys_noise = shape_sys_noise
            else:
                raise ValueError("shape_sys_noise must be 3x3 or 2x2")
        self.axis = self.axis @ axis_matrix.T
        self.axis_covariances = self._symmetrize_stack(
            axis_matrix @ self.axis_covariances @ axis_matrix.T
            + self.axis_sys_noise.reshape((1, 2, 2))
        )
        self._apply_axis_floor()
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    def predict(self, *args, **kwargs):
        self.predict_linear(*args, **kwargs)

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        measurements = self._normalize_measurements(measurements)
        if measurements.shape[0] == 0:
            return
        if meas_mat is not None:
            meas_mat = array(meas_mat)
            self._validate_measurement_matrix(meas_mat)
        else:
            meas_mat = self.measurement_matrix
        if meas_noise_cov is None:
            meas_noise_cov = self.meas_noise_cov
        else:
            meas_noise_cov = self._as_covariance(
                meas_noise_cov, 2, "meas_noise_cov", require_pd=False
            )
        if multiplicative_noise_cov is not None:
            multiplicative_noise_cov = self._as_covariance(
                multiplicative_noise_cov, 2, "multiplicative_noise_cov"
            )
            self._check_isotropic_multiplicative_noise(multiplicative_noise_cov)
            multiplicative_variance = float(multiplicative_noise_cov[0, 0])
        else:
            multiplicative_variance = self.multiplicative_noise_variance

        self._update_kinematics(
            measurements, meas_mat, meas_noise_cov, multiplicative_variance
        )
        self._propagate_orientation_particles()
        centered = measurements - (meas_mat @ self.kinematic_state)
        self._update_particle_weights(centered, meas_noise_cov, multiplicative_variance)
        aligned = einsum(
            "pab,mb->pma",
            self._rotation(-self.theta),
            centered,
        )
        self._update_axes(aligned, meas_noise_cov, multiplicative_variance)
        self._apply_axis_floor()
        if self._should_resample():
            self.resample()
        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _update_kinematics(
        self, measurements, meas_mat, meas_noise_cov, multiplicative_variance
    ):
        n_measurements = measurements.shape[0]
        shape_state = self.get_point_estimate_shape()
        rotation = self._rotation(shape_state[0])
        extent = rotation @ diag(shape_state[1:] ** 2) @ rotation.T
        innovation_cov = self._symmetrize(
            meas_mat @ self.covariance @ meas_mat.T
            + (meas_noise_cov + multiplicative_variance * extent) / n_measurements
        )
        if self.covariance_regularization > 0.0:
            innovation_cov = innovation_cov + self.covariance_regularization * eye(2)
        innovation = mean(measurements, axis=0) - (meas_mat @ self.kinematic_state)
        cross_cov = self.covariance @ meas_mat.T
        gain = linalg.solve(innovation_cov.T, cross_cov.T).T
        self.kinematic_state = self.kinematic_state + gain @ innovation
        self.covariance = self._symmetrize(
            self.covariance - gain @ innovation_cov @ gain.T
        )

    def _propagate_orientation_particles(self):
        if self.orientation_process_variance <= 0.0:
            return
        self.theta = (
            self.theta
            + random.normal(
                loc=0.0,
                scale=sqrt(self.orientation_process_variance),
                size=(self.n_particles,),
            )
        ) % (2.0 * pi)

    def _update_particle_weights(self, centered, meas_noise_cov, mult_var):
        log_likelihoods = []
        for particle_index in range(self.n_particles):
            rotation = self._rotation(self.theta[particle_index])
            extent_cov = diag(mult_var * self.axis[particle_index] ** 2)
            extent_cov = extent_cov + mult_var * self.axis_covariances[particle_index]
            marginal_cov = rotation @ extent_cov @ rotation.T + meas_noise_cov
            marginal_cov = self._symmetrize(marginal_cov)
            if self.covariance_regularization > 0.0:
                marginal_cov = marginal_cov + self.covariance_regularization * eye(2)
            determinant = linalg.det(marginal_cov)
            if float(determinant) <= 0.0:
                log_likelihoods.append(array(-float("inf")))
                continue
            inverse_cov = linalg.pinv(marginal_cov)
            quad = einsum("ma,ab,mb->m", centered, inverse_cov, centered)
            log_likelihoods.append(-0.5 * backend_sum(log(determinant) + quad))
        log_likelihoods = array(log_likelihoods)
        log_weights = log(maximum(self.weights, 1e-300))
        self.weights = self._normalize_log_weights(log_weights + log_likelihoods)

    def _update_axes(self, aligned, meas_noise_cov, mult_var):
        axis_state = copy(self.axis)
        covariances = copy(self.axis_covariances)
        for measurement_index in range(aligned.shape[1]):
            y = aligned[:, measurement_index, :]
            pseudo_measurement = y**2
            for particle_index in range(self.n_particles):
                rotation_to_axis_frame = self._rotation(-self.theta[particle_index])
                local_noise = (
                    rotation_to_axis_frame @ meas_noise_cov @ rotation_to_axis_frame.T
                )
                expected = diag(local_noise) + mult_var * (
                    diag(covariances[particle_index]) + axis_state[particle_index] ** 2
                )
                pseudo_cov = array(
                    [
                        [2.0 * expected[0] ** 2, 2.0 * local_noise[0, 1] ** 2],
                        [2.0 * local_noise[1, 0] ** 2, 2.0 * expected[1] ** 2],
                    ]
                )
                if self.covariance_regularization > 0.0:
                    pseudo_cov = pseudo_cov + self.covariance_regularization * eye(2)
                cross_cov = diag(
                    2.0
                    * mult_var
                    * axis_state[particle_index]
                    * diag(covariances[particle_index])
                )
                gain = cross_cov @ linalg.pinv(pseudo_cov)
                axis_state[particle_index] = axis_state[particle_index] + gain @ (
                    pseudo_measurement[particle_index] - expected
                )
                covariances[particle_index] = covariances[particle_index] - (
                    gain @ pseudo_cov @ gain.T
                )
                covariances[particle_index] = self._symmetrize(
                    covariances[particle_index]
                )
        self.axis = axis_state
        self.axis_covariances = covariances

    @staticmethod
    def _normalize_log_weights(log_weights):
        finite = isfinite(log_weights)
        n_weights = int(log_weights.shape[0])
        if not bool(backend_any(finite)):
            return full((n_weights,), 1.0 / n_weights)
        shifted = log_weights - backend_max(where(finite, log_weights, -float("inf")))
        weights = where(finite, exp(shifted), 0.0)
        weight_sum = backend_sum(weights)
        if float(weight_sum) <= 0.0 or not bool(isfinite(weight_sum)):
            return full((n_weights,), 1.0 / n_weights)
        return weights / weight_sum

    @property
    def effective_sample_size(self):
        weights = self.weights / backend_sum(self.weights)
        return float(1.0 / backend_sum(weights**2))

    def _should_resample(self):
        if self.resampling_threshold is None:
            return True
        return self.effective_sample_size <= float(self.resampling_threshold)

    def _resample_indices(self):
        weights = self.weights / backend_sum(self.weights)
        particles = arange(self.n_particles)
        if self.resampling_mode == "multinomial":
            return random.choice(
                particles,
                size=(self.n_particles,),
                replace=True,
                p=weights,
            )
        if self.resampling_mode == "systematic":
            positions = (random.uniform(size=()) + arange(self.n_particles)) / (
                self.n_particles
            )
        elif self.resampling_mode == "stratified":
            positions = (
                random.uniform(size=(self.n_particles,)) + arange(self.n_particles)
            ) / self.n_particles
        elif self.resampling_mode == "residual":
            return self._residual_resample_indices(weights, particles)
        else:
            raise NotImplementedError(
                f"unknown resampling mode: {self.resampling_mode}"
            )

        cumulative_sum = cumsum(weights)
        try:
            return searchsorted(cumulative_sum, positions)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"MEMRBPFTracker {self.resampling_mode!r} resampling requires "
                f"backend.searchsorted, which is not supported by the "
                f"{backend.__backend_name__!r} backend."
            ) from exc

    def _residual_resample_indices(self, weights, particles):
        counts = floor(self.n_particles * weights)
        count_values = [int(counts[index]) for index in range(self.n_particles)]
        deterministic_indices = []
        for particle_index, count in enumerate(count_values):
            deterministic_indices.extend([particle_index] * count)
        deterministic = array(deterministic_indices)
        residual_count = self.n_particles - len(deterministic_indices)
        if residual_count <= 0:
            return deterministic[: self.n_particles]

        residual_weights = weights - counts / self.n_particles
        residual_weight_sum = backend_sum(residual_weights)
        if float(residual_weight_sum) <= 0.0:
            residual_weights = full((self.n_particles,), 1.0 / self.n_particles)
        else:
            residual_weights = residual_weights / residual_weight_sum
        residual = random.choice(
            particles,
            size=(residual_count,),
            replace=True,
            p=residual_weights,
        )
        if deterministic_indices:
            return concatenate([deterministic, residual])
        return residual

    def resample(self):
        indices = self._resample_indices()
        self.theta = self.theta[indices]
        self.axis = self.axis[indices]
        self.axis_covariances = self.axis_covariances[indices]
        self.weights = full((self.n_particles,), 1.0 / self.n_particles)

    def _apply_axis_floor(self):
        if self.axis_floor is None:
            return
        self.axis = maximum(self.axis, float(self.axis_floor))

    def get_point_estimate_shape(self):
        weights = self.weights / backend_sum(self.weights)
        rotations = self._rotation(self.theta)
        shape_matrices = rotations * self.axis.reshape((self.n_particles, 1, 2))
        mean_shape_matrix = backend_sum(
            shape_matrices * weights.reshape((self.n_particles, 1, 1)),
            axis=0,
        )
        extent = mean_shape_matrix @ mean_shape_matrix.T
        eigenvalues, eigenvectors = linalg.eigh(extent)
        major_eigenvalue = maximum(eigenvalues[1], 0.0)
        minor_eigenvalue = maximum(eigenvalues[0], 0.0)
        orientation = arctan2(eigenvectors[1, 1], eigenvectors[0, 1]) % (2.0 * pi)
        return array([orientation, sqrt(major_eigenvalue), sqrt(minor_eigenvalue)])

    @property
    def shape_state(self):
        return self.get_point_estimate_shape()

    @property
    def extent(self):
        return self.get_point_estimate_extent()

    def get_point_estimate(self):
        return concatenate([self.kinematic_state, self.get_point_estimate_shape()])

    def get_point_estimate_kinematics(self):
        return self.kinematic_state

    def get_point_estimate_extent(self, flatten_matrix=False):
        shape_state = self.get_point_estimate_shape()
        rotation = self._rotation(shape_state[0])
        extent = self._symmetrize(rotation @ diag(shape_state[1:] ** 2) @ rotation.T)
        if flatten_matrix:
            return extent.flatten()
        return extent

    def get_state(self, full_axis_lengths=True):
        state = self.get_point_estimate()
        if full_axis_lengths:
            return concatenate([state[:-2], 2.0 * state[-2:]])
        return state

    def get_state_array(self, with_weight=False, full_axis_lengths=False):
        kinematic_rows = ones((self.n_particles, self.state_dim)) * self.kinematic_state
        axis = self.axis
        if full_axis_lengths:
            axis = 2.0 * axis
        rows = concatenate(
            [kinematic_rows, self.theta.reshape((self.n_particles, 1)), axis],
            axis=1,
        )
        if with_weight:
            rows = concatenate(
                [rows, self.weights.reshape((self.n_particles, 1))], axis=1
            )
        return rows

    def set_R(self, meas_noise_cov):
        self.meas_noise_cov = self._as_covariance(
            meas_noise_cov, 2, "meas_noise_cov", require_pd=False
        )

    def set_meas_noise_cov(self, meas_noise_cov):
        self.set_R(meas_noise_cov)

    def get_contour_points(self, n, scaling_factor=1.0):
        if n <= 0:
            raise ValueError("n must be positive")
        shape_state = self.get_point_estimate_shape()
        rotation = self._rotation(shape_state[0])
        angles = linspace(0.0, 2.0 * pi, n, endpoint=False)
        unit_circle = array([cos(angles), sin(angles)])
        center = self.measurement_matrix @ self.kinematic_state
        contour_points = (
            center[:, None]
            + scaling_factor * rotation @ diag(shape_state[1:]) @ unit_circle
        )
        return contour_points.T


MemRbpfTracker = MEMRBPFTracker
