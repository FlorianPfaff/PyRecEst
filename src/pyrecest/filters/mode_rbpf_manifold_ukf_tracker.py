from __future__ import annotations

from numbers import Integral

import numpy as np
from pyrecest import backend
from scipy.special import logsumexp

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-public-methods
class ModeRBPFManifoldUKFTracker(AbstractExtendedObjectTracker):
    """Mode-particle RBPF with a manifold UKF per particle.

    The discrete Rao-Blackwellized particle component carries a motion/shape
    mode. Conditional on a sampled mode, each particle runs a UKF on
    ``R^4 x RP^1 x R^2`` represented by the tangent vector
    ``[x, y, vx, vy, theta, log(a), log(b)]``.

    The orientation component is pi-periodic because ``theta`` and
    ``theta + pi`` describe the same ellipse. The log-axis representation keeps
    semi-axis lengths strictly positive after the UKF update.

    The default modes are:

    * ``free``: orientation follows a random walk.
    * ``velocity``: orientation is attracted to velocity heading.
    * ``maneuver``: velocity-heading attraction with inflated process noise.

    This is an approximate/marginalized RBPF: the conditional continuous state
    is filtered by a UKF rather than an exact Kalman filter.
    """

    _jitter = 1e-9
    _min_probability = 1e-300
    mode_names = ("free", "velocity", "maneuver")

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
        n_particles=50,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        resampling_mode="systematic",
        resampling_threshold=None,
        rng=None,
        time_step_length=1.0,
        alpha=1.0,
        beta=2.0,
        kappa=0.0,
        speed_threshold=1.0,
        velocity_alignment_gain=1.0,
        maneuver_alignment_gain=0.75,
        q_kinematic_scales=(1.0, 1.0, 4.0),
        q_theta_scales=(1.0, 0.25, 4.0),
        q_axis_scales=(1.0, 0.5, 1.0),
        transition_matrix=None,
        initial_mode_probs=None,
        minimum_axis_length=1e-6,
        maximum_axis_length=1e6,
        minimum_covariance_eigenvalue=1e-9,
        scatter_update=True,
        scatter_noise_scale=1.0,
        canonicalize_extent=True,
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

        self.n_particles = int(n_particles)
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        self.rng = self._prepare_rng(rng)
        self.time_step_length = float(time_step_length)
        self.resampling_mode = str(resampling_mode).lower()
        self.resampling_threshold = resampling_threshold
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.speed_threshold = float(speed_threshold)
        self.velocity_alignment_gain = float(velocity_alignment_gain)
        self.maneuver_alignment_gain = float(maneuver_alignment_gain)
        self.minimum_axis_length = float(minimum_axis_length)
        self.maximum_axis_length = float(maximum_axis_length)
        self.minimum_covariance_eigenvalue = float(minimum_covariance_eigenvalue)
        self.scatter_update = bool(scatter_update)
        self.scatter_noise_scale = float(scatter_noise_scale)
        self.canonicalize_extent = bool(canonicalize_extent)

        if self.minimum_axis_length <= 0.0:
            raise ValueError("minimum_axis_length must be positive")
        if self.maximum_axis_length <= self.minimum_axis_length:
            raise ValueError(
                "maximum_axis_length must be larger than minimum_axis_length"
            )
        if self.minimum_covariance_eigenvalue < 0.0:
            raise ValueError("minimum_covariance_eigenvalue must be non-negative")

        self.kinematic_state = np.asarray(kinematic_state, dtype=float).reshape(4)
        self.covariance = self._as_covariance(covariance, 4, "covariance")
        self.shape_state = np.asarray(shape_state, dtype=float).reshape(3)
        self.shape_covariance = self._as_covariance(
            shape_covariance, 3, "shape_covariance"
        )

        if meas_noise_cov is None:
            meas_noise_cov = np.zeros((2, 2), dtype=float)
        self.meas_noise_cov = self._as_covariance(
            meas_noise_cov, 2, "meas_noise_cov", require_pd=False
        )

        if measurement_matrix is None:
            measurement_matrix = np.eye(2, 4)
        self.measurement_matrix = np.asarray(measurement_matrix, dtype=float)
        self._validate_measurement_matrix(self.measurement_matrix)

        if system_matrix is None:
            system_matrix = self._default_system_matrix(self.time_step_length)
        self.system_matrix = np.asarray(system_matrix, dtype=float)
        if self.system_matrix.shape != (4, 4):
            raise ValueError("system_matrix must have shape (4, 4)")

        if sys_noise is None:
            sys_noise = np.zeros((4, 4), dtype=float)
        self.sys_noise = self._as_covariance(
            sys_noise, 4, "sys_noise", require_pd=False
        )
        if shape_sys_noise is None:
            shape_sys_noise = np.zeros((3, 3), dtype=float)
        self.shape_sys_noise = self._as_covariance(
            shape_sys_noise, 3, "shape_sys_noise", require_pd=False
        )
        if multiplicative_noise_cov is None:
            multiplicative_noise_cov = 0.25 * np.eye(2)
        self.multiplicative_noise_cov = self._as_covariance(
            multiplicative_noise_cov, 2, "multiplicative_noise_cov", require_pd=False
        )
        self.q_theta = float(self.shape_sys_noise[0, 0])

        self.q_kinematic_scales = self._validate_mode_scales(
            q_kinematic_scales, "q_kinematic_scales"
        )
        self.q_theta_scales = self._validate_mode_scales(
            q_theta_scales, "q_theta_scales"
        )
        self.q_axis_scales = self._validate_mode_scales(q_axis_scales, "q_axis_scales")

        if transition_matrix is None:
            transition_matrix = np.array(
                [
                    [0.985, 0.010, 0.005],
                    [0.020, 0.955, 0.025],
                    [0.015, 0.080, 0.905],
                ],
                dtype=float,
            )
        self.transition_matrix = self._normalize_rows(
            np.asarray(transition_matrix, dtype=float)
        )
        if self.transition_matrix.shape != (len(self.mode_names), len(self.mode_names)):
            raise ValueError("transition_matrix must have shape (n_modes, n_modes)")

        initial_axis = np.maximum(
            np.abs(self.shape_state[1:]), self.minimum_axis_length
        )
        initial_log_axis = np.log(initial_axis)
        initial_log_axis_cov = self._axis_covariance_to_log_covariance(
            initial_axis, self.shape_covariance[1:, 1:]
        )
        log_axis_sys_noise = self._axis_covariance_to_log_covariance(
            initial_axis, self.shape_sys_noise[1:, 1:]
        )

        self.state_dim = 7
        self.base_process_noise = np.zeros(
            (self.state_dim, self.state_dim), dtype=float
        )
        self.base_process_noise[:4, :4] = self.sys_noise
        self.base_process_noise[4, 4] = max(self.q_theta, 0.0)
        self.base_process_noise[5:, 5:] = log_axis_sys_noise
        self.base_process_noise = self._stabilize_covariance(
            self.base_process_noise, floor=0.0
        )
        self.mode_process_noises = self._build_mode_process_noises()

        initial_mu = np.array(
            [
                *self.kinematic_state,
                self._wrap_ellipse_angle(self.shape_state[0]),
                *initial_log_axis,
            ],
            dtype=float,
        )
        initial_covariance = np.zeros((self.state_dim, self.state_dim), dtype=float)
        initial_covariance[:4, :4] = self.covariance
        initial_covariance[4, 4] = max(
            float(self.shape_covariance[0, 0]), self.minimum_covariance_eigenvalue
        )
        initial_covariance[5:, 5:] = initial_log_axis_cov
        initial_covariance = self._stabilize_covariance(initial_covariance)

        self.mu = np.repeat(initial_mu[np.newaxis, :], self.n_particles, axis=0)
        self.covariances = np.repeat(
            initial_covariance[np.newaxis, :, :], self.n_particles, axis=0
        )

        speed = float(np.linalg.norm(self.kinematic_state[2:4]))
        if initial_mode_probs is None:
            initial_mode_probs = (
                [0.92, 0.06, 0.02]
                if speed <= self.speed_threshold
                else [0.05, 0.85, 0.10]
            )
        initial_mode_probs = self._normalize_probs(initial_mode_probs)
        self.modes = self.rng.choice(
            len(self.mode_names), size=self.n_particles, p=initial_mode_probs
        )
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        self._n_observations = 0
        self._canonicalize_particles()

    @staticmethod
    def _raise_if_backend_unsupported():
        if backend.__backend_name__ != "numpy":
            raise NotImplementedError(
                "ModeRBPFManifoldUKFTracker is currently supported on the NumPy backend only."
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
        n_particles=50,
        scaling_factor=0.25,
        **kwargs,
    ):
        """Build from the argument names used in the MEM-QKF clone."""
        return cls(
            kinematic_state=m_init,
            covariance=p_kinematic_init,
            shape_state=p_init,
            shape_covariance=p_shape_init,
            meas_noise_cov=r,
            sys_noise=q_kinematic,
            shape_sys_noise=q_shape,
            multiplicative_noise_cov=float(scaling_factor) * np.eye(2),
            n_particles=n_particles,
            **kwargs,
        )

    @staticmethod
    def _prepare_rng(rng):
        if rng is None:
            return np.random.default_rng()
        if isinstance(rng, Integral):
            return np.random.default_rng(int(rng))
        if isinstance(rng, np.random.Generator):
            return rng
        raise TypeError(
            "rng must be None, an integer seed, or a numpy.random.Generator"
        )

    @staticmethod
    def _default_system_matrix(time_step_length):
        return np.array(
            [
                [1.0, 0.0, time_step_length, 0.0],
                [0.0, 1.0, 0.0, time_step_length],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @staticmethod
    def _rotation(angle):
        return np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=float,
        )

    @staticmethod
    def _symmetrize(matrix):
        matrix = np.asarray(matrix, dtype=float)
        return 0.5 * (matrix + matrix.T)

    @classmethod
    def _as_covariance(cls, value, dim, name, require_pd=True):
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim == 0:
            matrix = float(matrix) * np.eye(dim)
        elif matrix.ndim == 1:
            if matrix.shape[0] != dim:
                raise ValueError(f"{name} vector must have length {dim}")
            matrix = np.diag(matrix)
        if matrix.shape != (dim, dim):
            raise ValueError(f"{name} must have shape ({dim}, {dim})")
        matrix = cls._symmetrize(matrix)
        if require_pd:
            np.linalg.cholesky(matrix)
        return matrix

    @staticmethod
    def _validate_measurement_matrix(measurement_matrix):
        if measurement_matrix.shape != (2, 4):
            raise ValueError("measurement_matrix must have shape (2, 4)")

    @property
    def log_weights(self):
        return np.log(np.maximum(self.weights, self._min_probability))

    @log_weights.setter
    def log_weights(self, value):
        self.weights = self._normalize_log_weights(np.asarray(value, dtype=float))

    def predict(self):
        self._propagate_modes()
        for particle_ix in range(self.n_particles):
            self.mu[particle_ix], self.covariances[particle_ix] = (
                self._ukf_predict_particle(
                    self.mu[particle_ix],
                    self.covariances[particle_ix],
                    int(self.modes[particle_ix]),
                )
            )
        self._canonicalize_particles()
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()
        return self.get_point_estimate()

    def update(self, measurements):
        measurements = np.asarray(measurements, dtype=float).reshape((-1, 2))
        if len(measurements) == 0:
            return self.get_point_estimate()

        observation, include_scatter = self._measurement_summary(measurements)
        log_likelihoods = np.empty(self.n_particles, dtype=float)

        for particle_ix in range(self.n_particles):
            updated_mu, updated_covariance, log_likelihood = self._ukf_update_particle(
                mu=self.mu[particle_ix],
                covariance=self.covariances[particle_ix],
                observation=observation,
                n_measurements=len(measurements),
                include_scatter=include_scatter,
            )
            self.mu[particle_ix] = updated_mu
            self.covariances[particle_ix] = updated_covariance
            log_likelihoods[particle_ix] = log_likelihood

        self.log_weights = self.log_weights + log_likelihoods
        self._canonicalize_particles()
        if self._should_resample():
            self._resample()
        self._n_observations += len(measurements)
        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()
        return self.get_point_estimate()

    def _ukf_predict_particle(self, mu, covariance, mode):
        sigma_points, mean_weights, covariance_weights = self._sigma_points(
            mu, covariance
        )
        propagated = np.array(
            [
                self._transition_function(sigma_point, mode)
                for sigma_point in sigma_points
            ]
        )
        predicted_mu = self._manifold_weighted_mean(propagated, mean_weights)
        state_sigma = self._boxminus(propagated, predicted_mu)
        predicted_covariance = state_sigma.T @ (
            covariance_weights[:, np.newaxis] * state_sigma
        )
        predicted_covariance += self.mode_process_noises[mode]
        return predicted_mu, self._stabilize_covariance(predicted_covariance)

    def _ukf_update_particle(
        self, mu, covariance, observation, n_measurements, include_scatter
    ):
        sigma_points, mean_weights, covariance_weights = self._sigma_points(
            mu, covariance
        )
        predicted_measurements = np.array(
            [
                self._measurement_function(sigma_point, include_scatter)
                for sigma_point in sigma_points
            ]
        )

        predicted_observation = mean_weights @ predicted_measurements
        innovation_sigma = predicted_measurements - predicted_observation
        state_sigma = self._boxminus(sigma_points, mu)
        pseudo_noise = self._pseudo_measurement_noise(
            mu, n_measurements, include_scatter
        )

        innovation_covariance = (
            innovation_sigma.T @ (covariance_weights[:, np.newaxis] * innovation_sigma)
            + pseudo_noise
        )
        innovation_covariance = self._stabilize_covariance(innovation_covariance)
        cross_covariance = state_sigma.T @ (
            covariance_weights[:, np.newaxis] * innovation_sigma
        )

        innovation = observation - predicted_observation
        kalman_gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
        update_step = kalman_gain @ innovation
        updated_mu = self._boxplus(mu, update_step)
        updated_covariance = (
            covariance - kalman_gain @ innovation_covariance @ kalman_gain.T
        )
        updated_covariance = self._stabilize_covariance(updated_covariance)

        log_likelihood = self._gaussian_logpdf(innovation, innovation_covariance)
        return updated_mu, updated_covariance, log_likelihood

    def _transition_function(self, tangent_state, mode):
        output = np.asarray(tangent_state, dtype=float).copy()
        output[:4] = self.system_matrix @ tangent_state[:4]

        if self.mode_names[mode] == "velocity":
            output[4] = self._align_orientation_to_velocity(
                tangent_state, self.velocity_alignment_gain
            )
        elif self.mode_names[mode] == "maneuver":
            output[4] = self._align_orientation_to_velocity(
                tangent_state, self.maneuver_alignment_gain
            )
        else:
            output[4] = self._wrap_ellipse_angle(tangent_state[4])

        output[5:] = self._clip_log_axes(output[5:])
        return output

    def _align_orientation_to_velocity(self, tangent_state, gain):
        vx, vy = tangent_state[2], tangent_state[3]
        speed_sq = float(vx**2 + vy**2)
        if speed_sq <= self.speed_threshold**2:
            return self._wrap_ellipse_angle(tangent_state[4])
        heading = np.arctan2(vy, vx)
        delta = self._ellipse_angle_delta(tangent_state[4], heading)
        return self._wrap_ellipse_angle(tangent_state[4] + float(gain) * delta)

    def _measurement_summary(self, measurements):
        centroid = np.average(measurements, axis=0)
        include_scatter = self.scatter_update and len(measurements) >= 2
        if not include_scatter:
            return centroid, False

        centered = measurements - centroid
        scatter = centered.T @ centered / (len(measurements) - 1)
        return (
            np.array(
                [centroid[0], centroid[1], scatter[0, 0], scatter[1, 1], scatter[0, 1]],
                dtype=float,
            ),
            True,
        )

    def _measurement_function(self, tangent_state, include_scatter):
        centroid = self.measurement_matrix @ tangent_state[:4]
        output = [centroid[0], centroid[1]]
        if include_scatter:
            measurement_covariance = self._single_detection_covariance(tangent_state)
            output.extend(
                [
                    measurement_covariance[0, 0],
                    measurement_covariance[1, 1],
                    measurement_covariance[0, 1],
                ]
            )
        return np.asarray(output, dtype=float)

    def _single_detection_covariance(self, tangent_state):
        semi_axis = np.exp(self._clip_log_axes(tangent_state[5:]))
        extent_transform = self._rotation(tangent_state[4]) @ np.diag(semi_axis)
        extent_covariance = (
            extent_transform @ self.multiplicative_noise_cov @ extent_transform.T
        )
        return self.meas_noise_cov + extent_covariance

    def _pseudo_measurement_noise(self, tangent_state, n_measurements, include_scatter):
        detection_covariance = self._single_detection_covariance(tangent_state)
        if not include_scatter:
            return self._stabilize_covariance(detection_covariance / n_measurements)

        noise = np.zeros((5, 5), dtype=float)
        noise[:2, :2] = detection_covariance / n_measurements

        denom = max(n_measurements - 1, 1)
        s_xx = detection_covariance[0, 0]
        s_yy = detection_covariance[1, 1]
        s_xy = detection_covariance[0, 1]
        scatter_noise = np.array(
            [
                [2.0 * s_xx**2, 2.0 * s_xy**2, 2.0 * s_xx * s_xy],
                [2.0 * s_xy**2, 2.0 * s_yy**2, 2.0 * s_yy * s_xy],
                [2.0 * s_xx * s_xy, 2.0 * s_yy * s_xy, s_xx * s_yy + s_xy**2],
            ],
            dtype=float,
        )
        noise[2:, 2:] = self.scatter_noise_scale * scatter_noise / denom
        return self._stabilize_covariance(noise)

    def _sigma_points(self, mu, covariance):
        n = self.state_dim
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        spread = n + lambda_
        if spread <= 0.0:
            raise ValueError("Invalid UKF spread. Increase alpha or kappa.")

        covariance = self._stabilize_covariance(covariance)
        chol = np.linalg.cholesky(spread * covariance + self._jitter * np.eye(n))
        sigma_points = np.empty((2 * n + 1, n), dtype=float)
        sigma_points[0] = mu
        for dim in range(n):
            sigma_points[1 + dim] = self._boxplus(mu, chol[:, dim])
            sigma_points[1 + n + dim] = self._boxplus(mu, -chol[:, dim])

        mean_weights = np.full(2 * n + 1, 1.0 / (2.0 * spread), dtype=float)
        covariance_weights = mean_weights.copy()
        mean_weights[0] = lambda_ / spread
        covariance_weights[0] = lambda_ / spread + (1.0 - self.alpha**2 + self.beta)
        return sigma_points, mean_weights, covariance_weights

    def _boxplus(self, state, delta):
        output = np.asarray(state, dtype=float) + np.asarray(delta, dtype=float)
        output[4] = self._wrap_ellipse_angle(output[4])
        output[5:] = self._clip_log_axes(output[5:])
        return output

    def _boxminus(self, states, reference):
        states = np.asarray(states, dtype=float)
        reference = np.asarray(reference, dtype=float)
        delta = states - reference
        delta[..., 4] = self._ellipse_angle_delta(reference[4], states[..., 4])
        return delta

    def _manifold_weighted_mean(self, sigma_points, mean_weights):
        mean = np.asarray(sigma_points[0], dtype=float).copy()
        mean[:4] = mean_weights @ sigma_points[:, :4]
        mean[5:] = mean_weights @ sigma_points[:, 5:]
        mean[4] = self._weighted_ellipse_angle_mean(sigma_points[:, 4], mean_weights)
        mean[5:] = self._clip_log_axes(mean[5:])

        for _ in range(5):
            correction = mean_weights @ self._boxminus(sigma_points, mean)
            if np.linalg.norm(correction) <= 1e-10:
                break
            mean = self._boxplus(mean, correction)
        return mean

    def _propagate_modes(self):
        new_modes = np.empty_like(self.modes)
        for particle_ix, mode in enumerate(self.modes):
            new_modes[particle_ix] = self.rng.choice(
                len(self.mode_names), p=self.transition_matrix[int(mode)]
            )
        self.modes = new_modes

    def _should_resample(self):
        threshold = (
            self.n_particles / 2.0
            if self.resampling_threshold is None
            else float(self.resampling_threshold)
        )
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        return effective_sample_size <= threshold

    def _resample(self):
        indices = self._sample_indices(self.weights, self.n_particles)
        self.mu = self.mu[indices].copy()
        self.covariances = self.covariances[indices].copy()
        self.modes = self.modes[indices].copy()
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)

    def _sample_indices(self, weights, size):
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        mode = str(self.resampling_mode).lower()
        if mode == "multinomial":
            return self.rng.choice(len(weights), size=size, p=weights)
        if mode == "systematic":
            return self._systematic_resample(weights, size)
        if mode == "stratified":
            positions = (self.rng.random(size) + np.arange(size)) / size
            cumulative = np.cumsum(weights)
            cumulative[-1] = 1.0
            return np.searchsorted(cumulative, positions)
        if mode == "residual":
            return self._residual_resample(weights, size)
        raise NotImplementedError(
            f"Resampling mode {self.resampling_mode} not supported"
        )

    def _systematic_resample(self, weights, size):
        positions = (self.rng.random() + np.arange(size)) / size
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions)

    def _residual_resample(self, weights, size):
        expected = size * weights
        deterministic_counts = np.floor(expected).astype(int)
        indices = np.repeat(np.arange(len(weights)), deterministic_counts)
        remainder = size - len(indices)
        if remainder > 0:
            residual = expected - deterministic_counts
            residual_sum = np.sum(residual)
            residual_weights = (
                residual / residual_sum
                if residual_sum > 0.0
                else np.full(len(weights), 1.0 / len(weights))
            )
            indices = np.concatenate(
                [indices, self._systematic_resample(residual_weights, remainder)]
            )
        self.rng.shuffle(indices)
        return indices

    def _canonicalize_particles(self):
        self.mu[:, 4] = self._wrap_ellipse_angle(self.mu[:, 4])
        self.mu[:, 5:] = self._clip_log_axes(self.mu[:, 5:])
        if not self.canonicalize_extent:
            return

        axes = np.exp(self.mu[:, 5:])
        swap = axes[:, 1] > axes[:, 0]
        if not np.any(swap):
            return
        swapped_log_axes = self.mu[swap, 5:7][:, ::-1].copy()
        swapped_cov_rows = self.covariances[swap, 5:7, :][:, ::-1, :].copy()
        swapped_cov_cols = self.covariances[swap, :, 5:7][:, :, ::-1].copy()
        self.mu[swap, 5:7] = swapped_log_axes
        self.covariances[swap, 5:7, :] = swapped_cov_rows
        self.covariances[swap, :, 5:7] = swapped_cov_cols
        self.mu[swap, 4] = self._wrap_ellipse_angle(self.mu[swap, 4] + np.pi / 2.0)

    def get_point_estimate(self):
        return np.concatenate(
            [self.get_point_estimate_kinematics(), self.get_point_estimate_shape()]
        )

    def get_point_estimate_kinematics(self):
        return self.weights @ self.mu[:, :4]

    def get_point_estimate_shape(self):
        theta = self._weighted_ellipse_angle_mean(self.mu[:, 4], self.weights)
        axis = self._weighted_axis_expectation()
        return np.array([theta, axis[0], axis[1]], dtype=float)

    def get_shape_point_estimate(self):
        return self.get_point_estimate_shape()

    def get_point_estimate_extent(self, flatten_matrix=False):
        shape_state = self.get_point_estimate_shape()
        rotation = self._rotation(shape_state[0])
        extent = self._symmetrize(rotation @ np.diag(shape_state[1:] ** 2) @ rotation.T)
        if flatten_matrix:
            return extent.flatten()
        return extent

    @property
    def extent(self):
        return self.get_point_estimate_extent()

    def get_state(self, full_axis_lengths=True):
        state = self.get_point_estimate()
        if full_axis_lengths:
            return np.concatenate([state[:-2], 2.0 * state[-2:]])
        return state

    def get_state_and_cov(self, full_axis_lengths=True):
        mean = self.get_state(full_axis_lengths=full_axis_lengths)
        covariance = np.zeros((7, 7), dtype=float)
        for particle_ix in range(self.n_particles):
            particle_state = self._particle_physical_state(
                particle_ix,
                include_log_axis_variance=True,
                full_axis_lengths=full_axis_lengths,
            )
            delta = particle_state - mean
            delta[4] = self._ellipse_angle_delta(mean[4], particle_state[4])
            conditional_covariance = self._particle_physical_covariance(
                particle_ix, full_axis_lengths=full_axis_lengths
            )
            covariance += self.weights[particle_ix] * (
                conditional_covariance + np.outer(delta, delta)
            )
        return mean, self._stabilize_covariance(covariance)

    def get_state_array(self, with_weight=False, full_axis_lengths=False):
        states = np.array(
            [
                self._particle_physical_state(
                    particle_ix,
                    include_log_axis_variance=False,
                    full_axis_lengths=full_axis_lengths,
                )
                for particle_ix in range(self.n_particles)
            ]
        )
        if with_weight:
            return np.column_stack([states, self.weights])
        return states

    def get_mode_probabilities(self):
        probabilities = np.zeros(len(self.mode_names), dtype=float)
        for mode_ix in range(len(self.mode_names)):
            probabilities[mode_ix] = np.sum(self.weights[self.modes == mode_ix])
        return dict(zip(self.mode_names, probabilities))

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
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        unit_circle = np.array([np.cos(angles), np.sin(angles)])
        center = self.measurement_matrix @ self.get_point_estimate_kinematics()
        contour_points = (
            center[:, None]
            + scaling_factor * rotation @ np.diag(shape_state[1:]) @ unit_circle
        )
        return contour_points.T

    def _weighted_axis_expectation(self):
        log_axes = self.mu[:, 5:]
        axis_variances = np.stack(
            [self.covariances[:, 5, 5], self.covariances[:, 6, 6]], axis=1
        )
        axis_expectations = np.exp(self._clip_log_axes(log_axes) + 0.5 * axis_variances)
        return self.weights @ axis_expectations

    def _particle_physical_state(
        self, particle_ix, include_log_axis_variance, full_axis_lengths
    ):
        tangent_state = self.mu[particle_ix]
        if include_log_axis_variance:
            log_axis = tangent_state[5:] + 0.5 * np.array(
                [
                    self.covariances[particle_ix, 5, 5],
                    self.covariances[particle_ix, 6, 6],
                ],
                dtype=float,
            )
        else:
            log_axis = tangent_state[5:]
        semi_axis = np.exp(self._clip_log_axes(log_axis))
        axis = 2.0 * semi_axis if full_axis_lengths else semi_axis
        return np.array([*tangent_state[:4], tangent_state[4], *axis], dtype=float)

    def _particle_physical_covariance(self, particle_ix, full_axis_lengths):
        tangent_covariance = self.covariances[particle_ix]
        semi_axis = np.exp(self._clip_log_axes(self.mu[particle_ix, 5:]))
        axis_scale = 2.0 * semi_axis if full_axis_lengths else semi_axis
        jacobian = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, axis_scale[0], axis_scale[1]])
        return jacobian @ tangent_covariance @ jacobian.T

    def _build_mode_process_noises(self):
        noises = []
        for q_kin_scale, q_theta_scale, q_axis_scale in zip(
            self.q_kinematic_scales, self.q_theta_scales, self.q_axis_scales
        ):
            noise = np.zeros_like(self.base_process_noise)
            noise[:4, :4] = q_kin_scale * self.base_process_noise[:4, :4]
            noise[4, 4] = q_theta_scale * self.base_process_noise[4, 4]
            noise[5:, 5:] = q_axis_scale * self.base_process_noise[5:, 5:]
            noises.append(self._stabilize_covariance(noise, floor=0.0))
        return np.asarray(noises)

    def _axis_covariance_to_log_covariance(self, axis, covariance):
        axis = np.maximum(np.asarray(axis, dtype=float), self.minimum_axis_length)
        covariance = np.asarray(covariance, dtype=float)
        jacobian = np.diag(1.0 / axis)
        return self._stabilize_covariance(jacobian @ covariance @ jacobian.T, floor=0.0)

    def _clip_log_axes(self, log_axis):
        lower = np.log(self.minimum_axis_length)
        upper = np.log(self.maximum_axis_length)
        return np.clip(log_axis, lower, upper)

    def _stabilize_covariance(self, covariance, floor=None):
        if floor is None:
            floor = self.minimum_covariance_eigenvalue
        covariance = np.asarray(covariance, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        if covariance.size == 0:
            return covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, floor)
        stabilized = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return 0.5 * (stabilized + stabilized.T)

    def _validate_mode_scales(self, scales, name):
        scales = np.asarray(scales, dtype=float)
        if scales.shape != (len(self.mode_names),):
            raise ValueError(f"{name} must have one entry per mode")
        return scales

    @staticmethod
    def _normalize_rows(matrix):
        matrix = np.maximum(matrix, ModeRBPFManifoldUKFTracker._jitter)
        return matrix / np.sum(matrix, axis=1, keepdims=True)

    @staticmethod
    def _normalize_probs(probs):
        probs = np.maximum(
            np.asarray(probs, dtype=float), ModeRBPFManifoldUKFTracker._jitter
        )
        return probs / np.sum(probs)

    @staticmethod
    def _normalize_log_weights(log_weights):
        log_weights = np.asarray(log_weights, dtype=float)
        finite = np.isfinite(log_weights)
        if not np.any(finite):
            return np.full(log_weights.shape, 1.0 / log_weights.size)
        safe_log_weights = np.where(finite, log_weights, -np.inf)
        return np.exp(safe_log_weights - logsumexp(safe_log_weights))

    @staticmethod
    def _gaussian_logpdf(innovation, covariance):
        innovation = np.asarray(innovation, dtype=float)
        covariance = 0.5 * (
            np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T
        )
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            covariance = covariance + 1e-9 * np.eye(covariance.shape[0])
            sign, logdet = np.linalg.slogdet(covariance)
        quadratic = innovation @ np.linalg.solve(covariance, innovation)
        return float(
            -0.5 * (len(innovation) * np.log(2.0 * np.pi) + logdet + quadratic)
        )

    @staticmethod
    def _wrap_ellipse_angle(theta):
        return np.asarray(theta) % np.pi

    @staticmethod
    def _ellipse_angle_delta(reference, theta):
        return (
            (np.asarray(theta) - np.asarray(reference) + np.pi / 2.0) % np.pi
        ) - np.pi / 2.0

    @staticmethod
    def _weighted_ellipse_angle_mean(theta, weights):
        theta = np.asarray(theta, dtype=float)
        weights = np.asarray(weights, dtype=float)
        doubled = 2.0 * theta
        sin_sum = np.sum(weights * np.sin(doubled))
        cos_sum = np.sum(weights * np.cos(doubled))
        if np.hypot(sin_sum, cos_sum) <= ModeRBPFManifoldUKFTracker._jitter:
            return float(np.asarray(theta).flat[0] % np.pi)
        return float((0.5 * np.arctan2(sin_sum, cos_sum)) % np.pi)


ModeRbpfManifoldUkfTracker = ModeRBPFManifoldUKFTracker
ModeRBPFManifoldUKF = ModeRBPFManifoldUKFTracker
