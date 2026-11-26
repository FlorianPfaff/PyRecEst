"""Spatial multiplicative-error tracker with Rao-Blackwellized axis lengths."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp

from pyrecest import backend
from pyrecest.exceptions import ShapeError

from ._spatial_mem_tracker import _SpatialMEMTracker, axis_squared_moments


def _positive_semidefinite(matrix):
    """Symmetrize and remove numerical negative eigenvalues, including in batches."""
    matrix = 0.5 * (matrix + matrix.swapaxes(-1, -2))
    eigenvalues, basis = np.linalg.eigh(matrix)
    if np.any(eigenvalues < 0):
        matrix = (basis * np.maximum(eigenvalues, 0)[..., None, :]) @ basis.swapaxes(
            -1, -2
        )
    return matrix


class MEMRBPF3DTracker(_SpatialMEMTracker):
    """Track one ellipsoid with SO(3) particles and conditional Gaussian axes.

    Each particle carries a body-to-world rotation, three labelled semi-axis
    means and their full conditional covariance. A shared Gaussian kinematic
    state is updated from the scan centroid. The normalized centroid innovation
    and orthogonal, center-free scan contrasts update particle likelihoods and
    conditional axis moments. The observation and moment closures assume
    Gaussian source and sensor noise. By default the source covariance is I/5.

    The initial and process shape covariances describe a Gaussian in
    [right rotation vector, additive semi-axis errors]. Sampling rotations and
    conditioning axes on each draw retains rotation/axis correlations.
    Prediction applies the known body increment before the sampled increment.
    The local NumPy generator accepts a seed or numpy.random.Generator.
    Systematic resampling uses an absolute effective sample size threshold,
    defaulting to half the ensemble size. A zero threshold disables it.
    """

    def __init__(
        self,
        kinematic_state,
        covariance,
        orientation,
        semi_axes,
        shape_covariance,
        *,
        n_particles=100,
        rng=None,
        resampling_threshold=None,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        minimum_axis_length=1e-9,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        if backend.__backend_name__ != "numpy":
            raise NotImplementedError("MEMRBPF3DTracker requires the NumPy backend")
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        if (
            isinstance(n_particles, bool)
            or not isinstance(n_particles, Integral)
            or n_particles <= 0
        ):
            raise ValueError("n_particles must be a positive integer")
        self.n_particles = int(n_particles)
        self.resampling_threshold = float(
            n_particles / 2 if resampling_threshold is None else resampling_threshold
        )
        if not 0 <= self.resampling_threshold <= self.n_particles:
            raise ValueError("resampling_threshold must be between 0 and n_particles")
        self.minimum_axis_length = float(minimum_axis_length)
        if not np.isfinite(self.minimum_axis_length) or self.minimum_axis_length <= 0:
            raise ValueError("minimum_axis_length must be finite and positive")
        self.kinematic_state = self._as_real_array(kinematic_state, "kinematic_state")
        if self.kinematic_state.ndim != 1 or self.kinematic_state.size < 3:
            raise ShapeError(
                "kinematic_state must be a vector containing a 3-D position"
            )
        self.covariance = self._as_covariance_matrix(
            covariance, self.kinematic_state.size, "covariance"
        )
        self.measurement_matrix = self._get_measurement_matrix(measurement_matrix)
        self.multiplicative_noise_cov = self._as_multiplicative_noise(
            0.2 if multiplicative_noise_cov is None else multiplicative_noise_cov
        )
        orientation = self._as_rotation(orientation, "orientation")
        semi_axes = self._as_vector(semi_axes, 3, "semi_axes")
        if np.any(semi_axes <= 0):
            raise ValueError("semi_axes must be positive")
        shape_covariance = self._as_covariance_matrix(
            shape_covariance, 6, "shape_covariance"
        )
        self.rng = np.random.default_rng(rng)
        rotations, offsets, conditional = self._draw_shape_noise(
            shape_covariance, self.rng
        )
        self.rotations = orientation @ rotations
        self.axis = np.maximum(semi_axes + offsets, self.minimum_axis_length)
        self.axis_covariances = np.broadcast_to(
            conditional, (self.n_particles, 3, 3)
        ).copy()
        self._log_weights = np.full(self.n_particles, -np.log(self.n_particles))

    def _draw_shape_noise(self, covariance, rng):
        tangent = covariance[:3, :3]
        if not np.any(tangent):
            return (
                np.broadcast_to(np.eye(3), (self.n_particles, 3, 3)),
                np.zeros((self.n_particles, 3)),
                covariance[3:, 3:],
            )
        eigenvalues, basis = np.linalg.eigh(tangent)
        root = basis * np.sqrt(np.maximum(eigenvalues, 0.0))
        coupling = covariance[3:, :3] @ np.linalg.pinv(tangent, hermitian=True)
        conditional = _positive_semidefinite(
            covariance[3:, 3:] - coupling @ covariance[:3, 3:]
        )
        draws = rng.standard_normal((self.n_particles, 3)) @ root.T
        return Rotation.from_rotvec(draws).as_matrix(), draws @ coupling.T, conditional

    @property
    def weights(self):
        """Normalized probabilities corresponding to the rotation particles."""
        return np.exp(self._log_weights)

    @property
    def effective_sample_size(self):
        weights = self.weights
        return float(1 / (weights @ weights))

    def get_point_estimate_extent(self, flatten_matrix=False):
        """Weighted mean of the particles' geometric point-estimate extents."""
        factors = self.rotations * self.axis[:, None, :]
        extent = np.einsum("p,pik,pjk->ij", self.weights, factors, factors)
        return extent.flatten() if flatten_matrix else extent

    def get_point_estimate_shape(self):
        """Return a right-handed principal frame and descending semi-axis lengths."""
        values, frame = np.linalg.eigh(self.get_point_estimate_extent())
        frame = frame[:, ::-1].copy()
        if np.linalg.det(frame) < 0:
            frame[:, -1] *= -1
        return frame, np.sqrt(np.maximum(values[::-1], 0.0))

    def _extent_transform(self):
        frame, axes = self.get_point_estimate_shape()
        return frame * axes

    @property
    def orientation(self):
        """Right-handed principal frame of the reported mixture extent."""
        return self.get_point_estimate_shape()[0]

    @property
    def semi_axes(self):
        """Semi-axis lengths of the reported mixture extent, largest first."""
        return self.get_point_estimate_shape()[1]

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_sys_noise=None,
        rotation_increment=None,
    ):
        """Predict shared kinematics and the conditional shape of every particle."""
        k = self.kinematic_state.size
        system_matrix = self._as_real_array(system_matrix, "system_matrix")
        if system_matrix.shape != (k, k):
            raise ShapeError(
                "system_matrix must have shape (kinematic_dim, kinematic_dim)"
            )
        process_noise = self._as_covariance_matrix(
            0.0 if sys_noise is None else sys_noise, k, "sys_noise"
        )
        control = 0.0 if inputs is None else self._as_vector(inputs, k, "inputs")
        shape_noise = self._as_covariance_matrix(
            0.0 if shape_sys_noise is None else shape_sys_noise, 6, "shape_sys_noise"
        )
        increment = (
            np.eye(3)
            if rotation_increment is None
            else self._rotation_exp(rotation_increment)
        )
        state = system_matrix @ self.kinematic_state + control
        covariance = _positive_semidefinite(
            system_matrix @ self.covariance @ system_matrix.T + process_noise
        )
        self._require_finite(state, "predicted kinematic state")
        self._require_finite(covariance, "predicted covariance")
        # Publish the RNG advance only when the sampled prediction succeeds.
        next_rng = deepcopy(self.rng)
        rotations, offsets, conditional = self._draw_shape_noise(shape_noise, next_rng)
        rotations = self.rotations @ increment @ rotations
        axes = np.maximum(self.axis + offsets, self.minimum_axis_length)
        axis_covariances = self.axis_covariances + conditional
        self._require_finite(rotations, "predicted rotations")
        self._require_finite(axes, "predicted semi_axes")
        self._require_finite(axis_covariances, "predicted axis covariance")
        self.rng.bit_generator.state = next_rng.bit_generator.state
        self.rotations = rotations
        self.axis = axes
        self.axis_covariances = axis_covariances
        self.kinematic_state = state
        self.covariance = covariance
        if self.log_prior_estimates:
            self.store_prior_estimates()
        if self.log_prior_extents:
            self.store_prior_extent()

    @staticmethod
    def _body_spread(axes, axis_covariances, source):
        return source * (axis_covariances + axes[:, :, None] * axes[:, None, :])

    def update(
        self,
        measurements,
        meas_mat=None,
        meas_noise_cov=None,
        multiplicative_noise_cov=None,
    ):
        """Assimilate one scan, forming a centroid and independent Gaussian contrasts."""
        points = self._normalize_measurements(measurements)
        n = points.shape[1]
        if n == 0:
            return
        matrix = self._get_measurement_matrix(meas_mat)
        sensor = self._as_covariance_matrix(
            0.0 if meas_noise_cov is None else meas_noise_cov, 3, "meas_noise_cov"
        )
        source = (
            self.multiplicative_noise_cov
            if multiplicative_noise_cov is None
            else self._as_multiplicative_noise(multiplicative_noise_cov)
        )
        body_from_world = self.rotations.swapaxes(1, 2)
        spread = self._body_spread(self.axis, self.axis_covariances, source)
        expected_spread = np.einsum(
            "p,pij->ij", self.weights, self.rotations @ spread @ body_from_world
        )
        cross = self.covariance @ matrix.T
        center_cov = matrix @ cross
        mean_noise = (expected_spread + sensor) / n
        gain = np.linalg.solve(center_cov + mean_noise, cross.T).T
        residuals = points - (matrix @ self.kinematic_state)[:, None]
        innovation = residuals.mean(axis=1)
        state = self.kinematic_state + gain @ innovation
        residual = np.eye(state.size) - gain @ matrix
        covariance = _positive_semidefinite(
            residual @ self.covariance @ residual.T + gain @ mean_noise @ gain.T
        )
        self._require_finite(state, "posterior kinematic state")
        self._require_finite(covariance, "posterior covariance")

        # An orthonormal transform of the scan preserves its Gaussian density.
        # Only the first transformed point contains uncertainty in the center.
        transformed = np.empty((3, n))
        transformed[:, 0] = np.sqrt(n) * innovation
        if n > 1:
            counts = np.arange(1, n)
            centered = residuals - innovation[:, None]
            transformed[:, 1:] = (
                centered[:, 1:] * counts - np.cumsum(centered[:, :-1], axis=1)
            ) / np.sqrt(counts * (counts + 1))
        local_sensor = body_from_world @ sensor @ self.rotations
        centroid_noise = local_sensor + n * (
            body_from_world @ center_cov @ self.rotations
        )
        axes = self.axis
        axis_covariances = self.axis_covariances
        log_weights = self._log_weights
        for index, observation in enumerate(transformed.T):
            noise = centroid_noise if index == 0 else local_sensor
            aligned = body_from_world @ observation
            marginal = self._body_spread(axes, axis_covariances, source) + noise
            chol = np.linalg.cholesky(marginal)
            whitened = np.linalg.solve(chol, aligned[..., None])[..., 0]
            log_likelihood = -np.log(chol.diagonal(axis1=1, axis2=2)).sum(
                axis=1
            ) - 0.5 * np.sum(whitened**2, axis=1)
            largest = np.max(log_likelihood)
            if not np.isfinite(largest):
                raise ValueError("observation has no finite particle likelihood")
            log_weights = log_weights + (log_likelihood - largest)
            normalizer = logsumexp(log_weights)
            if not np.isfinite(normalizer):
                raise ValueError("scan has no finite particle likelihood")
            log_weights -= normalizer
            mean, pseudo_cov, cross_cov = axis_squared_moments(
                axes, axis_covariances, source, noise
            )
            axis_gain = np.linalg.solve(pseudo_cov, cross_cov.swapaxes(1, 2)).swapaxes(
                1, 2
            )
            correction = (axis_gain @ (aligned**2 - mean)[..., None])[..., 0]
            axes = np.maximum(axes + correction, self.minimum_axis_length)
            axis_covariances = _positive_semidefinite(
                axis_covariances - axis_gain @ cross_cov.swapaxes(1, 2)
            )
        self._require_finite(axes, "posterior semi_axes")
        self._require_finite(axis_covariances, "posterior axis covariance")
        weights = np.exp(log_weights)
        rotations = self.rotations
        if 1 / (weights @ weights) <= self.resampling_threshold:
            rotations, axes, axis_covariances, log_weights = self._resampled_state(
                weights, axes, axis_covariances
            )
        self.rotations, self.axis, self.axis_covariances = (
            rotations,
            axes,
            axis_covariances,
        )
        self._log_weights = log_weights
        self.kinematic_state, self.covariance = state, covariance
        if self.log_posterior_estimates:
            self.store_posterior_estimates()
        if self.log_posterior_extents:
            self.store_posterior_extents()

    def _resampled_state(self, weights, axes, axis_covariances):
        cdf = np.cumsum(weights)
        cdf[-1] = 1.0
        positions = (self.rng.random() + np.arange(self.n_particles)) / self.n_particles
        indices = np.minimum(
            np.searchsorted(cdf, positions, side="right"), self.n_particles - 1
        )
        return (
            self.rotations[indices],
            axes[indices],
            axis_covariances[indices],
            np.full(self.n_particles, -np.log(self.n_particles)),
        )

    def resample(self):
        """Systematically resample complete conditional states and reset their weights."""
        self.rotations, self.axis, self.axis_covariances, self._log_weights = (
            self._resampled_state(self.weights, self.axis, self.axis_covariances)
        )


MemRbpf3DTracker = MEMRBPF3DTracker
