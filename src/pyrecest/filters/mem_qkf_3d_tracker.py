"""Decoupled quadratic Kalman filtering for a spatial ellipsoid."""

from itertools import product
from math import prod, sqrt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, einsum, eye, linalg, maximum, sin, stack, transpose

from ._spatial_mem_tracker import axis_squared_moments
from .mem_ekf_star_3d_tracker import MEMEKFStar3DTracker

# Tensor Gauss-Hermite rule for a standard 3-D normal (positive weights).
_NODES = tuple(product((-sqrt(3.0), 0.0, sqrt(3.0)), repeat=3))
_WEIGHTS = tuple(prod(2 / 3 if x == 0 else 1 / 6 for x in node) for node in _NODES)


class MEMQKF3DTracker(MEMEKFStar3DTracker):
    """Sequential decoupled MEM-QKF for one oriented 3-D ellipsoid.

    Construction, prediction, measurement layouts, and geometric output
    follow :class:`MEMEKFStar3DTracker`. The 6-by-6 shape covariance uses
    body-frame rotation errors followed by additive semi-axis errors. As in
    planar MEM-QKF, the orientation and axis blocks evolve separately.
    Cross-block covariances in the initial state and process noise are dropped.

    Each point updates kinematics using the nominal shape. Both shape updates
    use the same prior innovation and include projected center uncertainty in
    the additive covariance. A biased prior center can therefore influence
    the inferred extent until later points improve the center estimate.
    This spatial variant always processes points sequentially.

    Orientation uses the six distinct quadratic residuals with axes fixed at
    their prior mean. A 27-node Gauss-Hermite rule integrates their conditional
    Gaussian moments over right rotation-vector errors on SO(3). Axes use the
    three squared body-frame residuals with orientation fixed at its prior
    mean, including the fourth moments from Gaussian-uncertain axes. Gaussian
    source and sensor moments are assumed. The default source covariance
    I / 5 also matches the second moment of a uniform solid unit ball.
    """

    def __init__(
        self,
        kinematic_state,
        covariance,
        orientation,
        semi_axes,
        shape_covariance,
        measurement_matrix=None,
        multiplicative_noise_cov=None,
        covariance_regularization=0.0,
        minimum_axis_length=1e-9,
        log_prior_estimates=False,
        log_posterior_estimates=False,
        log_prior_extents=False,
        log_posterior_extents=False,
    ):
        super().__init__(
            kinematic_state,
            covariance,
            orientation,
            semi_axes,
            shape_covariance,
            measurement_matrix=measurement_matrix,
            multiplicative_noise_cov=multiplicative_noise_cov,
            covariance_regularization=covariance_regularization,
            minimum_axis_length=minimum_axis_length,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
            log_prior_extents=log_prior_extents,
            log_posterior_extents=log_posterior_extents,
        )
        self.shape_covariance = self._decoupled(self.shape_covariance)
        self._nodes = array(_NODES, dtype=self.orientation.dtype)
        self._weights = array(_WEIGHTS, dtype=self.orientation.dtype)
        self._skew_basis = stack([self._skew(axis) for axis in eye(3)])

    @staticmethod
    def _decoupled(covariance):
        return linalg.block_diag(covariance[:3, :3], covariance[3:, 3:])

    def predict_linear(
        self,
        system_matrix,
        sys_noise=None,
        inputs=None,
        shape_sys_noise=None,
        rotation_increment=None,
    ):
        """Transport the orientation covariance and add decoupled shape noise."""
        if shape_sys_noise is not None:
            shape_sys_noise = self._decoupled(
                self._as_covariance_matrix(shape_sys_noise, 6, "shape_sys_noise")
            )
        super().predict_linear(
            system_matrix, sys_noise, inputs, shape_sys_noise, rotation_increment
        )

    def _orientation_moments(self, source_spread, additive_covariance):
        """Integrate quadratic residual moments and their rotation cross-covariance."""
        values, vectors = linalg.eigh(self.shape_covariance[:3, :3])
        root = (vectors * maximum(values, 0.0) ** 0.5) @ vectors.T
        deltas = self._nodes @ root

        # Batched Rodrigues formula. The half-angle form avoids cancellation
        # in (1 - cos(t))/t**2 and handles the zero node continuously.
        skew = einsum("ni,ijk->njk", deltas, self._skew_basis)
        angle = maximum(einsum("ni,ni->n", deltas, deltas) ** 0.5, 1e-15)[:, None, None]
        rotations = (
            eye(3)
            + (sin(angle) / angle) * skew
            + 0.5 * (sin(angle / 2) / (angle / 2)) ** 2 * (skew @ skew)
        )
        frames = self.orientation @ rotations
        conditional = frames @ source_spread @ transpose(frames, (0, 2, 1))
        conditional = conditional + additive_covariance
        i, j = (array(indices) for indices in zip(*self._quadratic_pairs))
        products = conditional[:, i, j]
        weights = self._weights
        mean = weights @ products
        centered = products - mean

        # E[Cov(e_i e_j, e_k e_l | rotation)] plus covariance of the means.
        second = einsum("n,nij,nkl->ijkl", weights, conditional, conditional)
        fourth = (
            second[i[:, None], i, j[:, None], j] + second[i[:, None], j, j[:, None], i]
        )
        covariance = fourth + centered.T @ (weights[:, None] * centered)
        cross_covariance = deltas.T @ (weights[:, None] * centered)
        return mean, self._symmetrize(covariance), cross_covariance

    def _correction(
        self, innovation, innovation_covariance, cross_covariance, prior_covariance
    ):
        innovation_covariance = (
            innovation_covariance
            + self.covariance_regularization * eye(innovation.shape[0])
        )
        gain = linalg.solve(innovation_covariance, cross_covariance.T).T
        covariance = self._project_covariance(
            prior_covariance - gain @ cross_covariance.T
        )
        return gain @ innovation, covariance

    def _update_single_measurement(
        self, measurement, measurement_matrix, measurement_noise, multiplicative_noise
    ):
        residual = measurement - measurement_matrix @ self.kinematic_state
        position_cross = self.covariance @ measurement_matrix.T
        additive = self._symmetrize(
            measurement_matrix @ position_cross + measurement_noise
        )
        source_spread = self.semi_axes[:, None] * multiplicative_noise * self.semi_axes
        position_spread = (
            self.orientation @ source_spread @ self.orientation.T + additive
        )
        dx, covariance = self._correction(
            residual, position_spread, position_cross, self.covariance
        )

        mean, quadratic_cov, cross_cov = self._orientation_moments(
            source_spread, additive
        )
        products = stack([residual[i] * residual[j] for i, j in self._quadratic_pairs])
        dtheta, orientation_cov = self._correction(
            products - mean, quadratic_cov, cross_cov, self.shape_covariance[:3, :3]
        )
        reset = self._right_jacobian(dtheta)
        orientation = self.orientation @ self._rotation_exp(dtheta)
        orientation_cov = self._project_covariance(reset @ orientation_cov @ reset.T)

        body_residual = self.orientation.T @ residual
        axis_cov = self.shape_covariance[3:, 3:]
        mean, quadratic_cov, cross_cov = axis_squared_moments(
            self.semi_axes,
            axis_cov,
            multiplicative_noise,
            self.orientation.T @ additive @ self.orientation,
        )
        da, axis_cov = self._correction(
            body_residual**2 - mean, quadratic_cov, cross_cov, axis_cov
        )
        semi_axes = maximum(self.semi_axes + da, self.minimum_axis_length)
        shape_covariance = linalg.block_diag(orientation_cov, axis_cov)
        kinematic_state = self.kinematic_state + dx
        for value in (
            kinematic_state,
            covariance,
            orientation,
            semi_axes,
            shape_covariance,
        ):
            self._require_finite(value, "posterior state")
        self.kinematic_state = kinematic_state
        self.covariance = covariance
        self.orientation = orientation
        self.semi_axes = semi_axes
        self.shape_covariance = shape_covariance


MemQkf3DTracker = MEMQKF3DTracker
