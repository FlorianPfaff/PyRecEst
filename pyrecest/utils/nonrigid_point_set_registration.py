"""Thin-plate-spline point-set registration utilities.

This module adds a smooth non-rigid registration primitive that is useful when
pairwise rigid or affine alignment is not expressive enough. It is intended for
registration-aware tracking problems such as longitudinal neuron identity
tracking, where ROI centroids can undergo local distortions between sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,duplicate-code
from pyrecest.backend import (
    asarray,
    concatenate,
    eye,
    linalg,
    log,
    maximum,
    ones,
    quantile,
    where,
    zeros,
)
from scipy.spatial.distance import cdist

from ._point_set_registration_common import (
    RegistrationLoopCallbacks,
    RegistrationLoopConfig,
    RegistrationResultBase,
)
from ._point_set_registration_common import as_point_array as _as_point_array
from ._point_set_registration_common import (
    build_registration_result,
    run_registration_loop,
    solve_gated_assignment,
)
from ._point_set_registration_common import validate_pair as _validate_pair

NonRigidAssociationCostFn = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class ThinPlateSplineTransform:
    """Two-dimensional thin-plate-spline transform.

    Parameters
    ----------
    control_points:
        Control points with shape ``(n_control, 2)``.
    weights:
        Non-rigid TPS weights with shape ``(n_control, 2)``.
    affine_coefficients:
        Affine coefficients with shape ``(3, 2)`` acting on
        ``[1, x, y]``.
    """

    control_points: object
    weights: object
    affine_coefficients: object

    def __post_init__(self) -> None:
        control_points = asarray(self.control_points)
        weights = asarray(self.weights)
        affine_coefficients = asarray(self.affine_coefficients)

        if control_points.ndim != 2:
            raise ValueError("control_points must be two-dimensional.")
        if control_points.shape[1] != 2:
            raise ValueError(
                "ThinPlateSplineTransform currently supports 2D points only."
            )
        if weights.shape != control_points.shape:
            raise ValueError("weights must have the same shape as control_points.")
        if affine_coefficients.shape != (3, 2):
            raise ValueError("affine_coefficients must have shape (3, 2).")

        object.__setattr__(self, "control_points", control_points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "affine_coefficients", affine_coefficients)

    @staticmethod
    def identity() -> "ThinPlateSplineTransform":
        """Return the identity 2D TPS transform."""
        return ThinPlateSplineTransform(
            zeros((0, 2)),
            zeros((0, 2)),
            asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        )

    @staticmethod
    def from_translation(translation) -> "ThinPlateSplineTransform":
        """Return a TPS transform representing a pure translation."""
        offset = asarray(translation).reshape(-1)
        if offset.shape[0] != 2:
            raise ValueError("translation must be two-dimensional.")
        return ThinPlateSplineTransform(
            zeros((0, 2)),
            zeros((0, 2)),
            asarray([[offset[0], offset[1]], [1.0, 0.0], [0.0, 1.0]]),
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the transform domain."""
        return 2

    def apply(self, points):
        """Apply the transform to an ``(n_points, 2)`` array of points."""
        point_array = _as_point_array(
            points,
            expected_dim=2,
            expected_dim_error="Only 2D point sets are currently supported.",
        )

        basis = zeros((point_array.shape[0], 0))
        if self.control_points.shape[0] > 0:
            distances = asarray(
                cdist(point_array, self.control_points, metric="euclidean")
            )
            basis = _tps_kernel_from_distances(distances)

        polynomial = concatenate([ones((point_array.shape[0], 1)), point_array], axis=1)
        return polynomial @ self.affine_coefficients + basis @ self.weights


@dataclass(frozen=True)
class ThinPlateSplineRegistrationResult(
    RegistrationResultBase
):  # pylint: disable=too-many-instance-attributes
    """Result of alternating TPS registration and assignment."""

    transform: ThinPlateSplineTransform


def _tps_kernel_from_distances(distances, epsilon: float = 1e-12):
    squared_distances = distances * distances
    kernel = squared_distances * log(maximum(squared_distances, epsilon))
    return where(squared_distances > 0.0, kernel, 0.0)


def estimate_thin_plate_spline(
    source_points,
    target_points,
    *,
    regularization: float = 1e-3,
) -> ThinPlateSplineTransform:
    """Estimate a thin-plate-spline transform from matched 2D point pairs.

    Parameters
    ----------
    source_points, target_points:
        Arrays of shape ``(n_points, 2)`` describing matched point pairs.
    regularization:
        Non-negative ridge penalty applied to the TPS kernel matrix.
    """
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "estimate_thin_plate_spline is not supported on the JAX backend."
        )

    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")

    source, target = _validate_pair(
        source_points,
        target_points,
        expected_dim=2,
        expected_dim_error="Only 2D point sets are currently supported.",
    )
    n_points = source.shape[0]

    if n_points < 3:
        raise ValueError(
            "At least three matched 2D points are required for TPS fitting."
        )

    kernel = _tps_kernel_from_distances(
        asarray(cdist(source, source, metric="euclidean"))
    )
    polynomial = concatenate([ones((n_points, 1)), source], axis=1)

    lhs = zeros((n_points + 3, n_points + 3))
    lhs[:n_points, :n_points] = kernel + regularization * eye(n_points)
    lhs[:n_points, n_points:] = polynomial
    lhs[n_points:, :n_points] = polynomial.T

    rhs = zeros((n_points + 3, 2))
    rhs[:n_points, :] = target

    coefficients = linalg.pinv(lhs) @ rhs
    weights = coefficients[:n_points, :]
    affine_coefficients = coefficients[n_points:, :]

    return ThinPlateSplineTransform(
        control_points=source,
        weights=weights,
        affine_coefficients=affine_coefficients,
    )


def joint_tps_registration_assignment(  # pylint: disable=too-many-arguments,too-many-locals
    reference_points,
    moving_points,
    *,
    initial_transform: ThinPlateSplineTransform | None = None,
    max_cost: float = float("inf"),
    cost_function: NonRigidAssociationCostFn | None = None,
    max_iterations: int = 25,
    tolerance: float = 1e-6,
    min_matches: int = 3,
    regularization: float = 1e-3,
) -> ThinPlateSplineRegistrationResult:
    """Alternating thin-plate-spline registration and one-to-one assignment.

    This function alternates between:
      1. assigning transformed reference points to moving points using the
         Hungarian algorithm with optional gating; and
      2. refitting a smooth thin-plate-spline warp from the current matches.

    Parameters
    ----------
    reference_points:
        Landmark locations from the reference session, shape ``(n_ref, 2)``.
    moving_points:
        Landmark locations from the moving/current session, shape ``(n_moving, 2)``.
    initial_transform:
        Optional starting transform. If omitted, the transform is initialized
        with a robust median-based translation.
    max_cost:
        Optional gating threshold on the association cost.
    cost_function:
        Optional callable receiving transformed reference points and moving
        points and returning a cost matrix of shape ``(n_ref, n_moving)``.
        This allows centroid costs, ROI overlap costs, or morphology-aware
        hybrid costs to be plugged into the registration loop.
    max_iterations:
        Maximum number of alternating assignment/refit iterations.
    tolerance:
        Convergence threshold on the change of the transformed reference point set.
    min_matches:
        Minimum number of matched pairs required before refitting the TPS warp.
    regularization:
        Non-negative ridge penalty for TPS fitting.
    """
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "joint_tps_registration_assignment is not supported on the JAX backend."
        )

    reference = _as_point_array(
        reference_points,
        expected_dim=2,
        expected_dim_error="Only 2D point sets are currently supported.",
    )
    moving = _as_point_array(
        moving_points,
        expected_dim=2,
        expected_dim_error="Only 2D point sets are currently supported.",
    )

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if min_matches < 3:
        raise ValueError("min_matches must be at least 3 for TPS fitting.")

    if initial_transform is None:
        translation = quantile(moving, 0.5, axis=0) - quantile(reference, 0.5, axis=0)
        transform = ThinPlateSplineTransform.from_translation(translation)
    else:
        if initial_transform.dim != 2:
            raise ValueError(
                "initial_transform dimension must match the point dimension."
            )
        transform = initial_transform

    def _fit_transform(matched_reference, matched_moving):
        return estimate_thin_plate_spline(
            matched_reference,
            matched_moving,
            regularization=regularization,
        )

    def _compute_change(
        _previous_transform,
        updated_transform,
        reference_points_array,
        transformed_reference,
    ):
        updated_transformed_reference = updated_transform.apply(reference_points_array)
        return float(linalg.norm(updated_transformed_reference - transformed_reference))

    loop_state = run_registration_loop(
        reference,
        moving,
        transform,
        RegistrationLoopConfig(
            max_cost=max_cost,
            cost_function=cost_function,
            max_iterations=max_iterations,
            min_matches=min_matches,
            tolerance=tolerance,
        ),
        RegistrationLoopCallbacks(
            fit_transform=_fit_transform,
            compute_change=_compute_change,
            assignment_solver=solve_gated_assignment,
        ),
    )
    return build_registration_result(ThinPlateSplineRegistrationResult, loop_state)
