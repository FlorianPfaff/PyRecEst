"""Point-set registration utilities for registration-aware tracking.

This module is intentionally lightweight so it can be used directly in tracking
pipelines that operate on static landmarks such as neuron centroids or ROI
summaries across imaging sessions.

The main entry point is :func:`joint_registration_assignment`, which performs
alternating one-to-one assignment (Hungarian algorithm with optional gating) and
transform refitting. This provides a simple registration-aware matching block
that is directly useful for longitudinal neuron tracking where global drift,
rotation, or affine deformation must be estimated before data association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member,duplicate-code
from pyrecest.backend import (
    any,
    asarray,
    concatenate,
    eye,
    full,
    linalg,
    ones,
    quantile,
    sqrt,
    sum,
    zeros,
)

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

TransformModel = Literal["translation", "rigid", "affine"]
AssociationCostFn = Callable


@dataclass(frozen=True)
class AffineTransform:
    """An affine transform represented by its linear part and translation.

    Parameters
    ----------
    matrix:
        Linear part of shape ``(dim, dim)``.
    offset:
        Translation vector of shape ``(dim,)``.
    """

    matrix: Any
    offset: Any

    def __post_init__(self) -> None:
        matrix = asarray(self.matrix)
        offset = asarray(self.offset).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")
        if offset.shape[0] != matrix.shape[0]:
            raise ValueError("offset dimension must match matrix dimension.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    @property
    def dim(self) -> int:
        return self.offset.shape[0]

    @staticmethod
    def identity(dim: int) -> "AffineTransform":
        """Return the identity transform in ``dim`` dimensions."""
        if dim <= 0:
            raise ValueError("dim must be positive.")
        return AffineTransform(eye(dim), zeros(dim))

    def apply(self, points) -> Any:
        """Apply the transform to an ``(n_points, dim)`` array of points."""
        points_array = _as_point_array(points)
        if points_array.shape[1] != self.dim:
            raise ValueError("Point dimension does not match transform dimension.")
        return (self.matrix @ points_array.T).T + self.offset

    def inverse(self) -> "AffineTransform":
        """Return the inverse affine transform.

        For a transform ``y = A x + b``, the inverse is
        ``x = A^{-1} y - A^{-1} b``.
        """
        inverse_matrix = linalg.inv(self.matrix)
        inverse_offset = -(inverse_matrix @ self.offset)
        return AffineTransform(inverse_matrix, inverse_offset)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        """Return the composition of this transform with ``other``.

        The returned transform is equivalent to applying ``other`` first and
        this transform second, i.e. ``self.apply(other.apply(points))``.
        """
        if not isinstance(other, AffineTransform):
            raise TypeError("other must be an AffineTransform.")
        if other.dim != self.dim:
            raise ValueError("Transform dimensions must match.")
        return AffineTransform(
            self.matrix @ other.matrix,
            self.matrix @ other.offset + self.offset,
        )

    def homogeneous_matrix(self) -> Any:
        """Return the homogeneous representation of the affine transform."""
        transform = eye(self.dim + 1)
        transform[: self.dim, : self.dim] = self.matrix
        transform[: self.dim, -1] = self.offset
        return transform


@dataclass(frozen=True)
class RegistrationResult(
    RegistrationResultBase
):  # pylint: disable=too-many-instance-attributes
    """Result of alternating registration and assignment."""

    transform: AffineTransform


def _normalize_weights(weights, n_points):
    if weights is None:
        return full((n_points,), 1.0 / n_points)
    weights_array = asarray(weights).reshape(-1)
    if weights_array.shape[0] != n_points:
        raise ValueError("weights must have length n_points.")
    if any(weights_array < 0.0):
        raise ValueError("weights must be non-negative.")
    weight_sum = float(weights_array.sum())
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return weights_array / weight_sum


def _minimum_required_matches(model: TransformModel, dim: int) -> int:
    if model == "translation":
        return 1
    if model == "rigid":
        return max(2, dim)
    if model == "affine":
        return dim + 1
    raise ValueError(f"Unsupported transform model: {model}")


def estimate_transform(  # pylint: disable=too-many-locals
    source_points,
    target_points,
    *,
    model: TransformModel = "affine",
    weights=None,
    allow_reflection: bool = False,
) -> AffineTransform:
    """Estimate a transform from matched source/target point pairs.

    Parameters
    ----------
    source_points, target_points:
        Arrays of shape ``(n_points, dim)`` describing matched point pairs.
    model:
        ``"translation"``, ``"rigid"``, or ``"affine"``.
    weights:
        Optional non-negative per-point weights.
    allow_reflection:
        Only relevant for the rigid model. If ``False`` the returned rotation is
        constrained to have determinant ``+1``.
    """
    assert (
        pyrecest.backend.__backend_name__ != "jax"
    ), "Not supported for the JAX backend."

    source, target = _validate_pair(source_points, target_points)
    n_points, dim = source.shape
    min_matches = _minimum_required_matches(model, dim)
    if n_points < min_matches:
        raise ValueError(
            f"The '{model}' model requires at least {min_matches} matched points in {dim}D."
        )

    normalized_weights = _normalize_weights(weights, n_points)
    source_centroid = sum(normalized_weights[:, None] * source, axis=0)
    target_centroid = sum(normalized_weights[:, None] * target, axis=0)

    if model == "translation":
        return AffineTransform(eye(dim), target_centroid - source_centroid)

    if model == "rigid":
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        covariance = (normalized_weights[:, None] * source_centered).T @ target_centered
        left_singular_vectors, _, right_singular_vectors_transposed = linalg.svd(
            covariance
        )
        rotation = right_singular_vectors_transposed.T @ left_singular_vectors.T
        if linalg.det(rotation) < 0.0 and not allow_reflection:
            right_singular_vectors_transposed[-1, :] *= -1.0
            rotation = right_singular_vectors_transposed.T @ left_singular_vectors.T
        offset = target_centroid - rotation @ source_centroid
        return AffineTransform(rotation, offset)

    if model == "affine":
        design_matrix = concatenate([source, ones((n_points, 1))], axis=1)
        weighted_design_matrix = design_matrix * sqrt(normalized_weights)[:, None]
        weighted_targets = target * sqrt(normalized_weights)[:, None]
        coefficients = linalg.pinv(weighted_design_matrix) @ weighted_targets
        matrix = coefficients[:dim, :].T
        offset = coefficients[dim, :]
        return AffineTransform(matrix, offset)

    raise ValueError(f"Unsupported transform model: {model}")


def joint_registration_assignment(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    reference_points,
    moving_points,
    *,
    model: TransformModel = "affine",
    initial_transform: AffineTransform | None = None,
    max_cost: float = float("inf"),
    cost_function: AssociationCostFn | None = None,
    max_iterations: int = 25,
    tolerance: float = 1e-8,
    min_matches: int | None = None,
    allow_reflection: bool = False,
) -> RegistrationResult:
    """Alternating registration and one-to-one assignment.

    This function alternates between:

    1. assigning transformed reference points to moving points using the
       Hungarian algorithm with optional gating; and
    2. refitting the specified transform model using the current matches.

    Parameters
    ----------
    reference_points:
        Landmark locations from the reference session, shape ``(n_ref, dim)``.
    moving_points:
        Landmark locations from the moving/current session, shape ``(n_moving, dim)``.
    model:
        Registration model to fit: ``"translation"``, ``"rigid"``, or ``"affine"``.
    initial_transform:
        Optional starting transform. If omitted, a coordinate-wise median
        alignment is used as initialization. For challenging partial-overlap or
        high-outlier cases, providing an external coarse registration is
        recommended.
    max_cost:
        Optional gating threshold on the association cost.
    cost_function:
        Optional callable that receives transformed reference points and moving
        points and returns a cost matrix of shape ``(n_ref, n_moving)``. This
        makes it easy to plug in ROI-overlap or morphology-aware costs on top of
        centroid registration.
    max_iterations:
        Maximum number of alternating assignment/refit iterations.
    tolerance:
        Convergence threshold on the change of the affine parameters.
    min_matches:
        Minimum number of matched pairs required before refitting. Defaults to
        the identifiability threshold of the chosen model.
    allow_reflection:
        Passed through to :func:`estimate_transform` for the rigid model.
    """
    assert (
        pyrecest.backend.__backend_name__ != "jax"
    ), "Not supported for the JAX backend."

    reference = _as_point_array(reference_points)
    moving = _as_point_array(moving_points)
    if reference.shape[1] != moving.shape[1]:
        raise ValueError(
            "reference_points and moving_points must have the same dimension."
        )
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    dim = reference.shape[1]
    if min_matches is None:
        min_matches = _minimum_required_matches(model, dim)
    if min_matches <= 0:
        raise ValueError("min_matches must be positive.")

    if initial_transform is None:
        reference_location = quantile(reference, 0.5, axis=0)
        moving_location = quantile(moving, 0.5, axis=0)
        initial_transform = AffineTransform(
            eye(dim), moving_location - reference_location
        )
    elif initial_transform.dim != dim:
        raise ValueError("initial_transform dimension must match the point dimension.")

    def _fit_transform(matched_reference, matched_moving):
        return estimate_transform(
            matched_reference,
            matched_moving,
            model=model,
            allow_reflection=allow_reflection,
        )

    def _compute_change(
        previous_transform,
        updated_transform,
        _reference,
        _transformed_reference,
    ):
        return max(
            float(linalg.norm(updated_transform.matrix - previous_transform.matrix)),
            float(linalg.norm(updated_transform.offset - previous_transform.offset)),
        )

    loop_state = run_registration_loop(
        reference,
        moving,
        initial_transform,
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
    return build_registration_result(RegistrationResult, loop_state)
