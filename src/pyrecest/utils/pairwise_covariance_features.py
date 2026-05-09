"""Pairwise Gaussian and covariance-derived association features.

The helpers in this module operate on plain mean and covariance arrays rather
than on tracker-, sensor-, or application-specific objects.  They are intended
for building feature tensors for association models and assignment costs from
Gaussian state summaries.

All stacked mean and covariance inputs use the standard PyRecEst convention:

* means have shape ``(dim, n_items)``;
* covariance stacks have shape ``(dim, dim, n_items)``.
"""

from __future__ import annotations

import math
from typing import Any

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import (
    abs,
    all as backend_all,
    asarray,
    einsum,
    exp,
    eye,
    float64,
    isfinite,
    log,
    maximum,
    moveaxis,
    reshape,
    sqrt,
    sum as backend_sum,
    trace,
    transpose,
    where,
    zeros,
)
from pyrecest.backend import linalg


def pairwise_mahalanobis_distances(
    means_a: Any,
    covariances_a: Any,
    means_b: Any,
    covariances_b: Any,
    *,
    regularization: float = 0.0,
) -> Any:
    """Return covariance-normalized distances between two Gaussian stacks.

    For a pair of items ``i`` and ``j``, the returned value is

    ``sqrt((mu_i - nu_j)^T (Sigma_i + Lambda_j + regularization * I)^+ (mu_i - nu_j))``,

    where ``^+`` denotes the Moore-Penrose pseudoinverse.  Using the summed
    covariance makes the feature symmetric in the two uncertain estimates and is
    the standard normalization for comparing two independent Gaussian position
    estimates.

    Parameters
    ----------
    means_a, means_b:
        Mean stacks with shape ``(dim, n_items)``.
    covariances_a, covariances_b:
        Covariance stacks with shape ``(dim, dim, n_items)``.  The number of
        covariance matrices must match the corresponding number of means.
    regularization:
        Optional non-negative diagonal loading added to every summed covariance
        before inversion.

    Returns
    -------
    array-like
        Matrix with shape ``(n_a, n_b)``.
    """

    if not math.isfinite(float(regularization)) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")

    means_a, covariances_a = _validate_mean_covariance_stack(
        "means_a", means_a, "covariances_a", covariances_a
    )
    means_b, covariances_b = _validate_mean_covariance_stack(
        "means_b", means_b, "covariances_b", covariances_b
    )

    dim, n_a = means_a.shape
    dim_b, n_b = means_b.shape
    if dim != dim_b:
        raise ValueError(
            "means_a and means_b must have the same leading dimension"
        )

    if n_a == 0 or n_b == 0:
        return zeros((n_a, n_b), dtype=float64)

    moved_covariances_a = _symmetrized_covariance_batch(covariances_a)
    moved_covariances_b = _symmetrized_covariance_batch(covariances_b)
    covariance_sums = (
        moved_covariances_a[:, None, :, :] + moved_covariances_b[None, :, :, :]
    )
    if regularization > 0.0:
        covariance_sums = covariance_sums + float(regularization) * eye(dim)[
            None, None, :, :
        ]

    mean_differences = transpose(means_a)[:, None, :] - transpose(means_b)[None, :, :]
    flat_differences = reshape(mean_differences, (n_a * n_b, dim))
    flat_covariances = reshape(covariance_sums, (n_a * n_b, dim, dim))
    inverse_covariances = linalg.pinv(flat_covariances)
    squared_distances = einsum(
        "ni,nij,nj->n", flat_differences, inverse_covariances, flat_differences
    )
    squared_distances = reshape(squared_distances, (n_a, n_b))
    return sqrt(maximum(squared_distances, 0.0))


def pairwise_covariance_shape_components(
    covariances_a: Any,
    covariances_b: Any,
    *,
    epsilon: float = 1.0e-6,
) -> tuple[Any, Any, Any]:
    """Return pairwise covariance shape, scale, and similarity components.

    The covariance-shape cost compares trace-normalized covariance matrices with
    a Frobenius norm.  This makes the feature sensitive to orientation and
    anisotropy while ignoring overall scale.  The log-determinant cost measures
    scale mismatch separately.  The shape similarity is ``exp(-shape_cost)``.

    Parameters
    ----------
    covariances_a, covariances_b:
        Covariance stacks with shape ``(dim, dim, n_items)``.
    epsilon:
        Strictly positive floor used for traces and determinants.

    Returns
    -------
    shape_cost, logdet_cost, shape_similarity:
        Three matrices with shape ``(n_a, n_b)``.
    """

    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")

    covariances_a = _validate_covariance_stack("covariances_a", covariances_a)
    covariances_b = _validate_covariance_stack("covariances_b", covariances_b)
    dim = covariances_a.shape[0]
    if covariances_b.shape[0] != dim:
        raise ValueError(
            "covariances_a and covariances_b must have the same matrix dimension"
        )

    n_a = covariances_a.shape[2]
    n_b = covariances_b.shape[2]
    if n_a == 0 or n_b == 0:
        empty = zeros((n_a, n_b), dtype=float64)
        return empty, empty, empty

    moved_covariances_a = _symmetrized_covariance_batch(covariances_a)
    moved_covariances_b = _symmetrized_covariance_batch(covariances_b)

    traces_a = _positive_floor(
        trace(moved_covariances_a, axis1=1, axis2=2), epsilon
    )
    traces_b = _positive_floor(
        trace(moved_covariances_b, axis1=1, axis2=2), epsilon
    )
    normalized_a = moved_covariances_a / traces_a[:, None, None]
    normalized_b = moved_covariances_b / traces_b[:, None, None]

    shape_differences = normalized_a[:, None, :, :] - normalized_b[None, :, :, :]
    frobenius_squared = backend_sum(
        backend_sum(shape_differences * shape_differences, axis=-1), axis=-1
    )
    shape_cost = sqrt(maximum(frobenius_squared, 0.0)) / sqrt(2.0)
    shape_similarity = exp(-shape_cost)

    determinants_a = _positive_floor(linalg.det(moved_covariances_a), epsilon)
    determinants_b = _positive_floor(linalg.det(moved_covariances_b), epsilon)
    logdet_cost = abs(log(determinants_a[:, None] / determinants_b[None, :]))
    return shape_cost, logdet_cost, shape_similarity


def _validate_mean_covariance_stack(
    mean_name: str,
    means: Any,
    covariance_name: str,
    covariances: Any,
) -> tuple[Any, Any]:
    means = asarray(means, dtype=float64)
    if means.ndim != 2:
        raise ValueError(f"{mean_name} must have shape (dim, n_items)")
    if not bool(backend_all(isfinite(means))):
        raise ValueError(f"{mean_name} must contain only finite values")

    covariances = _validate_covariance_stack(covariance_name, covariances)
    if covariances.shape[0] != means.shape[0]:
        raise ValueError(
            f"{covariance_name} matrix dimension must match the leading dimension of {mean_name}"
        )
    if covariances.shape[2] != means.shape[1]:
        raise ValueError(
            f"{covariance_name} must contain one covariance matrix per column of {mean_name}"
        )
    return means, covariances


def _validate_covariance_stack(name: str, covariances: Any) -> Any:
    covariances = asarray(covariances, dtype=float64)
    if covariances.ndim != 3 or covariances.shape[0] != covariances.shape[1]:
        raise ValueError(f"{name} must have shape (dim, dim, n_items)")
    if not bool(backend_all(isfinite(covariances))):
        raise ValueError(f"{name} must contain only finite values")
    return covariances


def _symmetrized_covariance_batch(covariances: Any) -> Any:
    moved = moveaxis(covariances, -1, 0)
    return 0.5 * (moved + transpose(moved, (0, 2, 1)))


def _positive_floor(values: Any, epsilon: float) -> Any:
    values = asarray(values, dtype=float64)
    return where(values > float(epsilon), values, float(epsilon))


__all__ = [
    "pairwise_covariance_shape_components",
    "pairwise_mahalanobis_distances",
]
