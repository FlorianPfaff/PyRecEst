"""Replay particle-filter helpers for grid-valued position likelihoods.

The utilities in this module are intentionally filter-adapter oriented: they
turn a log likelihood evaluated on a position grid into callable particle
likelihood updates for goal-conditioned replay particle filters.  The functions
also expose reusable interpolation, proposal-weight, ESS, and posterior-grid
helpers so application code does not need to duplicate this logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import logsumexp


@dataclass(frozen=True)
class ReplayGridLikelihoodLookup:
    """Lookup data for evaluating grid log likelihoods at particle positions.

    ``method`` is either ``"nearest"`` or ``"linear"``.  Linear interpolation is
    used only for complete two-dimensional rectilinear grids; unsupported grids
    automatically fall back to nearest-neighbour lookup.
    """

    method: str
    x_values: np.ndarray | None = None
    y_values: np.ndarray | None = None
    grid_indices: np.ndarray | None = None


__all__ = (
    "ReplayGridLikelihoodLookup",
    "adaptive_position_proposal_probability",
    "build_replay_grid_likelihood_lookup",
    "effective_sample_size_fraction",
    "grid_proposal_weights",
    "particle_position_log_posterior",
    "replay_grid_log_likelihood_values",
    "update_position_grid_likelihood",
)


def build_replay_grid_likelihood_lookup(
    bin_centers,
    method: str = "linear",
) -> ReplayGridLikelihoodLookup:
    """Prepare interpolation metadata for a position-grid log likelihood.

    Parameters
    ----------
    bin_centers:
        Matrix with shape ``(n_bins, position_dim)`` containing grid point
        coordinates.
    method:
        ``"linear"`` for bilinear interpolation on complete two-dimensional
        rectilinear grids, or ``"nearest"`` for nearest-neighbour lookup.
    """

    method = str(method).lower()
    if method not in {"nearest", "linear"}:
        raise ValueError("method must be 'nearest' or 'linear'")

    bin_centers = _coerce_bin_centers(bin_centers)
    if method == "nearest" or bin_centers.shape[1] != 2:
        return ReplayGridLikelihoodLookup(method="nearest")

    x_values = np.unique(bin_centers[:, 0])
    y_values = np.unique(bin_centers[:, 1])
    if x_values.size < 2 or y_values.size < 2:
        return ReplayGridLikelihoodLookup(method="nearest")
    if x_values.size * y_values.size != bin_centers.shape[0]:
        return ReplayGridLikelihoodLookup(method="nearest")

    x_index = {float(value): index for index, value in enumerate(x_values)}
    y_index = {float(value): index for index, value in enumerate(y_values)}
    grid_indices = np.full((x_values.size, y_values.size), -1, dtype=int)
    for flat_index, center in enumerate(bin_centers):
        try:
            grid_indices[x_index[float(center[0])], y_index[float(center[1])]] = (
                flat_index
            )
        except KeyError:
            return ReplayGridLikelihoodLookup(method="nearest")

    if np.any(grid_indices < 0):
        return ReplayGridLikelihoodLookup(method="nearest")

    return ReplayGridLikelihoodLookup(
        method="linear",
        x_values=x_values,
        y_values=y_values,
        grid_indices=grid_indices,
    )


def replay_grid_log_likelihood_values(
    positions,
    log_likelihood,
    bin_centers,
    *,
    lookup: ReplayGridLikelihoodLookup | None = None,
    bin_tree: cKDTree | None = None,
    interpolation: str = "linear",
    log_zero: float = float("-inf"),
) -> np.ndarray:
    """Evaluate grid log-likelihood values at arbitrary positions."""

    bin_centers = _coerce_bin_centers(bin_centers)
    positions = _coerce_positions(positions, bin_centers.shape[1], "positions")
    values = _coerce_grid_values(log_likelihood, bin_centers.shape[0])
    if lookup is None:
        lookup = build_replay_grid_likelihood_lookup(bin_centers, interpolation)
    if bin_tree is None:
        bin_tree = cKDTree(bin_centers)

    if lookup.method != "linear":
        return _nearest_grid_values(positions, values, bin_tree, log_zero=log_zero)

    interpolated = _linear_rectilinear_grid_values(positions, values, lookup)
    if np.all(np.isfinite(interpolated)):
        return interpolated

    nearest = _nearest_grid_values(positions, values, bin_tree, log_zero=log_zero)
    return np.where(np.isfinite(interpolated), interpolated, nearest)


def update_position_grid_likelihood(
    filter_: Any,
    log_likelihood,
    bin_centers,
    *,
    interpolation: str = "linear",
    lookup: ReplayGridLikelihoodLookup | None = None,
    bin_tree: cKDTree | None = None,
    position_proposal_probability: float = 0.0,
    position_proposal_ess_threshold: float | None = None,
) -> float:
    """Update a replay particle filter from grid-valued position log likelihood.

    The target filter must expose ``position_particles``, ``filter_state.w``,
    and ``update_position_likelihood(...)``.  When proposal rejuvenation is
    requested, the filter must also expose
    ``update_position_likelihood_with_proposal(...)``.
    """

    bin_centers = _coerce_bin_centers(bin_centers)
    values = _coerce_grid_values(log_likelihood, bin_centers.shape[0])
    if lookup is None:
        lookup = build_replay_grid_likelihood_lookup(bin_centers, interpolation)
    if bin_tree is None:
        bin_tree = cKDTree(bin_centers)

    proposal_probability, _ = adaptive_position_proposal_probability(
        filter_,
        position_proposal_probability,
        position_proposal_ess_threshold,
    )

    def log_likelihood_at(positions) -> np.ndarray:
        return replay_grid_log_likelihood_values(
            positions,
            values,
            bin_centers,
            lookup=lookup,
            bin_tree=bin_tree,
            interpolation=interpolation,
        )

    particle_log_likelihood = log_likelihood_at(filter_.position_particles)
    finite_particles = np.isfinite(particle_log_likelihood)

    if proposal_probability > 0.0:
        finite_grid_values = values[np.isfinite(values)]
        if finite_grid_values.size == 0:
            raise ValueError("all grid log-likelihoods are non-finite")
        max_log = float(np.max(finite_grid_values))
        if np.any(finite_particles):
            max_log = max(
                max_log, float(np.max(particle_log_likelihood[finite_particles]))
            )
    else:
        if not np.any(finite_particles):
            raise ValueError("all particle log-likelihoods are non-finite")
        max_log = float(np.max(particle_log_likelihood[finite_particles]))

    def scaled_likelihood(positions) -> np.ndarray:
        position_log_likelihood = log_likelihood_at(positions)
        return np.exp(np.clip(position_log_likelihood - max_log, -745.0, 0.0))

    if proposal_probability > 0.0:
        if not hasattr(filter_, "update_position_likelihood_with_proposal"):
            raise AttributeError(
                "filter_ must implement update_position_likelihood_with_proposal when position proposals are requested"
            )
        update_log = filter_.update_position_likelihood_with_proposal(
            scaled_likelihood,
            position_proposal=bin_centers,
            proposal_weights=grid_proposal_weights(values),
            proposal_probability=proposal_probability,
            return_log_marginal=True,
        )
    else:
        update_log = filter_.update_position_likelihood(
            scaled_likelihood,
            return_log_marginal=True,
        )
    return max_log + float(update_log)


def effective_sample_size_fraction(weights) -> float:
    """Return ESS divided by the number of weights."""

    weights = _coerce_particle_weights(weights)
    if weights.size == 0:
        return 0.0
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    normalized = weights / total
    ess = 1.0 / float(np.sum(normalized * normalized))
    return float(ess / weights.size)


def adaptive_position_proposal_probability(
    filter_or_weights,
    base_probability: float,
    ess_threshold: float | None,
) -> tuple[float, float]:
    """Gate proposal rejuvenation by the current ESS fraction."""

    base_probability = _validate_probability(base_probability, "base_probability")
    if ess_threshold is not None:
        ess_threshold = _validate_probability(ess_threshold, "ess_threshold")

    weights = getattr(
        getattr(filter_or_weights, "filter_state", None), "w", filter_or_weights
    )
    ess_fraction = effective_sample_size_fraction(weights)
    if base_probability <= 0.0:
        return 0.0, ess_fraction
    if base_probability >= 1.0:
        return base_probability, ess_fraction
    if ess_threshold is None or ess_fraction < ess_threshold:
        return base_probability, ess_fraction
    return 0.0, ess_fraction


def grid_proposal_weights(log_likelihood) -> np.ndarray:
    """Convert finite grid log likelihoods to normalized proposal weights."""

    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("all grid log-likelihoods are non-finite")
    weights = np.zeros(values.shape, dtype=float)
    weights[finite] = np.exp(values[finite] - float(logsumexp(values[finite])))
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("grid proposal weights have no mass")
    return weights / total


def particle_position_log_posterior(
    positions,
    weights,
    bin_centers,
    *,
    bin_tree: cKDTree | None = None,
    log_zero: float = float("-inf"),
) -> np.ndarray:
    """Accumulate weighted position particles onto a grid as log posterior mass."""

    bin_centers = _coerce_bin_centers(bin_centers)
    positions = _coerce_positions(positions, bin_centers.shape[1], "positions")
    weights = _coerce_particle_weights(weights)
    if weights.shape != (positions.shape[0],):
        raise ValueError("weights must contain one entry per position particle")
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("particle weights must have positive finite total mass")
    if bin_tree is None:
        bin_tree = cKDTree(bin_centers)

    normalized = weights / total
    indices = _nearest_bin_indices(positions, bin_tree)
    masses = np.zeros(bin_centers.shape[0], dtype=float)
    np.add.at(masses, indices, normalized)
    if not np.any(masses > 0.0):
        raise ValueError("particle posterior has no mass")

    log_posterior = np.full(bin_centers.shape[0], float(log_zero), dtype=float)
    positive = masses > 0.0
    log_posterior[positive] = np.log(masses[positive])
    return log_posterior - float(logsumexp(log_posterior))


def _coerce_particle_weights(weights) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("particle weights must be finite")
    if np.any(weights < 0.0):
        raise ValueError("particle weights must be nonnegative")
    return weights


def _coerce_bin_centers(bin_centers) -> np.ndarray:
    bin_centers = np.asarray(bin_centers, dtype=float)
    if bin_centers.ndim != 2:
        raise ValueError("bin_centers must have shape (n_bins, position_dim)")
    if bin_centers.shape[0] == 0 or bin_centers.shape[1] == 0:
        raise ValueError(
            "bin_centers must contain at least one point and one dimension"
        )
    if not np.all(np.isfinite(bin_centers)):
        raise ValueError("bin_centers must be finite")
    return bin_centers


def _coerce_positions(positions, expected_dim: int, name: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    if positions.ndim == 0:
        positions = positions.reshape(1, 1)
    elif positions.ndim == 1:
        if expected_dim == 1:
            positions = positions.reshape(-1, 1)
        else:
            positions = positions.reshape(1, -1)
    if positions.ndim != 2 or positions.shape[1] != expected_dim:
        raise ValueError(f"{name} must have shape (n_positions, {expected_dim})")
    return positions


def _coerce_grid_values(values, expected_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (expected_size,):
        raise ValueError(f"log_likelihood must have shape ({expected_size},)")
    return values


def _linear_rectilinear_grid_values(
    positions: np.ndarray,
    values: np.ndarray,
    lookup: ReplayGridLikelihoodLookup,
) -> np.ndarray:
    if (
        lookup.x_values is None
        or lookup.y_values is None
        or lookup.grid_indices is None
    ):
        return np.full(positions.shape[0], np.nan, dtype=float)

    x_values = lookup.x_values
    y_values = lookup.y_values
    grid_values = values[lookup.grid_indices]

    x = positions[:, 0]
    y = positions[:, 1]
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= x_values[0])
        & (x <= x_values[-1])
        & (y >= y_values[0])
        & (y <= y_values[-1])
    )
    output = np.full(positions.shape[0], np.nan, dtype=float)
    if not np.any(inside):
        return output

    x0_index = np.searchsorted(x_values, x[inside], side="right") - 1
    y0_index = np.searchsorted(y_values, y[inside], side="right") - 1
    x0_index = np.clip(x0_index, 0, x_values.size - 2)
    y0_index = np.clip(y0_index, 0, y_values.size - 2)
    x1_index = x0_index + 1
    y1_index = y0_index + 1

    x0 = x_values[x0_index]
    x1 = x_values[x1_index]
    y0 = y_values[y0_index]
    y1 = y_values[y1_index]
    tx = (x[inside] - x0) / (x1 - x0)
    ty = (y[inside] - y0) / (y1 - y0)

    v00 = grid_values[x0_index, y0_index]
    v10 = grid_values[x1_index, y0_index]
    v01 = grid_values[x0_index, y1_index]
    v11 = grid_values[x1_index, y1_index]
    valid = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
    interpolated = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    inside_indices = np.flatnonzero(inside)
    output[inside_indices[valid]] = interpolated[valid]
    return output


def _nearest_grid_values(
    positions: np.ndarray,
    values: np.ndarray,
    bin_tree: cKDTree,
    *,
    log_zero: float,
) -> np.ndarray:
    finite_positions = np.isfinite(positions).all(axis=1)
    output = np.full(positions.shape[0], float(log_zero), dtype=float)
    if not np.any(finite_positions):
        return output

    indices = _nearest_bin_indices(positions[finite_positions], bin_tree)
    nearest_values = values[indices]
    if not np.all(np.isfinite(nearest_values)):
        finite_values = values[np.isfinite(values)]
        replacement = (
            float(np.min(finite_values)) if finite_values.size else float(log_zero)
        )
        nearest_values = np.where(
            np.isfinite(nearest_values), nearest_values, replacement
        )
    output[finite_positions] = nearest_values
    return output


def _nearest_bin_indices(positions: np.ndarray, bin_tree: cKDTree) -> np.ndarray:
    _, indices = bin_tree.query(positions, k=1)
    return np.asarray(indices, dtype=int)


def _validate_probability(probability: float, name: str) -> float:
    probability_array = np.asarray(probability)
    if probability_array.shape != ():
        raise ValueError(f"{name} must be a scalar probability in [0, 1]")
    probability_scalar = probability_array.item()
    if isinstance(probability_scalar, (bool, np.bool_)):
        raise ValueError(f"{name} must be a scalar probability in [0, 1]")
    try:
        value = float(probability_scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar probability in [0, 1]") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value
