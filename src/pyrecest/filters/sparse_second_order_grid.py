"""Exact sparse second-order grid evidence utilities.

The latent state is represented as sparse pair states ``(x[t-1], x[t])``.
Callers provide the initial pair weights and sparse transition rows.  The
utility computes a scaled forward likelihood exactly over the declared finite
support and can optionally return fixed-interval smoothed single-state
marginals.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyrecest.evidence import EvidenceComputationMode, resolve_evidence_computation_mode

from .discrete_state import probabilities_to_log_probabilities, scaled_emissions
from .sparse_transition_cache import SparseTransitionRowCache

SparseTransitionRows = list[tuple[np.ndarray, np.ndarray]]
SparsePairInitializer = Callable[
    [np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, list[int] | np.ndarray]
]
SparsePairTransitionRowBuilder = Callable[
    [int, int, int], tuple[np.ndarray, np.ndarray]
]
SparsePairTransitionCacheKeyBuilder = Callable[[int, int, int], Hashable | None]


@dataclass(frozen=True)
class SparseSecondOrderGridResult:
    """Result from :func:`sparse_second_order_grid_evidence`.

    Attributes
    ----------
    log_marginal_likelihood:
        Log evidence for the whole emission sequence.
    terminal_log_probabilities:
        Normalized log posterior over the final single grid state.
    smoothed_log_probabilities:
        Optional normalized smoothed single-state log marginals with shape
        ``(n_time, n_states)``.  ``None`` when ``return_smoothed=False``.
    filtered_pair_previous, filtered_pair_current, filtered_pair_probabilities:
        Sparse pair-state filtering lattice.  These are ``None`` unless
        ``return_pair_lattice=True``.
    scales:
        Forward scaling constants after row-wise emission offsets were removed.
    emission_offsets:
        Per-time emission offsets subtracted before exponentiating likelihoods.
    diagnostics:
        Support-size and cache diagnostics.
    """

    log_marginal_likelihood: float
    terminal_log_probabilities: np.ndarray
    smoothed_log_probabilities: np.ndarray | None
    filtered_pair_previous: tuple[np.ndarray, ...] | None
    filtered_pair_current: tuple[np.ndarray, ...] | None
    filtered_pair_probabilities: tuple[np.ndarray, ...] | None
    scales: np.ndarray
    emission_offsets: np.ndarray
    diagnostics: dict[str, int | float | str]


def sparse_second_order_grid_evidence(
    log_likelihood: np.ndarray,
    initial_pair_initializer: SparsePairInitializer,
    transition_row_builder: SparsePairTransitionRowBuilder,
    *,
    transition_cache_key_builder: SparsePairTransitionCacheKeyBuilder | None = None,
    transition_row_cache: SparseTransitionRowCache | None = None,
    return_smoothed: bool | None = True,
    evidence_mode: EvidenceComputationMode | str | None = None,
    return_pair_lattice: bool = False,
) -> SparseSecondOrderGridResult:
    """Compute exact evidence for a sparse second-order grid HMM.

    Parameters
    ----------
    log_likelihood:
        Emission log likelihoods with shape ``(n_time, n_states)``.
    initial_pair_initializer:
        Callable receiving the scaled emission likelihoods and returning
        ``(prev, curr, weights, edge_counts)`` for the initial pair lattice.
        The weights are unnormalized masses for ``(x[0], x[1])`` and should
        already include the first two emission rows and any initial prior.
    transition_row_builder:
        Callable ``(prev_idx, curr_idx, transition_index) -> (dst, weights)``
        returning a normalized sparse row over ``x[t+1]``.  ``transition_index``
        is ``time_index - 1`` and therefore ranges from ``1`` to
        ``n_time - 2``.
    transition_cache_key_builder:
        Optional callable used to share transition rows across repeated pair
        states.  Return ``None`` to skip caching for a row.
    transition_row_cache:
        Optional reusable cache instance. When omitted, a fresh cache is used
        for this call.
    return_smoothed:
        Compatibility flag.  If true, compute fixed-interval smoothed
        single-state marginals.  Ignored when ``evidence_mode`` is supplied.
    evidence_mode:
        Optional explicit computation mode; use ``"evidence_only"`` to skip
        backward smoothing while keeping the same log evidence.
    return_pair_lattice:
        If true, retain the normalized pair filtering lattice in the result.
    """

    scaled, offsets = scaled_emissions(log_likelihood)
    n_time, n_states = scaled.shape
    if n_time < 2:
        raise ValueError(
            "sparse second-order evidence requires at least two time steps"
        )
    mode = resolve_evidence_computation_mode(
        evidence_mode, return_smoothed=return_smoothed
    )
    return_smoothed = mode.return_smoothed

    prev, curr, alpha, initial_edge_counts = initial_pair_initializer(scaled)
    prev = np.asarray(prev, dtype=int)
    curr = np.asarray(curr, dtype=int)
    alpha = np.asarray(alpha, dtype=float)
    if prev.shape != curr.shape or prev.shape != alpha.shape:
        raise ValueError(
            "initial pair arrays must have matching one-dimensional shapes"
        )
    if (
        np.any(prev < 0)
        or np.any(prev >= n_states)
        or np.any(curr < 0)
        or np.any(curr >= n_states)
    ):
        raise ValueError("initial pair indices are outside the grid")
    first_scale = float(alpha.sum())
    if first_scale <= 0.0 or not np.isfinite(first_scale):
        raise ValueError("initial pair lattice has no positive finite mass")
    alpha = alpha / first_scale

    pair_prev: list[np.ndarray] = [prev]
    pair_curr: list[np.ndarray] = [curr]
    filtered: list[np.ndarray] = [alpha]
    scales: list[float] = [first_scale]
    edge_counts: list[int] = [
        int(value) for value in np.asarray(initial_edge_counts, dtype=int).ravel()
    ]
    transition_rows: list[SparseTransitionRows] = []
    row_cache = (
        transition_row_cache
        if transition_row_cache is not None
        else SparseTransitionRowCache()
    )
    cache_hits = 0
    cache_misses = 0
    logp = float(np.log(first_scale) + offsets[0] + offsets[1])

    for time_index in range(2, n_time):
        transition_index = time_index - 1
        (
            prev,
            curr,
            alpha,
            counts,
            rows_for_transition,
            hits,
            misses,
        ) = _advance_sparse_pair_alpha(
            prev,
            curr,
            alpha,
            scaled[time_index],
            n_states=n_states,
            transition_index=transition_index,
            transition_row_builder=transition_row_builder,
            cache_key_builder=transition_cache_key_builder,
            row_cache=row_cache,
            store_transition_rows=return_smoothed,
        )
        cache_hits += hits
        cache_misses += misses
        scale = float(alpha.sum())
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(
                f"emission row {time_index} has no positive sparse-pair predicted mass"
            )
        alpha = alpha / scale
        pair_prev.append(prev)
        pair_curr.append(curr)
        filtered.append(alpha)
        scales.append(scale)
        edge_counts.extend(counts)
        if return_smoothed:
            transition_rows.append(rows_for_transition)
        logp += float(np.log(scale) + offsets[time_index])

    terminal = _terminal_position_log_probabilities(
        pair_curr[-1], filtered[-1], n_states=n_states
    )

    if return_smoothed:
        betas = _backward_sparse_pair_betas(
            pair_prev,
            pair_curr,
            filtered,
            scaled,
            scales,
            transition_rows,
            n_states=n_states,
        )
        smoothed = _pair_position_marginals(
            pair_prev,
            pair_curr,
            filtered,
            betas,
            n_time=n_time,
            n_states=n_states,
        )
        terminal = probabilities_to_log_probabilities(
            smoothed[-1], axis=0, normalize=False
        )
        smoothed_log = probabilities_to_log_probabilities(smoothed, axis=1)
        backward_label = "forward_cached"
    else:
        smoothed_log = None
        backward_label = "skipped_evidence_only"

    pair_counts = np.asarray([values.shape[0] for values in filtered], dtype=float)
    diagnostics: dict[str, int | float | str] = {
        "state_support": "sparse_pair_grid",
        "transition_support": "caller_supplied_sparse_rows",
        "initial_pair_count": int(pair_counts[0]),
        "terminal_pair_count": int(pair_counts[-1]),
        "mean_pair_count": float(pair_counts.mean()),
        "max_pair_count": int(pair_counts.max()),
        "mean_outgoing_count": float(np.mean(edge_counts)) if edge_counts else 0.0,
        "max_outgoing_count": int(np.max(edge_counts)) if edge_counts else 0,
        "transition_row_cache_entries": int(row_cache.entries),
        "transition_row_cache_hits": int(cache_hits),
        "transition_row_cache_misses": int(cache_misses),
        "backward_transition_rows": backward_label,
    }
    diagnostics.update(mode.to_diagnostics())
    diagnostics["smoothed_posterior_returned"] = int(smoothed_log is not None)
    diagnostics["terminal_posterior_returned"] = 1

    return SparseSecondOrderGridResult(
        log_marginal_likelihood=float(logp),
        terminal_log_probabilities=terminal,
        smoothed_log_probabilities=smoothed_log,
        filtered_pair_previous=tuple(pair_prev) if return_pair_lattice else None,
        filtered_pair_current=tuple(pair_curr) if return_pair_lattice else None,
        filtered_pair_probabilities=tuple(filtered) if return_pair_lattice else None,
        scales=np.asarray(scales, dtype=float),
        emission_offsets=np.asarray(offsets, dtype=float),
        diagnostics=diagnostics,
    )


def _advance_sparse_pair_alpha(
    prev: np.ndarray,
    curr: np.ndarray,
    alpha: np.ndarray,
    current_emission: np.ndarray,
    *,
    n_states: int,
    transition_index: int,
    transition_row_builder: SparsePairTransitionRowBuilder,
    cache_key_builder: SparsePairTransitionCacheKeyBuilder | None,
    row_cache: SparseTransitionRowCache[Hashable],
    store_transition_rows: bool,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[int], SparseTransitionRows, int, int
]:
    prev_parts: list[np.ndarray] = []
    curr_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    edge_counts: list[int] = []
    transition_rows: SparseTransitionRows = []
    cache_hits = 0
    cache_misses = 0

    for src_prev, src_curr, source_mass in zip(prev, curr, alpha, strict=True):
        if source_mass <= 0.0:
            if store_transition_rows:
                transition_rows.append(
                    (np.empty(0, dtype=int), np.empty(0, dtype=float))
                )
            continue

        cache_key = (
            None
            if cache_key_builder is None
            else cache_key_builder(int(src_prev), int(src_curr), int(transition_index))
        )

        def build_row() -> tuple[np.ndarray, np.ndarray]:
            dst, weights = transition_row_builder(
                int(src_prev), int(src_curr), int(transition_index)
            )
            dst = np.asarray(dst, dtype=int)
            weights = np.asarray(weights, dtype=float)
            if dst.ndim != 1 or weights.shape != dst.shape:
                raise ValueError(
                    "transition rows must return one-dimensional dst and weights arrays with matching shapes"
                )
            if np.any(dst < 0) or np.any(dst >= n_states):
                raise ValueError(
                    "transition row contains destination indices outside the grid"
                )
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError(
                    "transition row weights must be finite and nonnegative"
                )
            total = float(weights.sum())
            if total <= 0.0:
                raise ValueError("transition row must contain positive mass")
            if not np.isclose(total, 1.0):
                weights = weights / total
            return dst, weights

        if cache_key is None:
            dst, weights = build_row()
            cache_misses += 1
        else:
            previous_hits = row_cache.hits
            previous_misses = row_cache.misses
            dst, weights = row_cache.get_or_build(cache_key, build_row)
            cache_hits += int(row_cache.hits - previous_hits)
            cache_misses += int(row_cache.misses - previous_misses)

        if store_transition_rows:
            transition_rows.append((dst, weights))
        values = float(source_mass) * weights * current_emission[dst]
        keep = values > 0.0
        edge_counts.append(int(dst.shape[0]))
        if not np.any(keep):
            continue
        prev_parts.append(np.full(int(np.sum(keep)), int(src_curr), dtype=int))
        curr_parts.append(dst[keep])
        value_parts.append(values[keep])

    next_prev, next_curr, next_alpha = _coalesce_pairs(
        prev_parts, curr_parts, value_parts, n_states
    )
    return (
        next_prev,
        next_curr,
        next_alpha,
        edge_counts,
        transition_rows,
        cache_hits,
        cache_misses,
    )


def _backward_sparse_pair_betas(
    pair_prev: list[np.ndarray],
    pair_curr: list[np.ndarray],
    filtered: list[np.ndarray],
    scaled: np.ndarray,
    scales: list[float],
    transition_rows: list[SparseTransitionRows],
    *,
    n_states: int,
) -> list[np.ndarray]:
    betas: list[np.ndarray] = [np.empty(0, dtype=float) for _ in filtered]
    betas[-1] = np.ones_like(filtered[-1], dtype=float)
    if len(transition_rows) != max(len(filtered) - 1, 0):
        raise ValueError(
            "transition row cache length does not match sparse pair lattice"
        )

    for pair_index in range(len(filtered) - 2, -1, -1):
        next_flat = pair_prev[pair_index + 1].astype(np.int64) * n_states + pair_curr[
            pair_index + 1
        ].astype(np.int64)
        order = np.argsort(next_flat, kind="stable")
        next_flat = next_flat[order]
        next_beta = betas[pair_index + 1][order]
        beta = np.zeros_like(filtered[pair_index], dtype=float)
        observation_index = pair_index + 2
        rows_for_transition = transition_rows[pair_index]
        if len(rows_for_transition) != len(filtered[pair_index]):
            raise ValueError("transition row cache does not match sparse pair count")
        for row, src_curr in enumerate(pair_curr[pair_index]):
            dst, weights = rows_for_transition[row]
            query = int(src_curr) * n_states + dst.astype(np.int64)
            continuation = _lookup_sorted(next_flat, next_beta, query)
            beta[row] = float(
                np.sum(weights * scaled[observation_index, dst] * continuation)
                / scales[pair_index + 1]
            )
        betas[pair_index] = beta
    return betas


def _pair_position_marginals(
    pair_prev: list[np.ndarray],
    pair_curr: list[np.ndarray],
    filtered: list[np.ndarray],
    betas: list[np.ndarray],
    *,
    n_time: int,
    n_states: int,
) -> np.ndarray:
    position = np.zeros((n_time, n_states), dtype=float)
    for pair_index, (prev, curr, alpha, beta) in enumerate(
        zip(pair_prev, pair_curr, filtered, betas, strict=True)
    ):
        posterior = np.asarray(alpha, dtype=float) * np.asarray(beta, dtype=float)
        total = float(posterior.sum())
        if total > 0.0 and np.isfinite(total):
            posterior = posterior / total
        if pair_index == 0:
            np.add.at(position[0], prev, posterior)
        np.add.at(position[pair_index + 1], curr, posterior)

    for time_index in range(n_time):
        total = float(position[time_index].sum())
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(
                f"smoothed posterior at time {time_index} has no positive finite mass"
            )
        position[time_index] /= total
    return position


def _terminal_position_log_probabilities(
    curr: np.ndarray, alpha: np.ndarray, *, n_states: int
) -> np.ndarray:
    position = np.zeros(int(n_states), dtype=float)
    np.add.at(position, curr, alpha)
    total = float(position.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("terminal posterior has no positive finite mass")
    position /= total
    return probabilities_to_log_probabilities(position, axis=0, normalize=False)


def _coalesce_pairs(
    prev_parts: list[np.ndarray],
    curr_parts: list[np.ndarray],
    value_parts: list[np.ndarray],
    n_states: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not value_parts:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    prev = np.concatenate(prev_parts).astype(np.int64, copy=False)
    curr = np.concatenate(curr_parts).astype(np.int64, copy=False)
    values = np.concatenate(value_parts).astype(float, copy=False)
    keep = np.isfinite(values) & (values > 0.0)
    if not np.any(keep):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=float)
    flat = prev[keep] * int(n_states) + curr[keep]
    values = values[keep]
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    values = values[order]
    unique, starts = np.unique(flat, return_index=True)
    summed = np.add.reduceat(values, starts)
    keep_summed = summed > 0.0
    unique = unique[keep_summed]
    summed = summed[keep_summed]
    return (
        (unique // int(n_states)).astype(int),
        (unique % int(n_states)).astype(int),
        summed,
    )


def _lookup_sorted(
    sorted_keys: np.ndarray, values: np.ndarray, query_keys: np.ndarray
) -> np.ndarray:
    query = np.asarray(query_keys, dtype=np.int64)
    positions = np.searchsorted(sorted_keys, query)
    out = np.zeros(query.shape[0], dtype=float)
    in_bounds = positions < sorted_keys.shape[0]
    if not np.any(in_bounds):
        return out
    query_rows = np.flatnonzero(in_bounds)
    matched = sorted_keys[positions[query_rows]] == query[query_rows]
    out[query_rows[matched]] = values[positions[query_rows[matched]]]
    return out


__all__ = [
    "SparsePairInitializer",
    "SparsePairTransitionCacheKeyBuilder",
    "SparsePairTransitionRowBuilder",
    "SparseSecondOrderGridResult",
    "SparseTransitionRows",
    "sparse_second_order_grid_evidence",
]
