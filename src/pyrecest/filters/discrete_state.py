"""Finite-state Bayesian filtering and smoothing utilities.

The transition matrices in this module use the column-stochastic convention
``transition[dst, src] = P(x_t = dst | x_{t-1} = src)``. With this convention a
prediction step is the sparse matrix-vector product ``transition @ weights``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.special import logsumexp

LOG_ZERO = -1.0e300

__all__ = [
    "DiscreteForwardBackwardResult",
    "DiscreteIMMForwardBackwardResult",
    "LOG_ZERO",
    "discrete_forward_backward",
    "discrete_forward_backward_time_varying",
    "imm_forward_backward",
    "mode_transition_matrix",
    "probabilities_to_log_probabilities",
    "scaled_emissions",
    "sparse_gaussian_transition_matrix",
    "sticky_mode_transition_matrix",
    "uniform_log_probabilities",
    "uniform_probabilities",
]


@dataclass(frozen=True)
class DiscreteForwardBackwardResult:
    """Result returned by :func:`discrete_forward_backward`.

    Attributes
    ----------
    log_marginal_likelihood:
        Log normalizing constant for the whole emission sequence.
    filtered_probabilities:
        Normalized filtering probabilities with shape ``(n_time, n_states)``.
    smoothed_probabilities:
        Normalized fixed-interval smoothing probabilities with shape
        ``(n_time, n_states)``.
    scales:
        Per-time-step forward scaling constants after the emission offset has
        been removed.
    emission_offsets:
        Per-time-step offsets subtracted from the emission log likelihoods
        before exponentiation.
    """

    log_marginal_likelihood: float
    filtered_probabilities: np.ndarray
    smoothed_probabilities: np.ndarray
    scales: np.ndarray
    emission_offsets: np.ndarray

    @property
    def filtered_log_probabilities(self) -> np.ndarray:
        """Filtering probabilities represented as normalized log probabilities."""

        return probabilities_to_log_probabilities(self.filtered_probabilities, axis=1)

    @property
    def smoothed_log_probabilities(self) -> np.ndarray:
        """Smoothing probabilities represented as normalized log probabilities."""

        return probabilities_to_log_probabilities(self.smoothed_probabilities, axis=1)


@dataclass(frozen=True)
class DiscreteIMMForwardBackwardResult:
    """Result returned by :func:`imm_forward_backward`.

    The joint probability arrays use axes ``(time, mode, state)``. Convenience
    properties expose the state and mode marginals obtained by summing the joint
    probabilities over the complementary axis.
    """

    log_marginal_likelihood: float
    filtered_joint_probabilities: np.ndarray
    smoothed_joint_probabilities: np.ndarray
    scales: np.ndarray
    emission_offsets: np.ndarray

    @property
    def filtered_state_probabilities(self) -> np.ndarray:
        """Filtering probabilities marginalized over modes."""

        return self.filtered_joint_probabilities.sum(axis=1)

    @property
    def smoothed_state_probabilities(self) -> np.ndarray:
        """Smoothing probabilities marginalized over modes."""

        return self.smoothed_joint_probabilities.sum(axis=1)

    @property
    def filtered_mode_probabilities(self) -> np.ndarray:
        """Filtering probabilities marginalized over states."""

        return self.filtered_joint_probabilities.sum(axis=2)

    @property
    def smoothed_mode_probabilities(self) -> np.ndarray:
        """Smoothing probabilities marginalized over states."""

        return self.smoothed_joint_probabilities.sum(axis=2)

    @property
    def smoothed_state_log_probabilities(self) -> np.ndarray:
        """State smoothing marginals represented as normalized log probabilities."""

        return probabilities_to_log_probabilities(
            self.smoothed_state_probabilities, axis=1
        )


def scaled_emissions(log_likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exponentiate log emissions after subtracting row-wise offsets.

    Parameters
    ----------
    log_likelihood:
        Emission log likelihoods with shape ``(n_time, n_states)``.

    Returns
    -------
    scaled, offsets:
        ``scaled[t, i] = exp(log_likelihood[t, i] - offsets[t])`` for finite
        emissions. Non-finite entries are returned as zero in ``scaled``.
    """

    values = np.asarray(log_likelihood, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_likelihood must have shape (n_time, n_states)")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(
            "log_likelihood must contain at least one time step and one state"
        )
    finite = np.isfinite(values)
    if not np.all(np.any(finite, axis=1)):
        raise ValueError("every emission row must contain at least one finite value")
    offsets = np.max(np.where(finite, values, -np.inf), axis=1)
    shifted = np.where(finite, values - offsets[:, None], -np.inf)
    scaled = np.exp(np.clip(shifted, -745.0, 0.0))
    scaled[~finite] = 0.0
    return scaled, offsets


def probabilities_to_log_probabilities(
    probabilities: np.ndarray,
    axis: int | tuple[int, ...] | None = -1,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Convert probabilities to log probabilities using ``LOG_ZERO`` for zeros."""

    probs = np.asarray(probabilities, dtype=float)
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")

    positive = probs > 0.0
    log_values = np.full(probs.shape, -np.inf, dtype=float)
    log_values[positive] = np.log(probs[positive])
    if normalize:
        normalizer = logsumexp(log_values, axis=axis, keepdims=True)
        if np.any(~np.isfinite(normalizer)):
            raise ValueError(
                "probabilities must contain positive mass along the normalization axis"
            )
        log_values = log_values - normalizer

    out = np.full(probs.shape, LOG_ZERO, dtype=float)
    finite = np.isfinite(log_values)
    out[finite] = log_values[finite]
    return out


def uniform_probabilities(
    n_states: int, valid_state_mask: np.ndarray | None = None
) -> np.ndarray:
    """Return a uniform probability vector over all valid states."""

    mask = _coerce_valid_state_mask(valid_state_mask, n_states)
    probs = np.zeros(n_states, dtype=float)
    if mask is None:
        probs.fill(1.0 / n_states)
    else:
        probs[mask] = 1.0 / int(np.sum(mask))
    return probs


def uniform_log_probabilities(
    n_states: int, valid_state_mask: np.ndarray | None = None
) -> np.ndarray:
    """Return a uniform log-probability vector over all valid states."""

    return probabilities_to_log_probabilities(
        uniform_probabilities(n_states, valid_state_mask),
        axis=0,
        normalize=False,
    )


def sparse_gaussian_transition_matrix(
    state_vectors: np.ndarray,
    sigma: float,
    max_step_sigma: float = 4.0,
    *,
    valid_state_mask: np.ndarray | None = None,
) -> csr_matrix:
    """Build a sparse column-stochastic Gaussian transition matrix on a grid.

    Parameters
    ----------
    state_vectors:
        Grid coordinates with shape ``(n_states,)`` for a one-dimensional grid or
        ``(n_states, state_dim)`` for a multidimensional grid.
    sigma:
        Isotropic Gaussian standard deviation in the same units as the grid.
    max_step_sigma:
        Radius, measured in multiples of ``sigma``, beyond which entries are
        omitted from the sparse matrix. Use ``np.inf`` for a dense Gaussian
        support. If truncation removes every valid destination for a source, the
        nearest valid destination is retained.
    valid_state_mask:
        Optional boolean mask that excludes invalid destination states. Invalid
        destination rows receive zero probability.
    """

    states = np.asarray(state_vectors, dtype=float)
    if states.ndim == 1:
        states = states[:, None]
    elif states.ndim != 2:
        raise ValueError(
            "state_vectors must have shape (n_states,) or (n_states, state_dim)"
        )
    n_states = states.shape[0]
    if n_states == 0:
        raise ValueError("state_vectors must contain at least one state")

    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    max_step_sigma = float(max_step_sigma)
    if max_step_sigma <= 0.0 or np.isnan(max_step_sigma):
        raise ValueError("max_step_sigma must be positive or np.inf")

    valid_mask = _coerce_valid_state_mask(valid_state_mask, n_states)
    allowed = (
        np.arange(n_states, dtype=int)
        if valid_mask is None
        else np.flatnonzero(valid_mask)
    )
    radius2 = np.inf if np.isinf(max_step_sigma) else (sigma * max_step_sigma) ** 2

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for src, center in enumerate(states):
        delta = states - center[None, :]
        dist2 = np.sum(delta * delta, axis=1)
        keep = dist2 <= radius2
        if valid_mask is not None:
            keep &= valid_mask
        if not np.any(keep):
            keep[int(allowed[int(np.argmin(dist2[allowed]))])] = True
        dst = np.flatnonzero(keep)
        weights = np.exp(-0.5 * dist2[dst] / (sigma * sigma))
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.zeros_like(weights)
            weights[int(np.argmin(dist2[dst]))] = 1.0
        else:
            weights /= weight_sum
        rows.extend(int(index) for index in dst)
        cols.extend([src] * len(dst))
        data.extend(float(value) for value in weights)

    return csr_matrix((data, (rows, cols)), shape=(n_states, n_states))


def sticky_mode_transition_matrix(n_modes: int, stickiness: float) -> np.ndarray:
    """Return an IMM-style row-stochastic mode transition matrix.

    ``stickiness`` is the probability of remaining in the same mode. The
    remaining probability mass is spread uniformly over all other modes.
    """

    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if not np.isfinite(stickiness) or not 0.0 <= stickiness <= 1.0:
        raise ValueError("stickiness must be finite and in [0, 1]")
    if n_modes == 1:
        return np.ones((1, 1), dtype=float)
    off_diag = (1.0 - float(stickiness)) / (n_modes - 1)
    matrix = np.full((n_modes, n_modes), off_diag, dtype=float)
    np.fill_diagonal(matrix, float(stickiness))
    return matrix


def discrete_forward_backward(
    log_likelihood: np.ndarray,
    transition: spmatrix | np.ndarray,
    *,
    initial_probabilities: np.ndarray | None = None,
    valid_state_mask: np.ndarray | None = None,
) -> DiscreteForwardBackwardResult:
    """Run scaled finite-state HMM filtering and fixed-interval smoothing."""

    scaled, offsets, mask = _prepare_emissions(log_likelihood, valid_state_mask)
    n_time, n_states = scaled.shape
    transition_matrix = _validate_transition_matrix(transition, n_states, "transition")
    return _discrete_forward_backward_from_scaled(
        scaled,
        offsets,
        [transition_matrix] * max(n_time - 1, 0),
        initial_probabilities=initial_probabilities,
        valid_state_mask=mask,
    )


def discrete_forward_backward_time_varying(
    log_likelihood: np.ndarray,
    transitions: Sequence[spmatrix | np.ndarray],
    *,
    initial_probabilities: np.ndarray | None = None,
    valid_state_mask: np.ndarray | None = None,
) -> DiscreteForwardBackwardResult:
    """Run scaled HMM filtering/smoothing with one transition per time step."""

    scaled, offsets, mask = _prepare_emissions(log_likelihood, valid_state_mask)
    n_time, n_states = scaled.shape
    if len(transitions) != max(n_time - 1, 0):
        raise ValueError(
            "transitions must contain one matrix per adjacent time-step pair"
        )
    transition_matrices = [
        _validate_transition_matrix(transition, n_states, f"transitions[{index}]")
        for index, transition in enumerate(transitions)
    ]
    return _discrete_forward_backward_from_scaled(
        scaled,
        offsets,
        transition_matrices,
        initial_probabilities=initial_probabilities,
        valid_state_mask=mask,
    )


def imm_forward_backward(
    log_likelihood: np.ndarray,
    state_transitions: Sequence[spmatrix | np.ndarray | None],
    mode_transition: np.ndarray,
    *,
    initial_state_probabilities: np.ndarray | None = None,
    initial_mode_probabilities: np.ndarray | None = None,
    valid_state_mask: np.ndarray | None = None,
) -> DiscreteIMMForwardBackwardResult:
    """Run a scaled interacting-multiple-model forward/backward recursion.

    Parameters
    ----------
    log_likelihood:
        Shared state-emission log likelihoods with shape ``(n_time, n_states)``.
    state_transitions:
        One state transition matrix per destination mode. ``None`` denotes a
        reset transition that redistributes the incoming mode mass uniformly over
        valid states.
    mode_transition:
        Row-stochastic matrix with entries ``P(mode_t = dst | mode_{t-1} = src)``.
    """

    scaled, offsets, mask = _prepare_emissions(log_likelihood, valid_state_mask)
    n_time, n_states = scaled.shape
    n_modes = len(state_transitions)
    if n_modes == 0:
        raise ValueError("state_transitions must contain at least one mode")
    transitions = [
        (
            None
            if transition is None
            else _validate_transition_matrix(
                transition, n_states, f"state_transitions[{index}]"
            )
        )
        for index, transition in enumerate(state_transitions)
    ]
    mode_matrix = _validate_mode_transition_matrix(mode_transition, n_modes)
    state_prior = _normalize_probability_vector(
        initial_state_probabilities,
        n_states,
        "initial_state_probabilities",
        valid_state_mask=mask,
    )
    mode_prior = _normalize_probability_vector(
        initial_mode_probabilities, n_modes, "initial_mode_probabilities"
    )

    filtered = np.zeros((n_time, n_modes, n_states), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = mode_prior[:, None] * state_prior[None, :] * scaled[0][None, :]
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError(
            "first emission row has no finite likelihood mass on the prior support"
        )
    alpha /= scales[0]
    filtered[0] = alpha
    log_marginal = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        predicted = np.zeros_like(alpha)
        for dst_idx, transition in enumerate(transitions):
            dst = np.zeros(n_states, dtype=float)
            for src_idx in range(n_modes):
                dst += mode_matrix[src_idx, dst_idx] * _apply_transition(
                    transition,
                    alpha[src_idx],
                    valid_state_mask=mask,
                )
            predicted[dst_idx] = dst
        alpha = predicted * scaled[time_index][None, :]
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        log_marginal += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones((n_modes, n_states), dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta_prev = np.zeros_like(beta)
        for src_idx in range(n_modes):
            for dst_idx, transition in enumerate(transitions):
                beta_prev[src_idx] += mode_matrix[
                    src_idx, dst_idx
                ] * _apply_transition_backward(
                    transition,
                    scaled[time_index] * beta[dst_idx],
                    valid_state_mask=mask,
                )
        beta = beta_prev / scales[time_index]
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = (
            gamma / total if total > 0.0 else filtered[time_index - 1]
        )

    return DiscreteIMMForwardBackwardResult(
        log_marginal_likelihood=log_marginal,
        filtered_joint_probabilities=filtered,
        smoothed_joint_probabilities=smoothed,
        scales=scales,
        emission_offsets=offsets,
    )


# Compatibility-oriented alias for the name used in the source replay utilities.
mode_transition_matrix = sticky_mode_transition_matrix


def _prepare_emissions(
    log_likelihood: np.ndarray, valid_state_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    scaled, offsets = scaled_emissions(log_likelihood)
    mask = _coerce_valid_state_mask(valid_state_mask, scaled.shape[1])
    if mask is not None:
        scaled = scaled.copy()
        scaled[:, ~mask] = 0.0
    return scaled, offsets, mask


def _discrete_forward_backward_from_scaled(
    scaled: np.ndarray,
    offsets: np.ndarray,
    transitions: Sequence[csr_matrix],
    *,
    initial_probabilities: np.ndarray | None,
    valid_state_mask: np.ndarray | None,
) -> DiscreteForwardBackwardResult:
    n_time, n_states = scaled.shape
    initial = _normalize_probability_vector(
        initial_probabilities,
        n_states,
        "initial_probabilities",
        valid_state_mask=valid_state_mask,
    )
    filtered = np.zeros((n_time, n_states), dtype=float)
    scales = np.zeros(n_time, dtype=float)

    alpha = scaled[0] * initial
    scales[0] = float(alpha.sum())
    if scales[0] <= 0.0:
        raise ValueError(
            "first emission row has no finite likelihood mass on the prior support"
        )
    alpha /= scales[0]
    filtered[0] = alpha
    log_marginal = float(np.log(scales[0]) + offsets[0])

    for time_index in range(1, n_time):
        alpha = (
            np.asarray(transitions[time_index - 1] @ alpha, dtype=float).reshape(
                n_states
            )
            * scaled[time_index]
        )
        scales[time_index] = float(alpha.sum())
        if scales[time_index] <= 0.0:
            raise ValueError(f"emission row {time_index} has no finite predicted mass")
        alpha /= scales[time_index]
        filtered[time_index] = alpha
        log_marginal += float(np.log(scales[time_index]) + offsets[time_index])

    smoothed = np.zeros_like(filtered)
    beta = np.ones(n_states, dtype=float)
    smoothed[-1] = filtered[-1]
    for time_index in range(n_time - 1, 0, -1):
        beta = (
            np.asarray(
                transitions[time_index - 1].T @ (scaled[time_index] * beta), dtype=float
            ).reshape(n_states)
            / scales[time_index]
        )
        gamma = filtered[time_index - 1] * beta
        total = float(gamma.sum())
        smoothed[time_index - 1] = (
            gamma / total if total > 0.0 else filtered[time_index - 1]
        )

    return DiscreteForwardBackwardResult(
        log_marginal_likelihood=log_marginal,
        filtered_probabilities=filtered,
        smoothed_probabilities=smoothed,
        scales=scales,
        emission_offsets=offsets,
    )


def _apply_transition(
    transition: csr_matrix | None,
    weights: np.ndarray,
    *,
    valid_state_mask: np.ndarray | None = None,
) -> np.ndarray:
    if transition is None:
        return uniform_probabilities(weights.shape[0], valid_state_mask) * float(
            weights.sum()
        )
    return np.asarray(transition @ weights, dtype=float).reshape(weights.shape[0])


def _apply_transition_backward(
    transition: csr_matrix | None,
    values: np.ndarray,
    *,
    valid_state_mask: np.ndarray | None = None,
) -> np.ndarray:
    if transition is None:
        mask = _coerce_valid_state_mask(valid_state_mask, values.shape[0])
        if mask is None:
            return np.full(
                values.shape, float(values.sum()) / values.shape[0], dtype=float
            )
        return np.full(
            values.shape, float(values[mask].sum()) / int(np.sum(mask)), dtype=float
        )
    return np.asarray(transition.T @ values, dtype=float).reshape(values.shape[0])


def _coerce_valid_state_mask(
    valid_state_mask: np.ndarray | None, n_states: int
) -> np.ndarray | None:
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    if valid_state_mask is None:
        return None
    mask = np.asarray(valid_state_mask, dtype=bool)
    if mask.shape != (n_states,):
        raise ValueError("valid_state_mask must contain one boolean value per state")
    if not np.any(mask):
        raise ValueError("valid_state_mask must contain at least one valid state")
    return mask


def _normalize_probability_vector(
    probabilities: np.ndarray | None,
    n_entries: int,
    name: str,
    *,
    valid_state_mask: np.ndarray | None = None,
) -> np.ndarray:
    if probabilities is None:
        return uniform_probabilities(n_entries, valid_state_mask)
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (n_entries,):
        raise ValueError(f"{name} must have shape ({n_entries},)")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    values = values.copy()
    mask = _coerce_valid_state_mask(valid_state_mask, n_entries)
    if mask is not None:
        values[~mask] = 0.0
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive probability mass")
    return values / total


def _validate_transition_matrix(
    transition: spmatrix | np.ndarray, n_states: int, name: str
) -> csr_matrix:
    if issparse(transition):
        matrix = transition.tocsr().astype(float)
    else:
        values = np.asarray(transition, dtype=float)
        if values.shape != (n_states, n_states):
            raise ValueError(f"{name} must have shape ({n_states}, {n_states})")
        matrix = csr_matrix(values)
    if matrix.shape != (n_states, n_states):
        raise ValueError(f"{name} must have shape ({n_states}, {n_states})")
    if np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0.0):
        raise ValueError(
            f"{name} must contain finite non-negative transition probabilities"
        )
    column_sums = np.asarray(matrix.sum(axis=0), dtype=float).reshape(n_states)
    if not np.allclose(column_sums, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be column-stochastic")
    return matrix


def _validate_mode_transition_matrix(
    mode_transition: np.ndarray, n_modes: int
) -> np.ndarray:
    matrix = np.asarray(mode_transition, dtype=float)
    if matrix.shape != (n_modes, n_modes):
        raise ValueError(f"mode_transition must have shape ({n_modes}, {n_modes})")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(
            "mode_transition must contain finite non-negative probabilities"
        )
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("mode_transition must be row-stochastic")
    return matrix
