"""Compatibility wrapper for finite-state filtering utilities."""

from __future__ import annotations

import runpy
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import issparse

_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_BOOLEAN_TYPES = (bool, np.bool_)
_COMPLEX_TYPES = (complex, np.complexfloating)
_REJECTED_STATE_KINDS = frozenset({"b", "c", "S", "U", "M", "m"})

_module_globals = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "discrete_state.py"),
    run_name=__name__,
)
_original_scaled_emissions = _module_globals["scaled_emissions"]
_original_probabilities_to_log_probabilities = _module_globals[
    "probabilities_to_log_probabilities"
]
_original_discrete_forward_backward = _module_globals[
    "discrete_forward_backward"
]
_original_discrete_forward_backward_time_varying = _module_globals[
    "discrete_forward_backward_time_varying"
]
_original_imm_forward_backward = _module_globals["imm_forward_backward"]
_original_sparse_gaussian_transition_matrix = _module_globals[
    "sparse_gaussian_transition_matrix"
]


def _reject_complex_values(values: Any, message: str) -> None:
    if issparse(values):
        raw_values = values.data
    else:
        try:
            raw_values = np.asarray(values)
        except (TypeError, ValueError):
            return

    contains_complex = raw_values.dtype.kind == "c"
    if raw_values.dtype == object:
        contains_complex = any(
            isinstance(value, _COMPLEX_TYPES) for value in raw_values.ravel()
        )
    if contains_complex:
        raise ValueError(message)


@wraps(_original_scaled_emissions)
def scaled_emissions(log_likelihood):
    _reject_complex_values(log_likelihood, "log_likelihood must contain real values")
    return _original_scaled_emissions(log_likelihood)


@wraps(_original_probabilities_to_log_probabilities)
def probabilities_to_log_probabilities(
    probabilities,
    axis=-1,
    *,
    normalize=True,
):
    _reject_complex_values(
        probabilities,
        "probabilities must contain real probability values",
    )
    return _original_probabilities_to_log_probabilities(
        probabilities,
        axis=axis,
        normalize=normalize,
    )


@wraps(_original_discrete_forward_backward)
def discrete_forward_backward(
    log_likelihood,
    transition,
    *,
    initial_probabilities=None,
    valid_state_mask=None,
):
    _reject_complex_values(log_likelihood, "log_likelihood must contain real values")
    _reject_complex_values(
        transition,
        "transition must contain real transition probabilities",
    )
    return _original_discrete_forward_backward(
        log_likelihood,
        transition,
        initial_probabilities=initial_probabilities,
        valid_state_mask=valid_state_mask,
    )


@wraps(_original_discrete_forward_backward_time_varying)
def discrete_forward_backward_time_varying(
    log_likelihood,
    transitions,
    *,
    initial_probabilities=None,
    valid_state_mask=None,
):
    _reject_complex_values(log_likelihood, "log_likelihood must contain real values")
    for index, transition in enumerate(transitions):
        _reject_complex_values(
            transition,
            f"transitions[{index}] must contain real transition probabilities",
        )
    return _original_discrete_forward_backward_time_varying(
        log_likelihood,
        transitions,
        initial_probabilities=initial_probabilities,
        valid_state_mask=valid_state_mask,
    )


@wraps(_original_imm_forward_backward)
def imm_forward_backward(
    log_likelihood,
    state_transitions,
    mode_transition,
    *,
    initial_state_probabilities=None,
    initial_mode_probabilities=None,
    valid_state_mask=None,
):
    _reject_complex_values(log_likelihood, "log_likelihood must contain real values")
    for index, transition in enumerate(state_transitions):
        if transition is not None:
            _reject_complex_values(
                transition,
                f"state_transitions[{index}] must contain real transition probabilities",
            )
    _reject_complex_values(
        mode_transition,
        "mode_transition must contain real transition probabilities",
    )
    return _original_imm_forward_backward(
        log_likelihood,
        state_transitions,
        mode_transition,
        initial_state_probabilities=initial_state_probabilities,
        initial_mode_probabilities=initial_mode_probabilities,
        valid_state_mask=valid_state_mask,
    )


def _validated_state_vectors(state_vectors: Any) -> np.ndarray:
    try:
        raw_states = np.asarray(state_vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("state_vectors must contain real numeric values") from exc

    if raw_states.dtype.kind in _REJECTED_STATE_KINDS:
        raise ValueError("state_vectors must contain real numeric values")
    if raw_states.dtype == object:
        for value in raw_states.ravel():
            if isinstance(
                value,
                _TEXT_TYPES + _BOOLEAN_TYPES + _COMPLEX_TYPES,
            ):
                raise ValueError("state_vectors must contain real numeric values")

    try:
        states = np.asarray(state_vectors, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("state_vectors must contain real numeric values") from exc

    if states.ndim == 1:
        finite_check_states = states[:, None]
    elif states.ndim == 2:
        finite_check_states = states
    else:
        finite_check_states = None
    if finite_check_states is not None and np.any(~np.isfinite(finite_check_states)):
        raise ValueError("state_vectors must contain only finite values")
    return states


def _validated_positive_scalar(
    value: Any,
    name: str,
    *,
    allow_infinite: bool = False,
) -> float:
    try:
        raw_value = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive real scalar") from exc
    if raw_value.shape != () or raw_value.dtype.kind in _REJECTED_STATE_KINDS:
        raise ValueError(f"{name} must be a positive real scalar")
    scalar = raw_value.item()
    if isinstance(scalar, _TEXT_TYPES + _BOOLEAN_TYPES + _COMPLEX_TYPES):
        raise ValueError(f"{name} must be a positive real scalar")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive real scalar") from exc
    if parsed <= 0.0 or np.isnan(parsed):
        raise ValueError(f"{name} must be a positive real scalar")
    if not allow_infinite and not np.isfinite(parsed):
        raise ValueError(f"{name} must be a positive real scalar")
    return parsed


def _validated_probability_vector(
    probabilities: Any,
    n_entries: int,
    name: str,
    *,
    valid_state_mask=None,
) -> np.ndarray:
    if probabilities is None:
        return _module_globals["uniform_probabilities"](n_entries, valid_state_mask)
    try:
        raw_values = np.asarray(probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real probability values") from exc
    if raw_values.shape != (n_entries,):
        raise ValueError(f"{name} must have shape ({n_entries},)")
    if raw_values.dtype.kind in _REJECTED_STATE_KINDS:
        raise ValueError(f"{name} must contain real probability values")
    if raw_values.dtype == object:
        for value in raw_values.ravel():
            if isinstance(
                value,
                _TEXT_TYPES + _BOOLEAN_TYPES + _COMPLEX_TYPES,
            ):
                raise ValueError(f"{name} must contain real probability values")
    try:
        values = raw_values.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real probability values") from exc
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    values = values.copy()
    mask = _module_globals["_coerce_valid_state_mask"](valid_state_mask, n_entries)
    if mask is not None:
        values[~mask] = 0.0
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive probability mass")
    return values / total


def sparse_gaussian_transition_matrix(
    state_vectors,
    sigma,
    max_step_sigma=4.0,
    *,
    valid_state_mask=None,
):
    states = _validated_state_vectors(state_vectors)
    sigma = _validated_positive_scalar(sigma, "sigma")
    max_step_sigma = _validated_positive_scalar(
        max_step_sigma,
        "max_step_sigma",
        allow_infinite=True,
    )
    return _original_sparse_gaussian_transition_matrix(
        states,
        sigma,
        max_step_sigma=max_step_sigma,
        valid_state_mask=valid_state_mask,
    )


_module_globals["scaled_emissions"] = scaled_emissions
_module_globals["probabilities_to_log_probabilities"] = (
    probabilities_to_log_probabilities
)
_module_globals["discrete_forward_backward"] = discrete_forward_backward
_module_globals["discrete_forward_backward_time_varying"] = (
    discrete_forward_backward_time_varying
)
_module_globals["imm_forward_backward"] = imm_forward_backward
_module_globals["sparse_gaussian_transition_matrix"] = sparse_gaussian_transition_matrix
_module_globals["_normalize_probability_vector"] = _validated_probability_vector
for name in _module_globals["__all__"]:
    globals()[name] = _module_globals[name]
__all__ = _module_globals["__all__"]
