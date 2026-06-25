"""Compatibility wrapper for finite-state filtering utilities."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np

_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)
_BOOLEAN_TYPES = (bool, np.bool_)
_COMPLEX_TYPES = (complex, np.complexfloating)
_REJECTED_STATE_KINDS = frozenset({"b", "c", "S", "U", "M", "m"})

_module_globals = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "discrete_state.py"),
    run_name=__name__,
)
_original_sparse_gaussian_transition_matrix = _module_globals[
    "sparse_gaussian_transition_matrix"
]


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


def sparse_gaussian_transition_matrix(
    state_vectors,
    sigma,
    max_step_sigma=4.0,
    *,
    valid_state_mask=None,
):
    states = _validated_state_vectors(state_vectors)
    return _original_sparse_gaussian_transition_matrix(
        states,
        sigma,
        max_step_sigma=max_step_sigma,
        valid_state_mask=valid_state_mask,
    )


_module_globals["sparse_gaussian_transition_matrix"] = sparse_gaussian_transition_matrix
for name in _module_globals["__all__"]:
    globals()[name] = _module_globals[name]
__all__ = _module_globals["__all__"]
