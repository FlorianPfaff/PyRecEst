"""Real-valued and numerically stable contracts for record-based Kalman smoothers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from pyrecest.numerics import is_positive_semidefinite, is_symmetric

from . import record_smoother as _record_smoother

# pylint: disable=protected-access

# The record smoother implements a real-valued RTS recursion. NumPy's explicit
# float conversion otherwise accepts native complex arrays by discarding their
# imaginary parts, which silently changes the supplied state-space model.

_original_record_arrays = _record_smoother._record_arrays


def _contains_complex_values(value: Any) -> bool:
    """Return whether an array-like value contains complex scalars."""

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return False
    if np.iscomplexobj(value_array):
        return True
    if value_array.dtype != object:
        return False
    return any(
        isinstance(item, (complex, np.complexfloating)) for item in value_array.flat
    )


def _reject_complex_values(value: Any, message: str) -> None:
    if _contains_complex_values(value):
        raise ValueError(message)


def _validate_covariance_structure(value: Any) -> None:
    """Reject finite square record covariances that are not symmetric PSD.

    Shape, type, and finiteness failures remain delegated to the record smoother's
    existing validation so this contract only tightens the covariance semantics.
    """

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return
    if (
        np.iscomplexobj(value_array)
        or value_array.ndim != 2
        or value_array.shape[0] != value_array.shape[1]
    ):
        return

    try:
        covariance = np.asarray(value_array, dtype=float)
    except (TypeError, ValueError, OverflowError, RuntimeError):
        return
    if not np.isfinite(covariance).all():
        return

    with np.errstate(over="ignore", invalid="ignore"):
        symmetric = is_symmetric(covariance)
    if not symmetric:
        raise ValueError("record covariances must be symmetric")
    if not is_positive_semidefinite(covariance):
        raise ValueError("record covariances must be positive semidefinite")


def _record_arrays(
    records: Sequence[Mapping[str, Any]],
    *,
    time_key: str,
    state_key: str,
    covariance_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for record in records:
        _reject_complex_values(
            record[state_key],
            "record states must contain real values",
        )
        _reject_complex_values(
            record[covariance_key],
            "record covariances must contain real values",
        )
        _validate_covariance_structure(record[covariance_key])
    return _original_record_arrays(
        records,
        time_key=time_key,
        state_key=state_key,
        covariance_key=covariance_key,
    )


def _call_model(
    model: Callable[..., np.ndarray], dt: float, state_dim: int, name: str
) -> np.ndarray:
    arity = _record_smoother._preferred_model_call_arity(model)
    if arity == 2:
        matrix = model(dt, state_dim)
    elif arity == 1:
        matrix = model(dt)
    else:
        matrix = _record_smoother._call_model_with_fallback(model, dt, state_dim, name)

    _reject_complex_values(matrix, f"{name} must return real values")
    array = np.asarray(matrix, dtype=float)
    if array.shape != (state_dim, state_dim):
        raise ValueError(f"{name} must return a ({state_dim}, {state_dim}) matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must return finite values")
    return array


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric average without overflowing finite operands."""

    return 0.5 * matrix + 0.5 * matrix.T


def install_record_smoother_numeric_contract() -> None:
    """Install numeric validation and stable covariance symmetrization."""

    setattr(_record_arrays, "_pyrecest_real_numeric_contract", True)
    setattr(_call_model, "_pyrecest_real_numeric_contract", True)
    setattr(_symmetrized, "_pyrecest_stable_symmetrization", True)

    if not getattr(
        _record_smoother._record_arrays,
        "_pyrecest_real_numeric_contract",
        False,
    ):
        _record_smoother._record_arrays = _record_arrays
    if not getattr(
        _record_smoother._call_model,
        "_pyrecest_real_numeric_contract",
        False,
    ):
        _record_smoother._call_model = _call_model
    if not getattr(
        _record_smoother._symmetrized,
        "_pyrecest_stable_symmetrization",
        False,
    ):
        _record_smoother._symmetrized = _symmetrized
