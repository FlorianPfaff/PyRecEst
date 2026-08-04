"""Real-valued input contract for record-based Kalman smoothers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from . import record_smoother as _record_smoother

# pylint: disable=protected-access

# The record smoother implements a real-valued RTS recursion. NumPy's explicit
# float conversion otherwise accepts native complex arrays by discarding their
# imaginary parts, and conversion of masked arrays drops their masks. Both cases
# can silently change the supplied state-space model.

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


def _contains_masked_values(
    value: Any,
    active_ids: set[int] | None = None,
) -> bool:
    """Return whether an array-like value contains genuinely masked entries."""

    if value is np.ma.masked or np.ma.is_masked(value):
        return True
    if isinstance(value, np.ma.MaskedArray):
        value = value.data

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return False
        items = value.flat
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        return False

    if active_ids is None:
        active_ids = set()
    value_id = id(value)
    if value_id in active_ids:
        return False
    active_ids.add(value_id)
    try:
        return any(_contains_masked_values(item, active_ids) for item in items)
    finally:
        active_ids.remove(value_id)


def _reject_complex_values(value: Any, message: str) -> None:
    if _contains_complex_values(value):
        raise ValueError(message)


def _reject_masked_values(value: Any, message: str) -> None:
    if _contains_masked_values(value):
        raise ValueError(message)


def _record_arrays(
    records: Sequence[Mapping[str, Any]],
    *,
    time_key: str,
    state_key: str,
    covariance_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for record in records:
        _reject_masked_values(
            record[time_key],
            "record times must not contain masked values",
        )
        _reject_complex_values(
            record[time_key],
            "record times must contain real values",
        )
        _reject_masked_values(
            record[state_key],
            "record states must not contain masked values",
        )
        _reject_complex_values(
            record[state_key],
            "record states must contain real values",
        )
        _reject_masked_values(
            record[covariance_key],
            "record covariances must not contain masked values",
        )
        _reject_complex_values(
            record[covariance_key],
            "record covariances must contain real values",
        )
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

    _reject_masked_values(matrix, f"{name} must not return masked values")
    _reject_complex_values(matrix, f"{name} must return real values")
    array = np.asarray(matrix, dtype=float)
    if array.shape != (state_dim, state_dim):
        raise ValueError(f"{name} must return a ({state_dim}, {state_dim}) matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must return finite values")
    return array


def install_record_smoother_numeric_contract() -> None:
    """Install real-value validation on the record smoother implementation."""

    setattr(_record_arrays, "_pyrecest_real_numeric_contract", True)
    setattr(_call_model, "_pyrecest_real_numeric_contract", True)

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
