"""Strict lag validation for fixed-lag MEM-QKF smoothers."""

from __future__ import annotations

from functools import wraps
from operator import index as _operator_index
from typing import Any

import numpy as np

from . import fixed_lag_mem_qkf_smoother as _implementation


def _normalize_lag(value: Any) -> int:
    """Return an exact non-negative integer lag."""

    message = "lag must be a non-negative integer."
    if np.ma.is_masked(value) or isinstance(
        value,
        (bool, np.bool_, np.datetime64, np.timedelta64),
    ):
        raise ValueError(message)
    if isinstance(value, np.ma.MaskedArray):
        value = np.ma.getdata(value)
    try:
        parsed = _operator_index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if parsed < 0:
        raise ValueError(message)
    return int(parsed)


def install_fixed_lag_mem_qkf_option_contract() -> None:
    """Install exact lag validation on public fixed-lag MEM-QKF paths."""

    smoother_type = _implementation.FixedLagMEMQKFSmoother

    current_init = smoother_type.__init__
    if not getattr(current_init, "_pyrecest_strict_lag_contract", False):

        @wraps(current_init)
        def validated_init(
            self,
            lag: int = 1,
            shape_smoothing: str = "rts",
        ):
            return current_init(
                self,
                lag=_normalize_lag(lag),
                shape_smoothing=shape_smoothing,
            )

        validated_init._pyrecest_strict_lag_contract = True
        smoother_type.__init__ = validated_init

    current_smooth = smoother_type.smooth
    if not getattr(current_smooth, "_pyrecest_strict_lag_contract", False):

        @wraps(current_smooth)
        def validated_smooth(
            self,
            filtered_states,
            predicted_states=None,
            system_matrices=None,
            shape_system_matrices=None,
            lag: int | None = None,
        ):
            if lag is not None:
                lag = _normalize_lag(lag)
            return current_smooth(
                self,
                filtered_states=filtered_states,
                predicted_states=predicted_states,
                system_matrices=system_matrices,
                shape_system_matrices=shape_system_matrices,
                lag=lag,
            )

        validated_smooth._pyrecest_strict_lag_contract = True
        smoother_type.smooth = validated_smooth
