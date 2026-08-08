"""Strict lag validation for fixed-lag random-matrix smoothers."""

from __future__ import annotations

from functools import wraps
from operator import index as _operator_index
from typing import Any

import numpy as np

from . import fixed_lag_random_matrix_smoother as _implementation


def _normalize_lag(value: Any) -> int:
    """Return ``value`` as an exact non-negative integer lag."""

    message = "lag must be a non-negative integer."
    if np.ma.is_masked(value) or isinstance(
        value,
        (bool, np.bool_, np.datetime64, np.timedelta64),
    ):
        raise ValueError(message)
    try:
        parsed = _operator_index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if parsed < 0:
        raise ValueError(message)
    return int(parsed)


def _install_random_matrix_smoother_contract() -> None:
    smoother_type = _implementation.FixedLagRandomMatrixSmoother

    current_init = smoother_type.__init__
    if not getattr(current_init, "_pyrecest_lag_contract", False):

        @wraps(current_init)
        def validated_init(
            self,
            lag: int = 1,
            extent_smoothing: str = "granstrom",
            extent_smoothing_factor: float = 1.0,
            minimum_extent_weight: float = 1e-12,
            extent_transition_dof: float | None = None,
        ):
            return current_init(
                self,
                lag=_normalize_lag(lag),
                extent_smoothing=extent_smoothing,
                extent_smoothing_factor=extent_smoothing_factor,
                minimum_extent_weight=minimum_extent_weight,
                extent_transition_dof=extent_transition_dof,
            )

        validated_init._pyrecest_lag_contract = True
        smoother_type.__init__ = validated_init

    current_smooth = smoother_type.smooth
    if not getattr(current_smooth, "_pyrecest_lag_contract", False):

        @wraps(current_smooth)
        def validated_smooth(
            self,
            filtered_states,
            predicted_states=None,
            system_matrices=None,
            lag: int | None = None,
        ):
            if lag is not None:
                lag = _normalize_lag(lag)
            return current_smooth(
                self,
                filtered_states=filtered_states,
                predicted_states=predicted_states,
                system_matrices=system_matrices,
                lag=lag,
            )

        validated_smooth._pyrecest_lag_contract = True
        smoother_type.smooth = validated_smooth


def _install_factorized_giw_smoother_contract() -> None:
    smoother_type = _implementation.FixedLagFactorizedGIWRandomMatrixSmoother

    current_init = smoother_type.__init__
    if not getattr(current_init, "_pyrecest_lag_contract", False):

        @wraps(current_init)
        def validated_init(
            self,
            lag: int = 1,
            extent_smoothing: str = "granstrom",
            extent_transition_dof: float = 100.0,
            minimum_extent_weight: float = 1e-12,
            minimum_extent_eigenvalue: float = 1e-12,
        ):
            return current_init(
                self,
                lag=_normalize_lag(lag),
                extent_smoothing=extent_smoothing,
                extent_transition_dof=extent_transition_dof,
                minimum_extent_weight=minimum_extent_weight,
                minimum_extent_eigenvalue=minimum_extent_eigenvalue,
            )

        validated_init._pyrecest_lag_contract = True
        smoother_type.__init__ = validated_init

    current_smooth = smoother_type.smooth
    if not getattr(current_smooth, "_pyrecest_lag_contract", False):

        @wraps(current_smooth)
        def validated_smooth(
            self,
            filtered_states,
            predicted_states=None,
            system_matrices=None,
            extent_transition_matrices=None,
            lag: int | None = None,
        ):
            if lag is not None:
                lag = _normalize_lag(lag)
            return current_smooth(
                self,
                filtered_states=filtered_states,
                predicted_states=predicted_states,
                system_matrices=system_matrices,
                extent_transition_matrices=extent_transition_matrices,
                lag=lag,
            )

        validated_smooth._pyrecest_lag_contract = True
        smoother_type.smooth = validated_smooth


def install_fixed_lag_random_matrix_lag_contract() -> None:
    """Install exact lag validation on both random-matrix smoother variants."""

    _install_random_matrix_smoother_contract()
    _install_factorized_giw_smoother_contract()
