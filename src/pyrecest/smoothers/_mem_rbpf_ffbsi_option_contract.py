"""Strict option validation for the MEM-RBPF FFBSi smoother."""

from __future__ import annotations

from functools import wraps
from operator import index as _operator_index
from typing import Any

import numpy as np

from . import mem_rbpf_ffbsi_smoother as _implementation


def _normalize_integer_option(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int:
    """Return an exact integer option satisfying the requested lower bound."""

    qualifier = "positive" if minimum == 1 else "non-negative"
    message = f"{name} must be a {qualifier} integer"
    if np.ma.is_masked(value) or isinstance(
        value,
        (bool, np.bool_, np.datetime64, np.timedelta64),
    ):
        raise ValueError(message)
    try:
        parsed = _operator_index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if parsed < minimum:
        raise ValueError(message)
    return int(parsed)


def _normalize_bool_option(value: Any, name: str) -> bool:
    """Return a strict Python Boolean without truthiness coercion."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a bool")


def install_mem_rbpf_ffbsi_option_contract() -> None:
    """Install strict validation for constructor and per-call FFBSi options."""

    smoother_type = _implementation.MEMRBPFFFBSiSmoother

    current_init = smoother_type.__init__
    if not getattr(current_init, "_pyrecest_strict_option_contract", False):

        @wraps(current_init)
        def validated_init(
            self,
            n_trajectories: int | None = None,
            sample_axis: bool = True,
            angle_wrap_terms: int = 2,
            axis_floor: float = 1e-9,
        ):
            if n_trajectories is not None:
                n_trajectories = _normalize_integer_option(
                    n_trajectories,
                    "n_trajectories",
                    minimum=1,
                )
            angle_wrap_terms = _normalize_integer_option(
                angle_wrap_terms,
                "angle_wrap_terms",
                minimum=0,
            )
            return current_init(
                self,
                n_trajectories=n_trajectories,
                sample_axis=sample_axis,
                angle_wrap_terms=angle_wrap_terms,
                axis_floor=axis_floor,
            )

        validated_init._pyrecest_strict_option_contract = True
        smoother_type.__init__ = validated_init

    current_smooth = smoother_type.smooth
    if not getattr(current_smooth, "_pyrecest_strict_option_contract", False):

        @wraps(current_smooth)
        def validated_smooth(
            self,
            records,
            rng=None,
            *,
            n_trajectories: int | None = None,
            sample_axis: bool | None = None,
            angle_wrap_terms: int | None = None,
            full_axis_lengths: bool = True,
        ):
            if n_trajectories is not None:
                n_trajectories = _normalize_integer_option(
                    n_trajectories,
                    "n_trajectories",
                    minimum=1,
                )
            if angle_wrap_terms is not None:
                angle_wrap_terms = _normalize_integer_option(
                    angle_wrap_terms,
                    "angle_wrap_terms",
                    minimum=0,
                )
            full_axis_lengths = _normalize_bool_option(
                full_axis_lengths,
                "full_axis_lengths",
            )
            return current_smooth(
                self,
                records,
                rng,
                n_trajectories=n_trajectories,
                sample_axis=sample_axis,
                angle_wrap_terms=angle_wrap_terms,
                full_axis_lengths=full_axis_lengths,
            )

        validated_smooth._pyrecest_strict_option_contract = True
        smoother_type.smooth = validated_smooth
