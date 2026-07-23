"""Planning utilities for robust linear-Gaussian measurement updates.

The helpers in this module compute the quantities needed before a Kalman-style
linear measurement update: innovation, normalized innovation squared (NIS),
chi-square gating thresholds, hard-rejection decisions, and robust covariance
inflation scales.  They deliberately stop short of mutating a filter state so
that trackers, out-of-sequence updaters, and evaluation code can share the same
pre-update policy logic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.stats import chi2

try:  # Prefer PyRecEst's backend-aware primitives when the full package exists.
    from ._linear_gaussian import huber_covariance_scale as _huber_covariance_scale
    from ._linear_gaussian import (
        normalized_innovation_squared as _normalized_innovation_squared,
    )
    from ._linear_gaussian import (
        student_t_covariance_scale as _student_t_covariance_scale,
    )
except Exception:  # pragma: no cover - only used by standalone downstream copies
    _huber_covariance_scale = None
    _normalized_innovation_squared = None
    _student_t_covariance_scale = None

ROBUST_UPDATE_MODES = ("nis-inflate", "student-t", "huber")
DEFAULT_STUDENT_T_DOF = 4.0
DEFAULT_HUBER_THRESHOLD = 2.0
_TEMPORAL_TYPES = (np.datetime64, np.timedelta64)
_INVALID_REAL_NUMERIC_TYPES = (bool, np.bool_, *_TEMPORAL_TYPES)


class MeasurementLike(Protocol):
    """Structural protocol for source-specific update-policy lookup."""

    source: str
    vector: Any


@dataclass(frozen=True)
class LinearUpdatePlan:
    """Precomputed decision and diagnostics for one linear measurement update."""

    vector: np.ndarray
    covariance: np.ndarray
    observation: np.ndarray
    residual: np.ndarray
    nominal_innovation_covariance: np.ndarray
    innovation_covariance: np.ndarray
    nis: float
    residual_norm: float
    gate_threshold: float | None
    safety_gate_threshold: float | None
    residual_threshold: float | None
    covariance_scale: float
    action: str
    accepted: bool
    inflation_alpha: float
    robust_update: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def chi_square_gate_threshold(
    probability: float | None, measurement_dim: int
) -> float | None:
    """Return the chi-square NIS gate for a measurement probability.

    Parameters
    ----------
    probability : float or None
        Gate probability in ``(0, 1)``.  ``None`` disables the gate and returns
        ``None``.
    measurement_dim : int
        Measurement dimension, i.e. the chi-square degrees of freedom.
    """

    if probability is None:
        return None
    probability = _as_finite_scalar(probability, "probability")
    measurement_dim = _as_positive_integer(measurement_dim, "measurement_dim")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    return float(chi2.ppf(probability, measurement_dim))


def normalized_innovation_squared(
    residual: np.ndarray, innovation_covariance: np.ndarray
) -> float:
    """Return ``residual.T @ inv(innovation_covariance) @ residual`` as a float."""

    residual = _as_finite_array(residual, "residual").reshape(-1)
    innovation_covariance = _as_finite_array(
        innovation_covariance,
        "innovation_covariance",
    )
    if innovation_covariance.shape != (residual.size, residual.size):
        raise ValueError(
            "innovation_covariance must have shape (residual_dim, residual_dim)"
        )
    if _normalized_innovation_squared is not None:
        return float(
            np.asarray(
                _normalized_innovation_squared(residual, innovation_covariance),
                dtype=float,
            )
        )
    return float(residual.T @ np.linalg.solve(innovation_covariance, residual))


def student_t_covariance_scale(
    nis: float,
    measurement_dim: int,
    degrees_of_freedom: float = DEFAULT_STUDENT_T_DOF,
) -> float:
    """Return Student-t covariance inflation from NIS."""

    nis = _as_nonnegative_scalar(nis, "nis")
    measurement_dim = _as_positive_integer(measurement_dim, "measurement_dim")
    dof = _as_finite_scalar(degrees_of_freedom, "degrees_of_freedom")
    if dof <= 2.0:
        raise ValueError("degrees_of_freedom must be greater than 2")
    if _student_t_covariance_scale is not None:
        return float(
            np.asarray(
                _student_t_covariance_scale(
                    nis,
                    measurement_dim,
                    dof=dof,
                ),
                dtype=float,
            )
        )
    return float(max(1.0, (dof + float(nis)) / (dof + measurement_dim)))


def huber_covariance_scale(
    nis: float, threshold: float = DEFAULT_HUBER_THRESHOLD
) -> float:
    """Return multivariate Huber covariance inflation from NIS."""

    nis = _as_nonnegative_scalar(nis, "nis")
    threshold = _as_positive_scalar(threshold, "threshold")
    if _huber_covariance_scale is not None:
        return float(
            np.asarray(
                _huber_covariance_scale(nis, huber_threshold=threshold), dtype=float
            )
        )
    return float(max(1.0, np.sqrt(float(nis)) / threshold))


def robust_update_covariance_scale(
    robust_update: str | None,
    *,
    nis: float,
    measurement_dim: int,
    gate_threshold: float | None,
    inflation_alpha: float = 1.0,
    student_t_dof: float = DEFAULT_STUDENT_T_DOF,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
) -> tuple[float, str | None]:
    """Return covariance scale and diagnostic action for a robust update mode."""

    mode = None if robust_update in (None, "none") else str(robust_update)
    if mode is None:
        return 1.0, None
    nis = _as_nonnegative_scalar(nis, "nis")
    if mode == "nis-inflate":
        inflation_alpha = _as_positive_scalar(inflation_alpha, "inflation_alpha")
        if gate_threshold is None:
            return 1.0, None
        gate_threshold = _as_positive_scalar(gate_threshold, "gate_threshold")
        if nis <= gate_threshold:
            return 1.0, None
        scale = max(1.0, (nis / gate_threshold) ** inflation_alpha)
        return float(scale), "inflated"
    if mode == "student-t":
        scale = student_t_covariance_scale(nis, measurement_dim, student_t_dof)
        return scale, "student_t" if scale > 1.0 else None
    if mode == "huber":
        scale = huber_covariance_scale(nis, huber_threshold)
        return scale, "huberized" if scale > 1.0 else None
    raise ValueError(f"unknown robust update mode {robust_update!r}")


def plan_linear_measurement_update(
    *,
    mean: np.ndarray,
    covariance_matrix: np.ndarray,
    measurement_vector: np.ndarray,
    measurement_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    gate_threshold: float | None = None,
    gate_probability: float | None = None,
    safety_gate_threshold: float | None = None,
    safety_gate_probability: float | None = None,
    max_residual_norm: float | None = None,
    robust_update: str | None = None,
    inflation_alpha: float = 1.0,
    student_t_dof: float = DEFAULT_STUDENT_T_DOF,
    huber_threshold: float = DEFAULT_HUBER_THRESHOLD,
    metadata: Mapping[str, Any] | None = None,
) -> LinearUpdatePlan:
    """Prepare shared NIS gating and robust-inflation quantities.

    ``gate_threshold`` and ``safety_gate_threshold`` are NIS thresholds.  Their
    probability counterparts are converted with a chi-square inverse CDF when a
    threshold is not supplied explicitly.  The returned covariance and
    ``innovation_covariance`` are the effective values after any accepted robust
    inflation.  ``nis`` is always computed against the nominal covariance.
    """

    alpha = _as_positive_scalar(inflation_alpha, "inflation_alpha")

    vector = _as_finite_array(measurement_vector, "measurement_vector").reshape(-1)
    measurement_dim = vector.size
    covariance = _as_finite_array(measurement_covariance, "measurement_covariance")
    observation = _as_finite_array(observation_matrix, "observation_matrix")
    state_mean = _as_finite_array(mean, "mean").reshape(-1)
    state_covariance = _as_finite_array(covariance_matrix, "covariance_matrix")

    if measurement_dim <= 0:
        raise ValueError("measurement_vector must contain at least one element")
    if observation.ndim != 2 or observation.shape[0] != measurement_dim:
        raise ValueError(
            "observation_matrix must have shape (measurement_dim, state_dim)"
        )
    if observation.shape[1] != state_mean.size:
        raise ValueError("observation_matrix and mean have incompatible shapes")
    if state_covariance.shape != (state_mean.size, state_mean.size):
        raise ValueError("covariance_matrix must have shape (state_dim, state_dim)")
    if covariance.shape != (measurement_dim, measurement_dim):
        raise ValueError(
            "measurement_covariance must have shape (measurement_dim, measurement_dim)"
        )

    resolved_gate_threshold = _resolve_threshold(
        gate_threshold, gate_probability, measurement_dim, "gate"
    )
    resolved_safety_threshold = _resolve_threshold(
        safety_gate_threshold,
        safety_gate_probability,
        measurement_dim,
        "safety_gate",
    )
    residual_threshold = (
        None
        if max_residual_norm is None
        else _as_nonnegative_scalar(max_residual_norm, "max_residual_norm")
    )

    residual = vector - observation @ state_mean
    nominal_innovation_covariance = (
        observation @ state_covariance @ observation.T + covariance
    )
    nominal_innovation_covariance = _symmetrized(nominal_innovation_covariance)
    nis = normalized_innovation_squared(residual, nominal_innovation_covariance)
    residual_norm = float(np.linalg.norm(residual))

    accepted = True
    action = "updated"
    covariance_scale = 1.0
    effective_covariance = covariance.copy()
    effective_innovation_covariance = nominal_innovation_covariance.copy()

    residual_over_threshold = (
        residual_threshold is not None and residual_norm > residual_threshold
    )
    safety_over_threshold = (
        resolved_safety_threshold is not None and nis > resolved_safety_threshold
    )
    reject_by_residual = residual_over_threshold and (
        resolved_safety_threshold is None or safety_over_threshold
    )

    if reject_by_residual:
        accepted = False
        action = "residual_rejected"
    elif safety_over_threshold:
        accepted = False
        action = "safety_rejected"
    elif (
        resolved_gate_threshold is not None
        and nis > resolved_gate_threshold
        and robust_update in (None, "none")
    ):
        accepted = False
        action = "rejected"
    else:
        covariance_scale, robust_action = robust_update_covariance_scale(
            robust_update,
            nis=nis,
            measurement_dim=measurement_dim,
            gate_threshold=resolved_gate_threshold,
            inflation_alpha=alpha,
            student_t_dof=student_t_dof,
            huber_threshold=huber_threshold,
        )
        if covariance_scale > 1.0:
            effective_covariance = covariance * covariance_scale
            effective_innovation_covariance = _symmetrized(
                observation @ state_covariance @ observation.T + effective_covariance
            )
        if robust_action is not None:
            action = robust_action

    return LinearUpdatePlan(
        vector=vector,
        covariance=effective_covariance,
        observation=observation,
        residual=residual,
        nominal_innovation_covariance=nominal_innovation_covariance,
        innovation_covariance=effective_innovation_covariance,
        nis=float(nis),
        residual_norm=residual_norm,
        gate_threshold=resolved_gate_threshold,
        safety_gate_threshold=resolved_safety_threshold,
        residual_threshold=residual_threshold,
        covariance_scale=float(covariance_scale),
        action=action,
        accepted=bool(accepted),
        inflation_alpha=alpha,
        robust_update=robust_update,
        metadata={} if metadata is None else dict(metadata),
    )


def gate_threshold_for_measurement(
    measurement: MeasurementLike,
    *,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
) -> float | None:
    """Resolve a source-specific NIS gate threshold for a measurement object."""

    source = str(measurement.source)
    if gate_thresholds_by_source and source in gate_thresholds_by_source:
        value = gate_thresholds_by_source[source]
        return (
            None if value is None else _as_nonnegative_scalar(value, "gate_threshold")
        )
    if gate_probabilities_by_source and source in gate_probabilities_by_source:
        probability = gate_probabilities_by_source[source]
        vector = _as_finite_array(measurement.vector, "measurement.vector").reshape(-1)
        return chi_square_gate_threshold(probability, vector.size)
    return None


def robust_update_for_measurement(
    measurement: MeasurementLike,
    *,
    robust_update_by_source: Mapping[str, str | None] | None = None,
) -> str | None:
    """Resolve a source-specific robust update mode for a measurement object."""

    if robust_update_by_source and str(measurement.source) in robust_update_by_source:
        return robust_update_by_source[str(measurement.source)]
    return None


def source_float_value(
    measurement: MeasurementLike,
    values_by_source: Mapping[str, float | None] | None,
    default: float | None = None,
) -> float | None:
    """Return a source-specific scalar value, falling back to ``default``."""

    if values_by_source and str(measurement.source) in values_by_source:
        value = values_by_source[str(measurement.source)]
        return None if value is None else _as_finite_scalar(value, "source value")
    return default


def _resolve_threshold(
    threshold: float | None,
    probability: float | None,
    measurement_dim: int,
    name: str,
) -> float | None:
    if threshold is not None:
        return _as_nonnegative_scalar(threshold, f"{name}_threshold")
    return chi_square_gate_threshold(probability, measurement_dim)


def _symmetrized(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _contains_values_of_type(value: Any, types: tuple[type, ...]) -> bool:
    if isinstance(value, types):
        return True
    try:
        values = np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, types) for item in values)


def _as_finite_array(value: Any, name: str) -> np.ndarray:
    message = f"{name} must contain finite numeric values"
    try:
        raw_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if raw_array.dtype == np.bool_ or raw_array.dtype.kind in "Mm":
        raise ValueError(message)
    if _contains_values_of_type(
        value, _INVALID_REAL_NUMERIC_TYPES
    ) or _contains_values_of_type(raw_array, _INVALID_REAL_NUMERIC_TYPES):
        raise ValueError(message)
    try:
        array = np.asarray(raw_array, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_finite_scalar(value: Any, name: str) -> float:
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if (
        value_array.shape != ()
        or value_array.dtype == np.bool_
        or value_array.dtype.kind in "Mm"
    ):
        raise ValueError(f"{name} must be a finite scalar")
    scalar = value_array.item()
    if isinstance(scalar, _INVALID_REAL_NUMERIC_TYPES):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite scalar")
    return parsed


def _as_positive_scalar(value: Any, name: str) -> float:
    parsed = _as_finite_scalar(value, name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _as_nonnegative_scalar(value: Any, name: str) -> float:
    parsed = _as_finite_scalar(value, name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return parsed


def _as_positive_integer(value: Any, name: str) -> int:
    parsed = _as_nonnegative_scalar(value, name)
    if parsed <= 0.0 or not parsed.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(parsed)


__all__ = [
    "DEFAULT_HUBER_THRESHOLD",
    "DEFAULT_STUDENT_T_DOF",
    "LinearUpdatePlan",
    "MeasurementLike",
    "ROBUST_UPDATE_MODES",
    "chi_square_gate_threshold",
    "gate_threshold_for_measurement",
    "huber_covariance_scale",
    "normalized_innovation_squared",
    "plan_linear_measurement_update",
    "robust_update_covariance_scale",
    "robust_update_for_measurement",
    "source_float_value",
    "student_t_covariance_scale",
]
