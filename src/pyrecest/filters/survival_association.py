# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""Survival-aware association priors for multitarget track management.

This module provides a small CRP-inspired prior layer that can be composed with
PyRecEst association hypotheses and :class:`TrackManager` associators.  It
replaces raw cluster popularity with effective track mass, existence, survival,
detection, and visibility factors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import log
from typing import Any, Literal

import numpy as np

from .association_hypotheses import (
    AssociationHypothesis,
    _coerce_measurements,
    association_result_from_hypotheses,
    hypothesis_cost,
    linear_gaussian_association_hypotheses,
)

CostMode = Literal["auto", "cost", "log_likelihood"]


@dataclass(frozen=True)
class TrackSurvivalPriorComponents:
    """Resolved prior factors for one track."""

    track_mass: float
    existence_probability: float
    survival_probability: float
    detection_probability: float
    visibility_probability: float
    steps_since_seen: int
    epsilon: float = 1e-12

    @property
    def assignment_weight(self) -> float:
        return max(
            self.track_mass
            * self.existence_probability
            * self.survival_probability
            * self.detection_probability
            * self.visibility_probability,
            self.epsilon,
        )

    @property
    def log_assignment_weight(self) -> float:
        return log(self.assignment_weight)

    @property
    def missed_detection_probability(self) -> float:
        detection_probability = min(
            self.existence_probability
            * self.survival_probability
            * self.detection_probability
            * self.visibility_probability,
            1.0,
        )
        return max(1.0 - detection_probability, self.epsilon)

    @property
    def missed_detection_cost(self) -> float:
        return -log(self.missed_detection_probability)


@dataclass(frozen=True)
class SurvivalAwareAssociationConfig:
    """Configuration for survival-aware association weighting.

    Probability-valued fields may be scalars, length-matching vectors, or
    callables. Track callables receive ``track``, ``track_index`` and
    ``current_step``. Measurement callables receive ``measurement``,
    ``measurement_index`` and ``current_step``.
    """

    existence_probability: Any = None
    survival_probability: Any = 1.0
    detection_probability: Any = None
    visibility_probability: Any = None
    track_mass: Any = None
    track_mass_decay: float = 1.0
    minimum_track_mass: float = 1.0
    birth_probability: Any = 1.0
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        _as_probability(self.track_mass_decay, "track_mass_decay")
        _as_nonnegative_scalar(self.minimum_track_mass, "minimum_track_mass")
        _as_positive_scalar(self.epsilon, "epsilon")


def track_survival_prior_components(
    tracks: Sequence[Any],
    *,
    current_step: int | None = None,
    config: SurvivalAwareAssociationConfig | dict[str, Any] | None = None,
) -> list[TrackSurvivalPriorComponents]:
    """Resolve survival-aware prior factors for every track."""
    cfg = _as_config(config)
    return [
        _components_for_track(tracks, i, current_step=current_step, config=cfg)
        for i in range(len(tracks))
    ]


def survival_aware_missed_detection_costs(
    tracks: Sequence[Any],
    *,
    current_step: int | None = None,
    config: SurvivalAwareAssociationConfig | dict[str, Any] | None = None,
) -> np.ndarray:
    """Return per-track costs for leaving tracks unmatched."""
    return np.asarray(
        [
            component.missed_detection_cost
            for component in track_survival_prior_components(
                tracks, current_step=current_step, config=config
            )
        ]
    )


def survival_aware_birth_costs(
    measurements: Sequence[Any],
    *,
    current_step: int | None = None,
    config: SurvivalAwareAssociationConfig | dict[str, Any] | None = None,
    measurement_axis: str = "auto",
    measurement_dim: int | None = None,
) -> np.ndarray:
    """Return per-measurement costs for leaving measurements unmatched."""
    cfg = _as_config(config)
    measurement_vectors = _coerce_measurements(
        measurements,
        measurement_axis=measurement_axis,  # type: ignore[arg-type]
        measurement_dim=measurement_dim,
    )
    return np.asarray(
        [
            -log(
                max(
                    _resolve_measurement_probability(
                        cfg.birth_probability,
                        measurement_vectors,
                        i,
                        "birth_probability",
                        current_step=current_step,
                        default_value=1.0,
                    ),
                    cfg.epsilon,
                )
            )
            for i in range(len(measurement_vectors))
        ]
    )


def apply_survival_association_prior(
    hypotheses: Sequence[AssociationHypothesis],
    tracks: Sequence[Any],
    *,
    current_step: int | None = None,
    config: SurvivalAwareAssociationConfig | dict[str, Any] | None = None,
    cost_mode: CostMode = "auto",
) -> list[AssociationHypothesis]:
    """Fold survival-aware prior factors into pairwise hypothesis scores."""
    _validate_cost_mode(cost_mode)
    components = track_survival_prior_components(
        tracks, current_step=current_step, config=config
    )
    adjusted = []
    for hypothesis in hypotheses:
        if hypothesis.is_missed_detection:
            adjusted.append(hypothesis)
            continue
        component = components[
            _valid_track_index(hypothesis.track_index, len(components))
        ]
        log_weight = component.log_assignment_weight
        adjusted_log_likelihood = (
            None
            if hypothesis.log_likelihood is None
            else float(hypothesis.log_likelihood) + log_weight
        )
        adjusted_probability = (
            None
            if hypothesis.probability is None
            else float(hypothesis.probability) * component.assignment_weight
        )
        metadata = dict(hypothesis.metadata or {})
        metadata["survival_prior"] = {
            "track_mass": component.track_mass,
            "existence_probability": component.existence_probability,
            "survival_probability": component.survival_probability,
            "detection_probability": component.detection_probability,
            "visibility_probability": component.visibility_probability,
            "steps_since_seen": component.steps_since_seen,
            "assignment_weight": component.assignment_weight,
            "log_assignment_weight": log_weight,
        }
        adjusted.append(
            replace(
                hypothesis,
                cost=_adjusted_cost(
                    hypothesis, log_weight, adjusted_log_likelihood, cost_mode
                ),
                log_likelihood=adjusted_log_likelihood,
                probability=adjusted_probability,
                metadata=metadata,
            )
        )
    return adjusted


def build_survival_aware_linear_gaussian_associator(
    measurement_matrix,
    meas_noise,
    *,
    config: SurvivalAwareAssociationConfig | dict[str, Any] | None = None,
    gates=None,
    missing_cost: float = np.inf,
    unassigned_track_cost: float | Sequence[float] | None = None,
    unassigned_measurement_cost: float | Sequence[float] | None = None,
    measurement_axis: str = "auto",
    cost_mode: CostMode = "auto",
):
    """Create a TrackManager-compatible linear/Gaussian associator."""
    default_config = _as_config(config)
    _validate_cost_mode(cost_mode)

    def associator(tracks, measurements, **kwargs):
        cfg = _as_config(kwargs.get("survival_config", default_config))
        eff_matrix = kwargs.get("measurement_matrix", measurement_matrix)
        eff_axis = kwargs.get("measurement_axis", measurement_axis)
        eff_cost_mode = kwargs.get("cost_mode", cost_mode)
        eff_step = kwargs.get("current_step", None)
        _validate_cost_mode(eff_cost_mode)
        measurement_dim = int(np.asarray(eff_matrix, dtype=float).shape[0])
        hypotheses = linear_gaussian_association_hypotheses(
            tracks,
            measurements,
            eff_matrix,
            kwargs.get("meas_noise", meas_noise),
            gates=kwargs.get("gates", gates),
            measurement_axis=eff_axis,
            strict_backend=kwargs.get("strict_backend", False),
        )
        adjusted = apply_survival_association_prior(
            hypotheses,
            tracks,
            current_step=eff_step,
            config=cfg,
            cost_mode=eff_cost_mode,
        )
        measurement_vectors = _coerce_measurements(
            measurements, measurement_axis=eff_axis, measurement_dim=measurement_dim
        )
        track_cost = kwargs.get("unassigned_track_cost", unassigned_track_cost)
        if track_cost is None:
            track_cost = survival_aware_missed_detection_costs(
                tracks, current_step=eff_step, config=cfg
            )
        measurement_cost = kwargs.get(
            "unassigned_measurement_cost", unassigned_measurement_cost
        )
        if measurement_cost is None:
            measurement_cost = survival_aware_birth_costs(
                measurement_vectors,
                current_step=eff_step,
                config=cfg,
                measurement_axis="sequence",
                measurement_dim=measurement_dim,
            )
        return association_result_from_hypotheses(
            adjusted,
            num_tracks=len(tracks),
            num_measurements=len(measurement_vectors),
            missing_cost=kwargs.get("missing_cost", missing_cost),
            unassigned_track_cost=track_cost,
            unassigned_measurement_cost=measurement_cost,
        )

    return associator


def _as_config(config):
    if config is None:
        return SurvivalAwareAssociationConfig()
    if isinstance(config, SurvivalAwareAssociationConfig):
        return config
    return SurvivalAwareAssociationConfig(**dict(config))


def _components_for_track(tracks, track_index, *, current_step, config):
    track = tracks[track_index]
    miss_count = _misses(track)
    steps_since_seen = _steps_since_seen(track, current_step=current_step)
    base_mass = _resolve_track_nonnegative(
        config.track_mass,
        tracks,
        track_index,
        "track_mass",
        current_step=current_step,
        default_value=max(float(_hits(track)), 1.0),
    )
    track_mass = max(
        base_mass * float(config.track_mass_decay) ** miss_count,
        float(config.minimum_track_mass),
    )
    survival = (
        _resolve_track_probability(
            config.survival_probability,
            tracks,
            track_index,
            "survival_probability",
            current_step=current_step,
            default_value=1.0,
        )
        ** steps_since_seen
    )
    existence = _resolve_track_probability(
        config.existence_probability,
        tracks,
        track_index,
        "existence_probability",
        current_step=current_step,
        attr_name="existence_probability",
        metadata_key="existence_probability",
        default_value=1.0,
    )
    detection = _resolve_track_probability(
        config.detection_probability,
        tracks,
        track_index,
        "detection_probability",
        current_step=current_step,
        metadata_key="detection_probability",
        default_value=1.0,
    )
    visibility = _resolve_track_probability(
        config.visibility_probability,
        tracks,
        track_index,
        "visibility_probability",
        current_step=current_step,
        metadata_key="visibility_probability",
        default_value=1.0,
    )
    return TrackSurvivalPriorComponents(
        track_mass=track_mass,
        existence_probability=existence,
        survival_probability=survival,
        detection_probability=detection,
        visibility_probability=visibility,
        steps_since_seen=steps_since_seen,
        epsilon=float(config.epsilon),
    )


def _adjusted_cost(hypothesis, log_weight, adjusted_log_likelihood, cost_mode):
    if cost_mode == "log_likelihood" or (
        cost_mode == "auto" and adjusted_log_likelihood is not None
    ):
        if adjusted_log_likelihood is not None:
            return -float(adjusted_log_likelihood)
    return hypothesis_cost(hypothesis) - log_weight


def _resolve_track_probability(
    value,
    tracks,
    index,
    name,
    *,
    current_step,
    attr_name=None,
    metadata_key=None,
    default_value,
):
    return _as_probability(
        _resolve_track_value(
            value,
            tracks,
            index,
            name,
            current_step=current_step,
            attr_name=attr_name,
            metadata_key=metadata_key,
            default_value=default_value,
        ),
        name,
    )


def _resolve_track_nonnegative(
    value, tracks, index, name, *, current_step, default_value
):
    return _as_nonnegative_scalar(
        _resolve_track_value(
            value,
            tracks,
            index,
            name,
            current_step=current_step,
            attr_name=None,
            metadata_key=None,
            default_value=default_value,
        ),
        name,
    )


def _resolve_track_value(
    value, tracks, index, name, *, current_step, attr_name, metadata_key, default_value
):
    track = tracks[index]
    if value is None:
        metadata = _metadata(track)
        if attr_name is not None and hasattr(track, attr_name):
            return getattr(track, attr_name)
        if metadata_key is not None and metadata_key in metadata:
            return metadata[metadata_key]
        return default_value
    if callable(value):
        return value(track, track_index=index, current_step=current_step)
    values = np.asarray(value)
    if values.shape == ():
        return values.item()
    if values.size != len(tracks):
        raise ValueError(
            f"{name} must be scalar, callable, or have length {len(tracks)}"
        )
    return values.reshape(-1)[index]


def _resolve_measurement_probability(
    value, measurements, index, name, *, current_step, default_value
):
    if value is None:
        resolved = default_value
    elif callable(value):
        resolved = value(
            measurements[index], measurement_index=index, current_step=current_step
        )
    else:
        values = np.asarray(value)
        if values.shape == ():
            resolved = values.item()
        else:
            if values.size != len(measurements):
                raise ValueError(
                    f"{name} must be scalar, callable, or have length {len(measurements)}"
                )
            resolved = values.reshape(-1)[index]
    return _as_probability(resolved, name)


def _metadata(track):
    metadata = getattr(track, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _hits(track):
    return max(int(getattr(track, "hits", 1)), 1)


def _misses(track):
    return max(int(getattr(track, "misses", 0)), 0)


def _steps_since_seen(track, *, current_step):
    metadata = _metadata(track)
    if current_step is not None and "last_seen_step" in metadata:
        return max(int(current_step) - int(metadata["last_seen_step"]), 0)
    if hasattr(track, "misses"):
        return _misses(track) + 1
    if current_step is not None and hasattr(track, "last_step"):
        return max(int(current_step) - int(getattr(track, "last_step")), 0)
    return 1


def _valid_track_index(value, num_tracks):
    values = np.asarray(value)
    if values.shape != () or values.dtype == np.bool_:
        raise ValueError("hypothesis.track_index must be a nonnegative integer")
    scalar = values.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("hypothesis.track_index must be a nonnegative integer")
    if isinstance(scalar, (int, np.integer)):
        index = int(scalar)
    elif (
        isinstance(scalar, (float, np.floating))
        and np.isfinite(scalar)
        and float(scalar).is_integer()
    ):
        index = int(scalar)
    else:
        raise ValueError("hypothesis.track_index must be a nonnegative integer")
    if index < 0 or index >= num_tracks:
        raise ValueError("hypothesis.track_index is out of range")
    return index


def _validate_cost_mode(cost_mode):
    if cost_mode not in ("auto", "cost", "log_likelihood"):
        raise ValueError("cost_mode must be 'auto', 'cost', or 'log_likelihood'")


def _as_scalar_float(value, name):
    values = np.asarray(value)
    if values.shape != () or values.dtype == np.bool_:
        raise ValueError(f"{name} must be a scalar number")
    try:
        scalar = float(values.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar number") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_probability(value, name):
    scalar = _as_scalar_float(value, name)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return scalar


def _as_nonnegative_scalar(value, name):
    scalar = _as_scalar_float(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _as_positive_scalar(value, name):
    scalar = _as_scalar_float(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


__all__ = [
    "SurvivalAwareAssociationConfig",
    "TrackSurvivalPriorComponents",
    "apply_survival_association_prior",
    "build_survival_aware_linear_gaussian_associator",
    "survival_aware_birth_costs",
    "survival_aware_missed_detection_costs",
    "track_survival_prior_components",
]
