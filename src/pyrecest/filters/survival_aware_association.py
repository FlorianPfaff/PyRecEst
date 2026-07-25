# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""Track-manager adapters for survival-aware CRP association priors.

The mathematical prior lives in :mod:`pyrecest.filters.survival_aware_crp`.
This module only resolves lifecycle metadata from managed tracks and translates
those prior weights into association-hypothesis costs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import log
from typing import Any

import numpy as np

from .association_hypotheses import (
    AssociationHypothesis,
    MeasurementAxis,
    association_result_from_hypotheses,
    linear_gaussian_association_hypotheses,
)
from .survival_aware_crp import (
    SurvivalAwareCRPAssociationPrior,
    SurvivalAwareTrackEvidence,
)
from .track_manager import AssociationResult

FactorSpec = float | Callable[..., float]


@dataclass(frozen=True)
class SurvivalAwareAssociationConfig:
    """Configuration for the TrackManager survival-aware CRP adapter.

    If ``crp_prior`` is omitted, a default
    :class:`SurvivalAwareCRPAssociationPrior` is created with
    ``temporal_decay=mass_decay`` so a managed track's ``misses`` count acts as
    the CRP last-seen age. ``existence_probability=None`` first looks for
    ``track.metadata[metadata_existence_key]``, then for
    ``track.existence_probability``, and finally falls back to one.
    """

    crp_prior: SurvivalAwareCRPAssociationPrior | None = None
    survival_probability: FactorSpec = 0.99
    detection_probability: FactorSpec = 0.95
    visibility_probability: FactorSpec = 1.0
    existence_probability: FactorSpec | None = None
    appearance_likelihood: FactorSpec = 1.0
    mass_decay: float = 1.0
    mass_power: float = 1.0
    birth_weight: float = 1.0
    clutter_weight: float = 1.0
    minimum_probability: float = 1.0e-12
    metadata_existence_key: str = "existence_probability"

    def __post_init__(self) -> None:
        if self.crp_prior is not None and not isinstance(
            self.crp_prior, SurvivalAwareCRPAssociationPrior
        ):
            raise TypeError("crp_prior must be a SurvivalAwareCRPAssociationPrior")
        _validate_probability_spec(self.survival_probability, "survival_probability")
        _validate_probability_spec(self.detection_probability, "detection_probability")
        _validate_probability_spec(
            self.visibility_probability, "visibility_probability"
        )
        _validate_nonnegative_likelihood_spec(
            self.appearance_likelihood, "appearance_likelihood"
        )
        if self.existence_probability is not None:
            _validate_probability_spec(
                self.existence_probability, "existence_probability"
            )
        _validate_probability(self.mass_decay, "mass_decay", allow_zero=True)
        _validate_nonnegative_finite(self.mass_power, "mass_power")
        _validate_nonnegative_finite(self.birth_weight, "birth_weight")
        _validate_nonnegative_finite(self.clutter_weight, "clutter_weight")
        if float(self.birth_weight) + float(self.clutter_weight) <= 0.0:
            raise ValueError("birth_weight and clutter_weight cannot both be zero")
        _validate_probability(
            self.minimum_probability, "minimum_probability", allow_zero=False
        )
        if not isinstance(self.metadata_existence_key, str):
            raise ValueError("metadata_existence_key must be a string")


def survival_aware_track_log_prior(
    track: Any,
    measurement: Any | None = None,
    *,
    measurement_index: int | None = None,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
    kinematic_likelihood: float = 1.0,
    appearance_likelihood: float | None = None,
) -> float:
    """Return the CRP-core log prior attractiveness of assigning to ``track``."""

    config = SurvivalAwareAssociationConfig() if config is None else config
    evidence = _survival_aware_track_evidence(
        track,
        measurement,
        measurement_index=measurement_index,
        step=step,
        config=config,
        kinematic_likelihood=kinematic_likelihood,
        appearance_likelihood=appearance_likelihood,
    )
    weight = _crp_prior(config).existing_track_weight(evidence)
    return log(max(float(config.minimum_probability), weight))


def survival_aware_missed_detection_costs(
    tracks: Sequence[Any],
    *,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> np.ndarray:
    """Return per-track GNN costs for leaving tracks unassigned.

    The cost is ``-log(1 - r^- p_D v)`` with predicted existence
    ``r^- = r p_S``. A miss is therefore cheap for low existence, low survival,
    low detection probability, or low visibility, and expensive for a likely
    surviving visible track.
    """

    config = SurvivalAwareAssociationConfig() if config is None else config
    costs = []
    for track in tracks:
        existence = _resolve_existence_probability(
            track, None, measurement_index=None, step=step, config=config
        )
        survival = _resolve_probability(
            config.survival_probability,
            "survival_probability",
            track,
            None,
            measurement_index=None,
            step=step,
        )
        detection = _resolve_probability(
            config.detection_probability,
            "detection_probability",
            track,
            None,
            measurement_index=None,
            step=step,
        )
        visibility = _resolve_probability(
            config.visibility_probability,
            "visibility_probability",
            track,
            None,
            measurement_index=None,
            step=step,
        )
        predicted_existence = (
            SurvivalAwareCRPAssociationPrior.predict_existence_probability(
                existence,
                survival,
            )
        )
        missed_probability = 1.0 - predicted_existence * detection * visibility
        missed_probability = max(
            float(config.minimum_probability), min(1.0, missed_probability)
        )
        costs.append(-log(missed_probability))
    return np.asarray(costs, dtype=float)


def survival_aware_unassigned_measurement_cost(
    config: SurvivalAwareAssociationConfig | None = None,
    *,
    num_existing_tracks: int = 0,
) -> float:
    """Return the GNN cost for treating a measurement as birth or clutter."""

    config = SurvivalAwareAssociationConfig() if config is None else config
    birth_weight = _crp_prior(config).birth_weight(
        num_existing_tracks, base_birth_weight=config.birth_weight
    )
    total_weight = birth_weight + float(config.clutter_weight)
    return -log(max(float(config.minimum_probability), total_weight))


def apply_survival_aware_prior_to_hypotheses(
    hypotheses: Sequence[AssociationHypothesis],
    tracks: Sequence[Any],
    measurements: Sequence[Any] | None = None,
    *,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> list[AssociationHypothesis]:
    """Fold survival-aware CRP prior weights into association-hypothesis costs."""

    config = SurvivalAwareAssociationConfig() if config is None else config
    adjusted_hypotheses = []
    for hypothesis in hypotheses:
        if hypothesis.is_missed_detection:
            adjusted_hypotheses.append(hypothesis)
            continue

        track_index = int(hypothesis.track_index)
        measurement_index = int(hypothesis.measurement_index)
        measurement = None if measurements is None else measurements[measurement_index]
        prior_log_score = survival_aware_track_log_prior(
            tracks[track_index],
            measurement,
            measurement_index=measurement_index,
            step=step,
            config=config,
        )
        hypothesis_log_score = _hypothesis_log_score(hypothesis)
        log_score = prior_log_score + hypothesis_log_score
        metadata = dict(hypothesis.metadata or {})
        metadata["survival_aware_log_prior"] = prior_log_score
        metadata["survival_aware_measurement_log_score"] = hypothesis_log_score
        metadata["survival_aware_log_score"] = log_score
        metadata["survival_aware_crp_weight"] = float(np.exp(prior_log_score))
        adjusted_hypotheses.append(
            replace(
                hypothesis,
                cost=-log_score,
                log_likelihood=log_score,
                probability=None,
                metadata=metadata,
            )
        )
    return adjusted_hypotheses


def survival_aware_linear_gaussian_association_hypotheses(
    tracks,
    measurements,
    measurement_matrix,
    meas_noise,
    gates=None,
    *,
    measurement_axis: MeasurementAxis = "auto",
    include_rejected: bool = False,
    metadata_builder: Callable[..., dict[str, Any] | None] | None = None,
    strict_backend: bool = False,
    step: int | None = None,
    config: SurvivalAwareAssociationConfig | None = None,
) -> list[AssociationHypothesis]:
    """Build linear/Gaussian hypotheses and apply the survival-aware CRP prior."""

    measurement_vectors = _coerce_measurements_for_prior(
        measurements, measurement_matrix, measurement_axis
    )
    hypotheses = linear_gaussian_association_hypotheses(
        tracks,
        measurements,
        measurement_matrix,
        meas_noise,
        gates=gates,
        measurement_axis=measurement_axis,
        include_rejected=include_rejected,
        metadata_builder=metadata_builder,
        strict_backend=strict_backend,
    )
    return apply_survival_aware_prior_to_hypotheses(
        hypotheses, tracks, measurement_vectors, step=step, config=config
    )


def build_survival_aware_linear_gaussian_hypothesis_associator(
    measurement_matrix,
    meas_noise,
    *,
    config: SurvivalAwareAssociationConfig | None = None,
    gates=None,
    missing_cost: float = np.inf,
    unassigned_measurement_cost: float | Sequence[float] | None = None,
    measurement_axis: MeasurementAxis = "auto",
):
    """Create a :class:`TrackManager`-compatible survival-aware associator."""

    config = SurvivalAwareAssociationConfig() if config is None else config

    def associator(tracks, measurements, **kwargs) -> AssociationResult:
        effective_config = kwargs.get("config", config)
        effective_measurement_matrix = kwargs.get(
            "measurement_matrix", measurement_matrix
        )
        effective_measurement_axis = kwargs.get("measurement_axis", measurement_axis)
        hypotheses = survival_aware_linear_gaussian_association_hypotheses(
            tracks,
            measurements,
            effective_measurement_matrix,
            kwargs.get("meas_noise", meas_noise),
            gates=kwargs.get("gates", gates),
            measurement_axis=effective_measurement_axis,
            include_rejected=kwargs.get("include_rejected", False),
            metadata_builder=kwargs.get("metadata_builder", None),
            strict_backend=kwargs.get("strict_backend", False),
            step=kwargs.get("step", None),
            config=effective_config,
        )
        measurement_vectors = _coerce_measurements_for_prior(
            measurements, effective_measurement_matrix, effective_measurement_axis
        )
        if unassigned_measurement_cost is None:
            default_unassigned_measurement_cost = (
                survival_aware_unassigned_measurement_cost(
                    effective_config, num_existing_tracks=len(tracks)
                )
            )
        else:
            default_unassigned_measurement_cost = unassigned_measurement_cost
        return association_result_from_hypotheses(
            hypotheses,
            num_tracks=len(tracks),
            num_measurements=len(measurement_vectors),
            missing_cost=kwargs.get("missing_cost", missing_cost),
            unassigned_track_cost=kwargs.get(
                "unassigned_track_cost",
                survival_aware_missed_detection_costs(
                    tracks, step=kwargs.get("step", None), config=effective_config
                ),
            ),
            unassigned_measurement_cost=kwargs.get(
                "unassigned_measurement_cost", default_unassigned_measurement_cost
            ),
        )

    return associator


def _crp_prior(
    config: SurvivalAwareAssociationConfig,
) -> SurvivalAwareCRPAssociationPrior:
    if config.crp_prior is not None:
        return config.crp_prior
    return SurvivalAwareCRPAssociationPrior(
        temporal_decay=float(config.mass_decay),
        minimum_total_weight=float(config.minimum_probability),
    )


def _survival_aware_track_evidence(
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
    config: SurvivalAwareAssociationConfig,
    kinematic_likelihood: float,
    appearance_likelihood: float | None,
) -> SurvivalAwareTrackEvidence:
    return SurvivalAwareTrackEvidence(
        mass=_effective_track_mass(track, config),
        existence_probability=_resolve_existence_probability(
            track,
            measurement,
            measurement_index=measurement_index,
            step=step,
            config=config,
        ),
        survival_probability=_resolve_probability(
            config.survival_probability,
            "survival_probability",
            track,
            measurement,
            measurement_index=measurement_index,
            step=step,
        ),
        detection_probability=_resolve_probability(
            config.detection_probability,
            "detection_probability",
            track,
            measurement,
            measurement_index=measurement_index,
            step=step,
        ),
        visibility_probability=_resolve_probability(
            config.visibility_probability,
            "visibility_probability",
            track,
            measurement,
            measurement_index=measurement_index,
            step=step,
        ),
        kinematic_likelihood=_validate_nonnegative_likelihood(
            kinematic_likelihood, "kinematic_likelihood"
        ),
        appearance_likelihood=_resolve_appearance_likelihood(
            config,
            track,
            measurement,
            measurement_index=measurement_index,
            step=step,
            override=appearance_likelihood,
        ),
        last_seen_steps=max(0, int(getattr(track, "misses", 0))),
    )


def _effective_track_mass(track: Any, config: SurvivalAwareAssociationConfig) -> float:
    hits = max(1, int(getattr(track, "hits", 1)))
    mass = float(hits) ** float(config.mass_power)
    return max(float(config.minimum_probability), mass)


def _resolve_existence_probability(
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
    config: SurvivalAwareAssociationConfig,
) -> float:
    if config.existence_probability is not None:
        return _resolve_probability(
            config.existence_probability,
            "existence_probability",
            track,
            measurement,
            measurement_index=measurement_index,
            step=step,
        )

    metadata = getattr(track, "metadata", None)
    if isinstance(metadata, dict) and config.metadata_existence_key in metadata:
        return _validate_probability(
            metadata[config.metadata_existence_key],
            f"metadata[{config.metadata_existence_key!r}]",
            allow_zero=True,
        )
    if hasattr(track, "existence_probability"):
        return _validate_probability(
            getattr(track, "existence_probability"),
            "track.existence_probability",
            allow_zero=True,
        )
    return 1.0


def _resolve_probability(
    spec: FactorSpec,
    name: str,
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
) -> float:
    value = _resolve_value(
        spec, track, measurement, measurement_index=measurement_index, step=step
    )
    return _validate_probability(value, name, allow_zero=True)


def _resolve_appearance_likelihood(
    config: SurvivalAwareAssociationConfig,
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
    override: float | None,
) -> float:
    if override is not None:
        return _validate_nonnegative_likelihood(override, "appearance_likelihood")
    value = _resolve_value(
        config.appearance_likelihood,
        track,
        measurement,
        measurement_index=measurement_index,
        step=step,
    )
    return _validate_nonnegative_likelihood(value, "appearance_likelihood")


def _resolve_value(
    spec: FactorSpec,
    track: Any,
    measurement: Any | None,
    *,
    measurement_index: int | None,
    step: int | None,
) -> float:
    if not callable(spec):
        return float(_as_numpy_array(spec).item())

    call_attempts = (
        lambda: spec(
            track=track,
            measurement=measurement,
            measurement_index=measurement_index,
            step=step,
        ),
        lambda: spec(track, measurement, measurement_index, step),
        lambda: spec(track, measurement),
        lambda: spec(track),
    )
    last_error: TypeError | None = None
    for attempt in call_attempts:
        try:
            return float(_as_numpy_array(attempt()).item())
        except TypeError as exc:
            last_error = exc
    raise TypeError(
        "Callable factors must accept keyword arguments, "
        "(track, measurement, measurement_index, step), "
        "(track, measurement), or track"
    ) from last_error


def _hypothesis_log_score(hypothesis: AssociationHypothesis) -> float:
    if hypothesis.log_likelihood is not None:
        return float(hypothesis.log_likelihood)
    if hypothesis.probability is not None:
        probability = float(hypothesis.probability)
        if probability <= 0.0:
            return float("-inf")
        return log(probability)
    if hypothesis.cost is not None:
        return -float(hypothesis.cost)
    return 0.0


def _validate_probability_spec(spec: FactorSpec, name: str) -> None:
    if callable(spec):
        return
    _validate_probability(spec, name, allow_zero=True)


def _validate_nonnegative_likelihood_spec(spec: FactorSpec, name: str) -> None:
    if callable(spec):
        return
    _validate_nonnegative_likelihood(spec, name)


def _validate_probability(value: Any, name: str, *, allow_zero: bool) -> float:
    value_array = _as_numpy_array(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a scalar probability")
    probability = float(value_array.item())
    lower_ok = probability >= 0.0 if allow_zero else probability > 0.0
    if not lower_ok or probability > 1.0 or not np.isfinite(probability):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return probability


def _validate_nonnegative_finite(value: Any, name: str) -> float:
    value_array = _as_numpy_array(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a scalar number")
    scalar = float(value_array.item())
    if scalar < 0.0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def _validate_nonnegative_likelihood(value: Any, name: str) -> float:
    value_array = _as_numpy_array(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(f"{name} must be a scalar likelihood")
    likelihood = float(value_array.item())
    if likelihood < 0.0 or not np.isfinite(likelihood):
        raise ValueError(f"{name} must be finite and nonnegative")
    return likelihood


def _as_numpy_array(value: Any, *, dtype=float) -> np.ndarray:
    """Return ``value`` as a NumPy array, respecting active backend adapters."""

    try:
        from pyrecest.backend import to_numpy  # pylint: disable=import-outside-toplevel

        value = to_numpy(value)
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return np.asarray(value, dtype=dtype)


def _coerce_measurements_for_prior(
    measurements, measurement_matrix, measurement_axis: MeasurementAxis
) -> list[np.ndarray]:
    measurement_matrix = _as_numpy_array(measurement_matrix)
    measurement_dim = int(measurement_matrix.shape[0])
    try:
        array = _as_numpy_array(measurements)
    except (TypeError, ValueError, RuntimeError):
        return [
            _as_numpy_array(measurement).reshape(-1) for measurement in measurements
        ]
    if array.ndim == 1:
        return [array.reshape(-1)]
    if array.ndim != 2:
        return [
            _as_numpy_array(measurement).reshape(-1) for measurement in measurements
        ]
    if measurement_axis == "columns":
        return [array[:, index].reshape(-1) for index in range(array.shape[1])]
    if measurement_axis in ("rows", "sequence"):
        return [array[index, :].reshape(-1) for index in range(array.shape[0])]
    if measurement_axis != "auto":
        raise ValueError(
            "measurement_axis must be 'auto', 'columns', 'rows', or 'sequence'"
        )
    columns_match = array.shape[0] == measurement_dim
    rows_match = array.shape[1] == measurement_dim
    if columns_match and not rows_match:
        return [array[:, index].reshape(-1) for index in range(array.shape[1])]
    if rows_match and not columns_match:
        return [array[index, :].reshape(-1) for index in range(array.shape[0])]
    if columns_match and rows_match and array.shape == (1, 1):
        return [array[:, 0].reshape(-1)]
    if columns_match and rows_match:
        raise ValueError(
            "Ambiguous measurement array orientation for measurement_axis='auto'. "
            "Pass measurement_axis='columns' or 'rows' explicitly."
        )
    raise ValueError(
        "Neither axis of measurements matches the measurement dimension inferred "
        "from measurement_matrix"
    )


__all__ = [
    "SurvivalAwareAssociationConfig",
    "apply_survival_aware_prior_to_hypotheses",
    "build_survival_aware_linear_gaussian_hypothesis_associator",
    "survival_aware_linear_gaussian_association_hypotheses",
    "survival_aware_missed_detection_costs",
    "survival_aware_track_log_prior",
    "survival_aware_unassigned_measurement_cost",
]
