# pylint: disable=no-name-in-module,no-member,too-many-arguments,too-many-positional-arguments
"""Data-oriented association hypotheses and gating helpers.

The helpers in this module intentionally avoid a large tracking-framework
hierarchy.  They provide a small common representation for pairwise
track/measurement scores, a few reusable gates, and conversion utilities for
assignment solvers such as global-nearest-neighbor or Murty-style k-best
assignment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import log, pi
from typing import Any, Literal

import numpy as np
from scipy.stats import chi2

from ._linear_gaussian import (
    linear_gaussian_innovation,
    normalized_innovation_squared,
)

GateMode = Literal["track", "measurement"]
MeasurementAxis = Literal["auto", "columns", "rows", "sequence"]


@dataclass(frozen=True)
class AssociationHypothesis:
    """Pairwise association score between one track and one measurement.

    ``measurement_index=None`` denotes a missed-detection hypothesis.  Costs are
    interpreted as values to minimize; likelihoods and probabilities are values
    to maximize.  Conversion helpers use the most explicit available score in
    the order ``cost``, ``normalized_innovation_squared``, ``log_likelihood``,
    then ``probability``.
    """

    track_index: int
    measurement_index: int | None
    cost: float | None = None
    log_likelihood: float | None = None
    probability: float | None = None
    innovation: object | None = None
    innovation_covariance: object | None = None
    normalized_innovation_squared: float | None = None
    accepted: bool = True
    reason: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_missed_detection(self) -> bool:
        """Return whether the hypothesis represents an unmatched track."""
        return self.measurement_index is None

    def with_acceptance(self, accepted: bool, reason: str | None = None):
        """Return a copy with updated gate acceptance metadata."""
        return replace(self, accepted=bool(accepted), reason=reason)


def _measurement_index(hypothesis: AssociationHypothesis) -> int:
    """Return the concrete measurement index for a non-missed hypothesis."""
    measurement_index = hypothesis.measurement_index
    if measurement_index is None:
        raise ValueError("missed-detection hypotheses do not have a measurement index")
    return int(measurement_index)


def missed_detection_hypothesis(
    track_index: int,
    *,
    cost: float | None = None,
    log_likelihood: float | None = None,
    probability: float | None = None,
    reason: str = "missed_detection",
    metadata: dict[str, Any] | None = None,
) -> AssociationHypothesis:
    """Create a missed-detection hypothesis for one track."""
    return AssociationHypothesis(
        track_index=int(track_index),
        measurement_index=None,
        cost=cost,
        log_likelihood=log_likelihood,
        probability=probability,
        accepted=True,
        reason=reason,
        metadata=metadata,
    )


def hypothesis_cost(
    hypothesis: AssociationHypothesis, *, missing_cost: float = np.inf
) -> float:
    """Return a scalar minimization cost for a hypothesis."""
    if hypothesis.cost is not None:
        return float(hypothesis.cost)
    if hypothesis.normalized_innovation_squared is not None:
        return float(hypothesis.normalized_innovation_squared)
    if hypothesis.log_likelihood is not None:
        return -float(hypothesis.log_likelihood)
    if hypothesis.probability is not None:
        probability = float(hypothesis.probability)
        if probability <= 0.0:
            return float("inf")
        return -log(probability)
    return float(missing_cost)


class NISGate:
    """Gate association hypotheses by normalized innovation squared."""

    def __init__(
        self,
        threshold: float | None = None,
        *,
        measurement_dim: int | None = None,
        confidence: float | None = None,
    ):
        if threshold is None:
            if measurement_dim is None or confidence is None:
                raise ValueError(
                    "Either threshold or both measurement_dim and confidence must be provided."
                )
            threshold = chi2.ppf(float(confidence), int(measurement_dim))
        self.threshold = float(threshold)
        if self.threshold < 0.0:
            raise ValueError("threshold must be nonnegative")

    def accepts(self, hypothesis: AssociationHypothesis) -> bool:
        """Return whether ``hypothesis`` is accepted by the NIS threshold."""
        if hypothesis.is_missed_detection:
            return True
        if hypothesis.normalized_innovation_squared is None:
            return False
        return float(hypothesis.normalized_innovation_squared) <= self.threshold

    def __call__(self, hypothesis: AssociationHypothesis) -> bool:
        return self.accepts(hypothesis)


class CostThresholdGate:
    """Gate hypotheses by maximum minimization cost."""

    def __init__(self, threshold: float, *, missing_cost: float = np.inf):
        self.threshold = float(threshold)
        self.missing_cost = float(missing_cost)

    def accepts(self, hypothesis: AssociationHypothesis) -> bool:
        if hypothesis.is_missed_detection:
            return True
        return (
            hypothesis_cost(hypothesis, missing_cost=self.missing_cost)
            <= self.threshold
        )

    def __call__(self, hypothesis: AssociationHypothesis) -> bool:
        return self.accepts(hypothesis)


class ProbabilityThresholdGate:
    """Gate hypotheses by minimum probability or likelihood."""

    def __init__(self, threshold: float, *, use_likelihood: bool = False):
        self.threshold = float(threshold)
        self.use_likelihood = bool(use_likelihood)
        if self.threshold < 0.0:
            raise ValueError("threshold must be nonnegative")

    def accepts(self, hypothesis: AssociationHypothesis) -> bool:
        if hypothesis.is_missed_detection:
            return True
        if self.use_likelihood:
            if hypothesis.log_likelihood is None:
                return False
            return float(hypothesis.log_likelihood) >= log(self.threshold)
        if hypothesis.probability is None:
            return False
        return float(hypothesis.probability) >= self.threshold

    def __call__(self, hypothesis: AssociationHypothesis) -> bool:
        return self.accepts(hypothesis)


class TopKGate:
    """Keep the best ``k`` hypotheses per track or per measurement."""

    def __init__(
        self, k: int, *, mode: GateMode = "track", missing_cost: float = np.inf
    ):
        self.k = int(k)
        if self.k <= 0:
            raise ValueError("k must be positive")
        if mode not in ("track", "measurement"):
            raise ValueError("mode must be 'track' or 'measurement'")
        self.mode = mode
        self.missing_cost = float(missing_cost)

    def filter(
        self, hypotheses: Sequence[AssociationHypothesis]
    ) -> list[AssociationHypothesis]:
        """Return hypotheses accepted by the top-k rule."""
        accepted_keys = set()
        grouped: dict[int, list[AssociationHypothesis]] = defaultdict(list)
        missed = []
        for hypothesis in hypotheses:
            if hypothesis.is_missed_detection:
                missed.append(hypothesis)
                continue
            key = (
                hypothesis.track_index
                if self.mode == "track"
                else _measurement_index(hypothesis)
            )
            grouped[key].append(hypothesis)

        for group in grouped.values():
            sorted_group = sorted(
                group,
                key=lambda hypothesis: hypothesis_cost(
                    hypothesis, missing_cost=self.missing_cost
                ),
            )
            for hypothesis in sorted_group[: self.k]:
                accepted_keys.add(
                    (hypothesis.track_index, hypothesis.measurement_index)
                )

        result = []
        for hypothesis in hypotheses:
            if hypothesis.is_missed_detection:
                result.append(hypothesis)
                continue
            accepted = (
                hypothesis.track_index,
                hypothesis.measurement_index,
            ) in accepted_keys
            result.append(
                hypothesis.with_acceptance(
                    accepted,
                    None if accepted else f"top_{self.k}_{self.mode}_gate",
                )
            )
        return result

    def accepts(self, hypothesis: AssociationHypothesis) -> bool:
        raise TypeError("TopKGate operates on a collection; use filter(...)")

    def __call__(
        self, hypotheses: Sequence[AssociationHypothesis]
    ) -> list[AssociationHypothesis]:
        return self.filter(hypotheses)


def gate_hypotheses(
    hypotheses: Sequence[AssociationHypothesis],
    gate,
    *,
    reject_reason: str | None = None,
) -> list[AssociationHypothesis]:
    """Apply one gate to hypotheses and preserve rejected diagnostics."""
    if isinstance(gate, TopKGate):
        return gate.filter(hypotheses)
    if hasattr(gate, "filter"):
        filtered = gate.filter(hypotheses)
        if filtered is not None:
            return list(filtered)

    result = []
    for hypothesis in hypotheses:
        accepted = bool(
            gate(hypothesis) if callable(gate) else gate.accepts(hypothesis)
        )
        reason = None if accepted else reject_reason or type(gate).__name__
        result.append(hypothesis.with_acceptance(accepted, reason))
    return result


def filter_hypotheses(
    hypotheses: Sequence[AssociationHypothesis],
    gates=None,
    *,
    accepted_only: bool = True,
) -> list[AssociationHypothesis]:
    """Apply one or more gates and optionally drop rejected hypotheses."""
    result = list(hypotheses)
    if gates is None:
        return (
            [hypothesis for hypothesis in result if hypothesis.accepted]
            if accepted_only
            else result
        )
    if not isinstance(gates, (list, tuple)):
        gates = [gates]
    for gate in gates:
        result = gate_hypotheses(result, gate)
    if accepted_only:
        result = [hypothesis for hypothesis in result if hypothesis.accepted]
    return result


def linear_gaussian_association_hypotheses(
    tracks,
    measurements,
    measurement_matrix,
    meas_noise,
    gates=None,
    *,
    measurement_axis: MeasurementAxis = "auto",
    include_rejected: bool = False,
    metadata_builder: Callable[..., dict[str, Any] | None] | None = None,
) -> list[AssociationHypothesis]:
    """Build Gaussian innovation hypotheses for tracks and measurements."""
    measurement_vectors = _coerce_measurements(
        measurements, measurement_axis=measurement_axis
    )
    measurement_matrix = np.asarray(measurement_matrix, dtype=float)
    covariance_stack = _coerce_measurement_covariances(
        meas_noise, len(measurement_vectors)
    )

    hypotheses = []
    for track_index, track in enumerate(tracks):
        state = _track_filter_state(track)
        for measurement_index, measurement in enumerate(measurement_vectors):
            covariance = covariance_stack[measurement_index]
            innovation, innovation_covariance = linear_gaussian_innovation(
                state.mu,
                state.C,
                measurement,
                measurement_matrix,
                covariance,
            )
            nis = float(
                normalized_innovation_squared(innovation, innovation_covariance)
            )
            measurement_dim = int(measurement_matrix.shape[0])
            log_likelihood = _linear_gaussian_log_likelihood(
                innovation,
                innovation_covariance,
                measurement_dim,
            )
            metadata = None
            if metadata_builder is not None:
                metadata = metadata_builder(
                    track=track,
                    track_index=track_index,
                    measurement=measurement,
                    measurement_index=measurement_index,
                    innovation=innovation,
                    innovation_covariance=innovation_covariance,
                    normalized_innovation_squared=nis,
                    log_likelihood=log_likelihood,
                )
            hypotheses.append(
                AssociationHypothesis(
                    track_index=track_index,
                    measurement_index=measurement_index,
                    cost=nis,
                    log_likelihood=log_likelihood,
                    innovation=innovation,
                    innovation_covariance=innovation_covariance,
                    normalized_innovation_squared=nis,
                    accepted=True,
                    metadata=metadata,
                )
            )

    if gates is not None:
        hypotheses = filter_hypotheses(
            hypotheses, gates, accepted_only=not include_rejected
        )
    elif not include_rejected:
        hypotheses = [hypothesis for hypothesis in hypotheses if hypothesis.accepted]
    return hypotheses


def hypotheses_to_cost_matrix(
    hypotheses: Sequence[AssociationHypothesis],
    num_tracks: int | None = None,
    num_measurements: int | None = None,
    *,
    missing_cost: float = np.inf,
    rejected_cost: float | None = None,
    include_rejected: bool = False,
) -> np.ndarray:
    """Convert hypotheses to a dense assignment cost matrix."""
    num_tracks, num_measurements = infer_hypothesis_shape(
        hypotheses,
        num_tracks=num_tracks,
        num_measurements=num_measurements,
    )
    matrix = np.full((num_tracks, num_measurements), float(missing_cost))
    if rejected_cost is None:
        rejected_cost = missing_cost
    for hypothesis in hypotheses:
        if hypothesis.is_missed_detection:
            continue
        if not hypothesis.accepted and not include_rejected:
            continue
        cost = hypothesis_cost(hypothesis, missing_cost=missing_cost)
        if not hypothesis.accepted:
            cost = float(rejected_cost)
        matrix[hypothesis.track_index, _measurement_index(hypothesis)] = cost
    return matrix


def hypotheses_to_log_likelihood_matrix(
    hypotheses: Sequence[AssociationHypothesis],
    num_tracks: int | None = None,
    num_measurements: int | None = None,
    *,
    missing_value: float = -np.inf,
    include_rejected: bool = False,
) -> np.ndarray:
    """Convert hypotheses to a dense log-likelihood matrix."""
    num_tracks, num_measurements = infer_hypothesis_shape(
        hypotheses,
        num_tracks=num_tracks,
        num_measurements=num_measurements,
    )
    matrix = np.full((num_tracks, num_measurements), float(missing_value))
    for hypothesis in hypotheses:
        if hypothesis.is_missed_detection:
            continue
        if not hypothesis.accepted and not include_rejected:
            continue
        if hypothesis.log_likelihood is not None:
            value = float(hypothesis.log_likelihood)
        elif hypothesis.probability is not None and float(hypothesis.probability) > 0.0:
            value = log(float(hypothesis.probability))
        else:
            value = -hypothesis_cost(hypothesis)
        matrix[hypothesis.track_index, _measurement_index(hypothesis)] = value
    return matrix


def hypotheses_to_probability_matrix(
    hypotheses: Sequence[AssociationHypothesis],
    num_tracks: int | None = None,
    num_measurements: int | None = None,
    *,
    missing_value: float = 0.0,
    include_rejected: bool = False,
) -> np.ndarray:
    """Convert hypotheses to a dense probability-like matrix."""
    num_tracks, num_measurements = infer_hypothesis_shape(
        hypotheses,
        num_tracks=num_tracks,
        num_measurements=num_measurements,
    )
    matrix = np.full((num_tracks, num_measurements), float(missing_value))
    for hypothesis in hypotheses:
        if hypothesis.is_missed_detection:
            continue
        if not hypothesis.accepted and not include_rejected:
            continue
        if hypothesis.probability is not None:
            value = float(hypothesis.probability)
        elif hypothesis.log_likelihood is not None:
            value = float(np.exp(hypothesis.log_likelihood))
        else:
            value = float(np.exp(-hypothesis_cost(hypothesis)))
        matrix[hypothesis.track_index, _measurement_index(hypothesis)] = value
    return matrix


def association_result_from_hypotheses(
    hypotheses: Sequence[AssociationHypothesis],
    *,
    num_tracks: int | None = None,
    num_measurements: int | None = None,
    missing_cost: float = np.inf,
    unassigned_track_cost: float | Sequence[float] = np.inf,
    unassigned_measurement_cost: float | Sequence[float] | None = None,
):
    """Solve GNN assignment from hypotheses and return ``AssociationResult``."""
    from .track_manager import (  # pylint: disable=import-outside-toplevel
        solve_global_nearest_neighbor,
    )

    cost_matrix = hypotheses_to_cost_matrix(
        hypotheses,
        num_tracks=num_tracks,
        num_measurements=num_measurements,
        missing_cost=missing_cost,
    )
    return solve_global_nearest_neighbor(
        cost_matrix,
        unassigned_track_cost=unassigned_track_cost,
        unassigned_measurement_cost=unassigned_measurement_cost,
    )


def build_linear_gaussian_hypothesis_associator(
    measurement_matrix,
    meas_noise,
    *,
    gates=None,
    missing_cost: float = np.inf,
    unassigned_track_cost: float | Sequence[float] = np.inf,
    unassigned_measurement_cost: float | Sequence[float] | None = None,
    measurement_axis: MeasurementAxis = "auto",
):
    """Create a TrackManager-compatible linear-Gaussian associator."""

    def associator(tracks, measurements, **kwargs):
        hypotheses = linear_gaussian_association_hypotheses(
            tracks,
            measurements,
            kwargs.get("measurement_matrix", measurement_matrix),
            kwargs.get("meas_noise", meas_noise),
            gates=kwargs.get("gates", gates),
            measurement_axis=kwargs.get("measurement_axis", measurement_axis),
        )
        return association_result_from_hypotheses(
            hypotheses,
            num_tracks=len(tracks),
            num_measurements=len(
                _coerce_measurements(
                    measurements,
                    measurement_axis=kwargs.get("measurement_axis", measurement_axis),
                )
            ),
            missing_cost=kwargs.get("missing_cost", missing_cost),
            unassigned_track_cost=kwargs.get(
                "unassigned_track_cost", unassigned_track_cost
            ),
            unassigned_measurement_cost=kwargs.get(
                "unassigned_measurement_cost", unassigned_measurement_cost
            ),
        )

    return associator


def infer_hypothesis_shape(
    hypotheses: Sequence[AssociationHypothesis],
    num_tracks: int | None = None,
    num_measurements: int | None = None,
) -> tuple[int, int]:
    """Infer cost-matrix shape from hypotheses unless explicit sizes are given."""
    if num_tracks is None:
        num_tracks = 0
        if hypotheses:
            num_tracks = max(hypothesis.track_index for hypothesis in hypotheses) + 1
    if num_measurements is None:
        measurement_indices = [
            _measurement_index(hypothesis)
            for hypothesis in hypotheses
            if hypothesis.measurement_index is not None
        ]
        num_measurements = max(measurement_indices) + 1 if measurement_indices else 0
    if int(num_tracks) < 0 or int(num_measurements) < 0:
        raise ValueError("num_tracks and num_measurements must be nonnegative")
    return int(num_tracks), int(num_measurements)


def _track_filter_state(track):
    if hasattr(track, "filter_state"):
        return track.filter_state
    if hasattr(track, "single_target_filter"):
        return track.single_target_filter.filter_state
    raise TypeError(
        "track must expose filter_state or single_target_filter.filter_state"
    )


def _coerce_measurements(
    measurements, *, measurement_axis: MeasurementAxis = "auto"
) -> list[np.ndarray]:
    if isinstance(measurements, np.ndarray):
        return _coerce_measurement_array(
            measurements, measurement_axis=measurement_axis
        )
    try:
        from pyrecest.backend import to_numpy  # pylint: disable=import-outside-toplevel

        maybe_array = to_numpy(measurements)
        if isinstance(maybe_array, np.ndarray):
            return _coerce_measurement_array(
                maybe_array, measurement_axis=measurement_axis
            )
    except (ImportError, AttributeError, TypeError):
        pass
    return [
        np.asarray(measurement, dtype=float).reshape(-1) for measurement in measurements
    ]


def _coerce_measurement_array(
    array, *, measurement_axis: MeasurementAxis
) -> list[np.ndarray]:
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return [array.reshape(-1)]
    if array.ndim != 2:
        raise ValueError("measurements must be a sequence, vector, or 2D array")
    if measurement_axis == "columns":
        return [array[:, index].reshape(-1) for index in range(array.shape[1])]
    if measurement_axis == "rows":
        return [array[index, :].reshape(-1) for index in range(array.shape[0])]
    if measurement_axis == "sequence":
        return [array[index].reshape(-1) for index in range(array.shape[0])]
    if measurement_axis != "auto":
        raise ValueError(
            "measurement_axis must be 'auto', 'columns', 'rows', or 'sequence'"
        )
    if array.shape[0] <= array.shape[1]:
        return [array[:, index].reshape(-1) for index in range(array.shape[1])]
    return [array[index, :].reshape(-1) for index in range(array.shape[0])]


def _coerce_measurement_covariances(
    meas_noise, num_measurements: int
) -> list[np.ndarray]:
    covariance = np.asarray(meas_noise, dtype=float)
    if covariance.ndim == 2:
        return [covariance for _ in range(num_measurements)]
    if covariance.ndim == 3:
        if covariance.shape[2] == num_measurements:
            return [covariance[:, :, index] for index in range(num_measurements)]
        if covariance.shape[0] == num_measurements:
            return [covariance[index, :, :] for index in range(num_measurements)]
    raise ValueError("meas_noise must have shape (m, m), (m, m, n), or (n, m, m)")


def _linear_gaussian_log_likelihood(
    innovation, innovation_covariance, measurement_dim: int
) -> float:
    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    innovation_covariance = np.asarray(innovation_covariance, dtype=float)
    sign, logdet = np.linalg.slogdet(innovation_covariance)
    if sign <= 0.0:
        return float("-inf")
    nis = float(normalized_innovation_squared(innovation, innovation_covariance))
    return -0.5 * (measurement_dim * log(2.0 * pi) + logdet + nis)


__all__ = [
    "AssociationHypothesis",
    "CostThresholdGate",
    "NISGate",
    "ProbabilityThresholdGate",
    "TopKGate",
    "association_result_from_hypotheses",
    "build_linear_gaussian_hypothesis_associator",
    "filter_hypotheses",
    "gate_hypotheses",
    "hypotheses_to_cost_matrix",
    "hypotheses_to_log_likelihood_matrix",
    "hypotheses_to_probability_matrix",
    "hypothesis_cost",
    "infer_hypothesis_shape",
    "linear_gaussian_association_hypotheses",
    "missed_detection_hypothesis",
]
