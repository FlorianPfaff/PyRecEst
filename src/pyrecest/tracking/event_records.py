"""Asynchronous tracking events and replay-record containers.

The classes in this module are deliberately filter-independent.  They provide a
small, serializable schema for multi-sensor replay pipelines that need to keep
accepted updates, rejected measurements, and predict-only/coast events in one
chronological record table.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

TrackingAction = Literal["predict", "update", "reject", "coast"] | str


def _array_or_none(
    value: Any, *, name: str, ndim: int | None = None
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one element")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _square_matrix(value: Any, *, name: str, dim: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if dim is not None and matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix.copy()


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or None")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class TrackingEvent:
    """One asynchronous sensor event or predict-only replay event."""

    time: float
    source: str
    action: TrackingAction = "update"
    measurement: np.ndarray | None = None
    measurement_model: Any | None = None
    covariance: np.ndarray | None = None
    accepted: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = float(self.time)
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        measurement = _array_or_none(self.measurement, name="measurement", ndim=1)
        covariance = (
            None
            if self.covariance is None
            else _square_matrix(self.covariance, name="covariance")
        )
        if measurement is not None and covariance is not None:
            if covariance.shape != (measurement.size, measurement.size):
                raise ValueError("covariance must match measurement dimension")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(
            self, "accepted", _optional_bool(self.accepted, name="accepted")
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def measurement_dim(self) -> int | None:
        return None if self.measurement is None else int(self.measurement.size)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly shallow dictionary."""

        return {
            "time": self.time,
            "source": self.source,
            "action": self.action,
            "measurement": _jsonable(self.measurement),
            "covariance": _jsonable(self.covariance),
            "accepted": self.accepted,
            "measurement_dim": self.measurement_dim,
            "metadata": _jsonable(dict(self.metadata)),
        }


@dataclass(frozen=True)
class TrackingRecord:
    """Prior/posterior record for one processed tracking event."""

    time: float
    source: str
    action: str
    prior_mean: np.ndarray
    prior_cov: np.ndarray
    posterior_mean: np.ndarray
    posterior_cov: np.ndarray
    innovation: np.ndarray | None = None
    innovation_cov: np.ndarray | None = None
    nis: float | None = None
    accepted: bool | None = None
    measurement: np.ndarray | None = None
    event_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = float(self.time)
        if not np.isfinite(time):
            raise ValueError("time must be finite")
        prior_mean = _vector(self.prior_mean, name="prior_mean")
        posterior_mean = _vector(self.posterior_mean, name="posterior_mean")
        if posterior_mean.shape != prior_mean.shape:
            raise ValueError("posterior_mean must match prior_mean shape")
        prior_cov = _square_matrix(
            self.prior_cov, name="prior_cov", dim=prior_mean.size
        )
        posterior_cov = _square_matrix(
            self.posterior_cov, name="posterior_cov", dim=prior_mean.size
        )
        innovation = _array_or_none(self.innovation, name="innovation", ndim=1)
        innovation_cov = (
            None
            if self.innovation_cov is None
            else _square_matrix(self.innovation_cov, name="innovation_cov")
        )
        if innovation is not None and innovation_cov is not None:
            if innovation_cov.shape != (innovation.size, innovation.size):
                raise ValueError("innovation_cov must match innovation dimension")
        measurement = _array_or_none(self.measurement, name="measurement", ndim=1)
        nis = None if self.nis is None else float(self.nis)
        if nis is not None and (not np.isfinite(nis) or nis < 0.0):
            raise ValueError("nis must be finite and nonnegative")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "prior_mean", prior_mean)
        object.__setattr__(self, "prior_cov", prior_cov)
        object.__setattr__(self, "posterior_mean", posterior_mean)
        object.__setattr__(self, "posterior_cov", posterior_cov)
        object.__setattr__(self, "innovation", innovation)
        object.__setattr__(self, "innovation_cov", innovation_cov)
        object.__setattr__(self, "nis", nis)
        object.__setattr__(
            self, "accepted", _optional_bool(self.accepted, name="accepted")
        )
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "event_metadata", dict(self.event_metadata))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def state_dim(self) -> int:
        return int(self.posterior_mean.size)

    def to_dict(self, *, include_legacy_aliases: bool = False) -> dict[str, Any]:
        """Return a JSON/CSV-friendly dictionary.

        ``include_legacy_aliases`` adds ``time_s``, ``state``, and ``covariance``
        keys used by existing record-table code in many trackers.
        """

        result = {
            "time": self.time,
            "source": self.source,
            "action": self.action,
            "accepted": self.accepted,
            "prior_mean": _jsonable(self.prior_mean),
            "prior_cov": _jsonable(self.prior_cov),
            "posterior_mean": _jsonable(self.posterior_mean),
            "posterior_cov": _jsonable(self.posterior_cov),
            "innovation": _jsonable(self.innovation),
            "innovation_cov": _jsonable(self.innovation_cov),
            "nis": self.nis,
            "measurement": _jsonable(self.measurement),
            "event_metadata": _jsonable(dict(self.event_metadata)),
            "metadata": _jsonable(dict(self.metadata)),
        }
        if include_legacy_aliases:
            result.update(
                {
                    "time_s": self.time,
                    "state": _jsonable(self.posterior_mean),
                    "covariance": _jsonable(self.posterior_cov),
                }
            )
        return result


def event_from_measurement(
    *,
    time: float,
    source: str,
    measurement: Any | None = None,
    measurement_model: Any | None = None,
    covariance: Any | None = None,
    accepted: bool | None = None,
    action: TrackingAction = "update",
    metadata: Mapping[str, Any] | None = None,
) -> TrackingEvent:
    """Construct a :class:`TrackingEvent` from generic measurement fields."""

    return TrackingEvent(
        time=time,
        source=source,
        action=action,
        measurement=measurement,
        measurement_model=measurement_model,
        covariance=covariance,
        accepted=accepted,
        metadata={} if metadata is None else dict(metadata),
    )


def record_from_update(
    *,
    event: TrackingEvent,
    prior_mean: Any,
    prior_cov: Any,
    posterior_mean: Any,
    posterior_cov: Any,
    innovation: Any | None = None,
    innovation_cov: Any | None = None,
    nis: float | None = None,
    accepted: bool | None = None,
    action: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrackingRecord:
    """Construct a tracking record from one event and prior/posterior states."""

    resolved_accepted = event.accepted if accepted is None else accepted
    return TrackingRecord(
        time=event.time,
        source=event.source,
        action=event.action if action is None else action,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        posterior_mean=posterior_mean,
        posterior_cov=posterior_cov,
        innovation=innovation,
        innovation_cov=innovation_cov,
        nis=nis,
        accepted=resolved_accepted,
        measurement=event.measurement,
        event_metadata=event.metadata,
        metadata={} if metadata is None else dict(metadata),
    )


def records_to_dicts(
    records: Iterable[TrackingRecord], *, include_legacy_aliases: bool = False
) -> list[dict[str, Any]]:
    """Convert tracking records to dictionaries."""

    return [
        record.to_dict(include_legacy_aliases=include_legacy_aliases)
        for record in records
    ]


def records_to_matrix(
    records: Iterable[TrackingRecord], *, field: str = "posterior_mean"
) -> np.ndarray:
    """Stack a vector-valued field from records into a matrix."""

    arrays = [
        np.asarray(getattr(record, field), dtype=float).reshape(-1)
        for record in records
    ]
    if not arrays:
        return np.empty((0, 0), dtype=float)
    return np.stack(arrays)


def action_counts(
    records: Iterable[TrackingRecord | Mapping[str, Any]],
) -> dict[str, int]:
    """Count tracking records by action label."""

    counter: Counter[str] = Counter()
    for record in records:
        if isinstance(record, TrackingRecord):
            action = record.action
        else:
            action = str(record.get("action", record.get("update_action", "unknown")))
        counter[str(action)] += 1
    return dict(counter)
