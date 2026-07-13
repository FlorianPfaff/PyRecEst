"""Synthetic event-count models for DVS active-contour experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_contour import activity_profile, rectangle_contour_samples

EDGE_ORDER = ("left", "right", "top", "bottom")
_INVALID_REAL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
)


@dataclass(frozen=True)
class RectangleCountSimulation:
    """Synthetic rectangle event counts and model probabilities."""

    velocity: np.ndarray
    observed_counts: dict[str, int]
    true_probabilities: dict[str, float]
    normal_flow_probabilities: dict[str, float]
    uniform_probabilities: dict[str, float]


def summarize_edge_counts(
    edge_labels: list[str], point_counts: np.ndarray
) -> dict[str, int]:
    """Aggregate per-contour-sample counts by edge label."""
    labels = np.array(edge_labels)
    return {edge: int(np.sum(point_counts[labels == edge])) for edge in EDGE_ORDER}


def _edge_probabilities(
    edge_labels: list[str], point_weights: np.ndarray
) -> dict[str, float]:
    labels = np.array(edge_labels)
    weights = np.asarray(point_weights, dtype=float)
    if weights.shape != labels.shape:
        raise ValueError("point_weights must contain one value per edge label")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("point_weights must contain only finite non-negative values")

    max_weight = float(np.max(weights)) if weights.size else 0.0
    if max_weight <= 0.0:
        edge_weights = np.ones(len(EDGE_ORDER), dtype=float)
    else:
        scaled_weights = weights / max_weight
        edge_weights = np.array(
            [float(np.sum(scaled_weights[labels == edge])) for edge in EDGE_ORDER],
            dtype=float,
        )
        total_weight = float(np.sum(edge_weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("point_weights must have positive finite total weight")
        edge_weights /= total_weight

    return {
        edge: float(weight)
        for edge, weight in zip(EDGE_ORDER, edge_weights, strict=True)
    }


def edge_probabilities_from_activity(
    edge_labels: list[str],
    activities: np.ndarray,
    background_activity: float = 1e-3,
) -> dict[str, float]:
    """Convert normal-flow activities into edge-level event probabilities."""
    background_activity = _nonnegative_real_scalar(
        background_activity,
        "background_activity",
    )
    if background_activity < 0.0:
        raise ValueError("background_activity must be non-negative")
    weights = np.asarray(activities, dtype=float) + float(background_activity)
    return _edge_probabilities(edge_labels, weights)


def uniform_edge_probabilities(edge_labels: list[str]) -> dict[str, float]:
    """Return edge probabilities under a motion-blind uniform contour model."""
    return _edge_probabilities(edge_labels, np.ones(len(edge_labels), dtype=float))


def count_negative_log_likelihood(
    observed_counts: dict[str, int],
    probabilities: dict[str, float],
    probability_floor: float = 1e-12,
) -> float:
    """Return multinomial count NLL up to the count-dependent constant."""
    probability_floor = _validate_probability_floor(probability_floor)
    nll = 0.0
    for edge in EDGE_ORDER:
        count = _edge_count(observed_counts, edge)
        probability = _edge_probability(probabilities, edge, probability_floor)
        nll -= count * float(np.log(probability))
    return nll


def simulate_rectangle_event_counts(
    velocity: np.ndarray,
    total_events: int = 240,
    width: float = 2.0,
    height: float = 1.0,
    samples_per_edge: int = 80,
    background_activity: float = 1e-3,
    seed: int | None = 0,
) -> RectangleCountSimulation:
    """Sample event counts from a motion-gated rectangle contour model."""
    total_events = _positive_integer_count(total_events, "total_events")
    if total_events <= 0:
        raise ValueError("total_events must be positive")
    background_activity = _nonnegative_real_scalar(
        background_activity,
        "background_activity",
    )
    if background_activity < 0.0:
        raise ValueError("background_activity must be non-negative")

    contour = rectangle_contour_samples(
        width=width,
        height=height,
        samples_per_edge=samples_per_edge,
    )
    activities = activity_profile(contour.normals, velocity)
    point_weights = activities + float(background_activity)
    point_weight_sum = float(np.sum(point_weights))
    if not np.isfinite(point_weight_sum) or point_weight_sum <= 0.0:
        raise ValueError("event-generation weights must have positive finite sum")
    point_probabilities = point_weights / point_weight_sum
    rng = np.random.default_rng(seed)
    point_counts = rng.multinomial(total_events, point_probabilities)

    true_probabilities = edge_probabilities_from_activity(
        contour.edge_labels,
        activities,
        background_activity=background_activity,
    )
    return RectangleCountSimulation(
        velocity=np.asarray(velocity, dtype=float),
        observed_counts=summarize_edge_counts(contour.edge_labels, point_counts),
        true_probabilities=true_probabilities,
        normal_flow_probabilities=true_probabilities,
        uniform_probabilities=uniform_edge_probabilities(contour.edge_labels),
    )


def _mapping_value(mapping: dict[str, float], edge: str, name: str):
    try:
        return mapping[edge]
    except KeyError as exc:
        raise ValueError(f"{name} must include an entry for edge {edge!r}") from exc


def _as_finite_real_scalar(value, name: str) -> float:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype.kind in "bcSU":
        raise ValueError(f"{name} must be a finite real scalar")
    scalar = value_array.item()
    if isinstance(scalar, _INVALID_REAL_SCALAR_TYPES):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be a finite real scalar")
    return value_float


def _validate_probability_floor(probability_floor: float) -> float:
    value = _as_finite_real_scalar(probability_floor, "probability_floor")
    if value <= 0.0 or value > 1.0:
        raise ValueError("probability_floor must be finite and in (0, 1]")
    return value


def _nonnegative_real_scalar(value, name: str) -> float:
    scalar = _as_finite_real_scalar(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _positive_integer_count(value, name: str) -> int:
    scalar = _as_finite_real_scalar(value, name)
    if scalar <= 0.0 or not scalar.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(scalar)


def _edge_count(observed_counts: dict[str, int], edge: str) -> int:
    value = _as_finite_real_scalar(
        _mapping_value(observed_counts, edge, "observed_counts"),
        "observed_counts values",
    )
    if value < 0.0 or not value.is_integer():
        raise ValueError("observed_counts values must be non-negative integers")
    return int(value)


def _edge_probability(
    probabilities: dict[str, float], edge: str, probability_floor: float
) -> float:
    value = _as_finite_real_scalar(
        _mapping_value(probabilities, edge, "probabilities"),
        "probabilities values",
    )
    if value < 0.0 or value > 1.0:
        raise ValueError("probabilities values must lie in [0, 1]")
    return max(value, probability_floor)
