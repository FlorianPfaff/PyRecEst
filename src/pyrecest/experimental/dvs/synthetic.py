"""Synthetic event-count models for DVS active-contour experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_contour import activity_profile, rectangle_contour_samples

EDGE_ORDER = ("left", "right", "top", "bottom")


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
    edge_weights = np.array(
        [float(np.sum(point_weights[labels == edge])) for edge in EDGE_ORDER],
        dtype=float,
    )
    total_weight = float(np.sum(edge_weights))
    if total_weight <= 0.0:
        edge_weights = np.ones(len(EDGE_ORDER), dtype=float)
        total_weight = float(len(EDGE_ORDER))
    return {
        edge: float(weight / total_weight)
        for edge, weight in zip(EDGE_ORDER, edge_weights, strict=True)
    }


def edge_probabilities_from_activity(
    edge_labels: list[str],
    activities: np.ndarray,
    background_activity: float = 1e-3,
) -> dict[str, float]:
    """Convert normal-flow activities into edge-level event probabilities."""
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
    if probability_floor <= 0.0:
        raise ValueError("probability_floor must be positive")
    nll = 0.0
    for edge in EDGE_ORDER:
        probability = max(float(probabilities[edge]), probability_floor)
        nll -= int(observed_counts[edge]) * float(np.log(probability))
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
    if total_events <= 0:
        raise ValueError("total_events must be positive")
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
