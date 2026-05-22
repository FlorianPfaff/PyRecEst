"""Experimental event-camera/DVS active-contour tracking utilities.

This namespace contains DVS/event-camera extensions that are useful for
extended-object tracking research but are not yet part of PyRecEst's stable API.
"""

from __future__ import annotations

from .active_contour import (
    RectangleContour,
    activity_profile,
    normal_flow_activity,
    rectangle_contour_samples,
    signed_normal_flow,
    signed_normal_flow_profile,
    unit_vector_from_angle,
)
from .event_likelihood import (
    ContourSample,
    EventLikelihoodConfig,
    EventLikelihoodTerms,
    PointProcessUpdateConfig,
    contour_event_intensity,
    event_batch_log_likelihood,
    event_batch_log_likelihood_terms,
    expected_event_count,
    normal_flow_activities,
)
from .synthetic import (
    EDGE_ORDER,
    RectangleCountSimulation,
    count_negative_log_likelihood,
    edge_probabilities_from_activity,
    simulate_rectangle_event_counts,
    summarize_edge_counts,
    uniform_edge_probabilities,
)

_LAZY_EXPORTS = {
    "DVSFullSCGPTracker": ".trackers",
    "DVSSCGPTracker": ".trackers",
    "DVSPointProcessSCGP": ".point_process_tracker",
    "DVSPointProcessSCGPTracker": ".point_process_tracker",
    "tracker_signed_normal_flows_vectorized": ".vectorized_flow",
}


def __getattr__(name: str):
    """Import tracker-related objects lazily to keep core helpers lightweight."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "ContourSample",
    "DVSFullSCGPTracker",
    "DVSPointProcessSCGP",
    "DVSPointProcessSCGPTracker",
    "DVSSCGPTracker",
    "EDGE_ORDER",
    "EventLikelihoodConfig",
    "EventLikelihoodTerms",
    "PointProcessUpdateConfig",
    "RectangleContour",
    "RectangleCountSimulation",
    "activity_profile",
    "contour_event_intensity",
    "count_negative_log_likelihood",
    "edge_probabilities_from_activity",
    "event_batch_log_likelihood",
    "event_batch_log_likelihood_terms",
    "expected_event_count",
    "normal_flow_activities",
    "normal_flow_activity",
    "rectangle_contour_samples",
    "signed_normal_flow",
    "signed_normal_flow_profile",
    "simulate_rectangle_event_counts",
    "summarize_edge_counts",
    "tracker_signed_normal_flows_vectorized",
    "uniform_edge_probabilities",
    "unit_vector_from_angle",
]
