from __future__ import annotations

from collections.abc import Callable
from typing import Any

MeanExtractorFactory = Callable[[str, bool], Callable[[Any], Any]]
_EXTRACT_MEAN_FACTORIES: dict[str, MeanExtractorFactory] = {}


def register_extract_mean(
    manifold_name: str, factory: MeanExtractorFactory
) -> MeanExtractorFactory:
    """Register a custom mean-extraction factory for a manifold name."""
    if not manifold_name:
        raise ValueError("manifold_name must be a non-empty string")
    _EXTRACT_MEAN_FACTORIES[manifold_name.lower()] = factory
    return factory


def available_extract_mean_functions() -> tuple[str, ...]:
    return tuple(sorted(_EXTRACT_MEAN_FACTORIES))


def _point_estimate_or_mean(filter_state):
    if hasattr(filter_state, "get_point_estimate"):
        return filter_state.get_point_estimate()
    if hasattr(filter_state, "mean"):
        return filter_state.mean()
    if hasattr(filter_state, "mu"):
        return filter_state.mu
    return filter_state


def _extract_mtt_mean(filter_state):
    if hasattr(filter_state, "tracks"):
        return [_point_estimate_or_mean(track) for track in filter_state.tracks]
    if hasattr(filter_state, "single_target_filters"):
        return [
            _point_estimate_or_mean(track)
            for track in filter_state.single_target_filters
        ]
    if isinstance(filter_state, list | tuple):
        return [_point_estimate_or_mean(track) for track in filter_state]
    return _point_estimate_or_mean(filter_state)


def get_extract_mean(manifold_name, mtt_scenario=False):
    normalized_name = str(manifold_name).lower()
    registered_factory = _EXTRACT_MEAN_FACTORIES.get(normalized_name)
    if registered_factory is not None:
        return registered_factory(manifold_name, mtt_scenario)

    if "circle" in manifold_name or "hypertorus" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "hypersphereSymmetric" in manifold_name:
        extract_mean = "custom"

    elif "hypersphere" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "symm" in normalized_name:
        raise NotImplementedError("Symmetric mean extraction needs a custom extractor")

    elif "se2" in manifold_name or "se2linear" in manifold_name:
        raise NotImplementedError("Not implemented yet")

    elif "se2bounded" in manifold_name:
        raise NotImplementedError("Not implemented yet")

    elif "se3" in manifold_name or "se3linear" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "se3bounded" in manifold_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and not mtt_scenario:

        def extract_mean(filter_state):
            return _point_estimate_or_mean(filter_state)

    elif (
        "euclidean" in manifold_name or "Euclidean" in manifold_name
    ) and mtt_scenario:

        def extract_mean(filter_state):
            return _extract_mtt_mean(filter_state)

    else:
        raise ValueError("Mode not recognized")

    return extract_mean


__all__ = [
    "available_extract_mean_functions",
    "get_extract_mean",
    "register_extract_mean",
]
