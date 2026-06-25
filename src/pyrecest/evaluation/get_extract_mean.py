from __future__ import annotations

from collections.abc import Callable
from typing import Any

MeanExtractorFactory = Callable[[str, bool], Callable[[Any], Any]]
_EXTRACT_MEAN_FACTORIES: dict[str, MeanExtractorFactory] = {}


def _normalize_registry_name(manifold_name: str) -> str:
    if not isinstance(manifold_name, str) or not manifold_name.strip():
        raise ValueError("manifold_name must be a non-empty string")
    return manifold_name.strip().lower()


def register_extract_mean(
    manifold_name: str, factory: MeanExtractorFactory
) -> MeanExtractorFactory:
    """Register a custom mean-extraction factory for a manifold name."""
    normalized_name = _normalize_registry_name(manifold_name)
    if not callable(factory):
        raise TypeError("factory must be callable")
    _EXTRACT_MEAN_FACTORIES[normalized_name] = factory
    return factory


def available_extract_mean_functions() -> tuple[str, ...]:
    return tuple(sorted(_EXTRACT_MEAN_FACTORIES))


def _is_hypersphere_symmetric_name(normalized_name: str) -> bool:
    return (
        "hyperspheresymmetric" in normalized_name
        or "hypersphere_symmetric" in normalized_name
    )


def _is_array_state(filter_state) -> bool:
    return hasattr(filter_state, "ndim") and hasattr(filter_state, "shape")


def _point_estimate_or_mean(filter_state):
    if hasattr(filter_state, "get_point_estimate"):
        return filter_state.get_point_estimate()
    if hasattr(filter_state, "mu"):
        return filter_state.mu
    if _is_array_state(filter_state):
        return filter_state
    if hasattr(filter_state, "mean"):
        mean = filter_state.mean
        return mean() if callable(mean) else mean
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
    normalized_name = _normalize_registry_name(manifold_name)
    registered_factory = _EXTRACT_MEAN_FACTORIES.get(normalized_name)
    if registered_factory is not None:
        return registered_factory(manifold_name, mtt_scenario)

    if "circle" in normalized_name or "hypertorus" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif _is_hypersphere_symmetric_name(normalized_name):
        raise NotImplementedError(
            "Symmetric hypersphere mean extraction needs an explicit convention via a custom extractor."
        )

    elif "hypersphere" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "symm" in normalized_name:
        raise NotImplementedError(
            "Symmetric mean extraction needs an explicit convention"
        )

    elif "se2bounded" in normalized_name:
        raise NotImplementedError("Not implemented yet")

    elif "se2" in normalized_name or "se2linear" in normalized_name:
        raise NotImplementedError("Not implemented yet")

    elif "se3bounded" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "se3" in normalized_name or "se3linear" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "euclidean" in normalized_name and not mtt_scenario:

        def extract_mean(filter_state):
            return _point_estimate_or_mean(filter_state)

    elif "euclidean" in normalized_name and mtt_scenario:

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
