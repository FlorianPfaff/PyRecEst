from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

MeanExtractorFactory = Callable[[str, bool], Callable[[Any], Any]]
_EXTRACT_MEAN_FACTORIES: dict[str, MeanExtractorFactory] = {}


def _normalize_registry_name(manifold_name: str) -> str:
    if not isinstance(manifold_name, str) or not manifold_name.strip():
        raise ValueError("manifold_name must be a non-empty string")
    return manifold_name.strip().lower()


def _coerce_mtt_scenario_flag(value: Any) -> bool:
    if np.ma.isMaskedArray(value) and bool(np.any(np.ma.getmaskarray(value))):
        raise ValueError("mtt_scenario must be a bool")
    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("mtt_scenario must be a bool") from exc
    if value_array.shape == () and np.issubdtype(value_array.dtype, np.bool_):
        return bool(value_array.item())
    raise ValueError("mtt_scenario must be a bool")


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
    tokens = tuple(
        token for token in normalized_name.replace("-", "_").split("_") if token
    )
    if "hypersphere" in tokens and any(
        token in {"symmetric", "symm"} for token in tokens
    ):
        return True

    compact = "".join(tokens)
    return (
        "hyperspheresymmetric" in compact
        or "hyperspheresymm" in compact
        or compact in {"symmetrichypersphere", "symmhypersphere"}
    )


def _is_array_state(filter_state) -> bool:
    return hasattr(filter_state, "ndim") and hasattr(filter_state, "shape")


def _unsupported(message: str) -> None:
    raise NotImplementedError(message)


def _point_estimate_or_mean(filter_state):
    point_estimate = getattr(filter_state, "get_point_estimate", None)
    if point_estimate is not None:
        return point_estimate() if callable(point_estimate) else point_estimate
    if hasattr(filter_state, "mu"):
        return filter_state.mu
    if _is_array_state(filter_state):
        return filter_state
    if hasattr(filter_state, "mean"):
        mean = filter_state.mean
        return mean() if callable(mean) else mean
    return filter_state


def _extract_track_collection_mean(tracks):
    return [_point_estimate_or_mean(track) for track in tracks]


def _extract_mtt_mean(filter_state):
    get_tracks = getattr(filter_state, "get_tracks", None)
    if callable(get_tracks):
        return _extract_track_collection_mean(get_tracks())
    if hasattr(filter_state, "tracks"):
        return _extract_track_collection_mean(filter_state.tracks)
    if hasattr(filter_state, "single_target_filters"):
        return _extract_track_collection_mean(filter_state.single_target_filters)
    if isinstance(filter_state, (list, tuple)):
        return _extract_track_collection_mean(filter_state)
    return _point_estimate_or_mean(filter_state)


def get_extract_mean(manifold_name, mtt_scenario=False):
    normalized_name = _normalize_registry_name(manifold_name)
    is_mtt_scenario = (
        _coerce_mtt_scenario_flag(mtt_scenario) or "mtt" in normalized_name
    )
    registered_factory = _EXTRACT_MEAN_FACTORIES.get(normalized_name)
    if registered_factory is not None:
        return registered_factory(manifold_name, is_mtt_scenario)

    if _is_hypersphere_symmetric_name(normalized_name):
        _unsupported(
            "Symmetric hypersphere mean extraction needs an explicit convention via a custom extractor."
        )
    elif "symm" in normalized_name:
        _unsupported("Symmetric mean extraction needs an explicit convention")
    elif "circle" in normalized_name or "hypertorus" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "hypersphere" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.mean_direction()

    elif "se2bounded" in normalized_name:
        _unsupported("Not implemented yet")

    elif "se2" in normalized_name or "se2linear" in normalized_name:
        _unsupported("Not implemented yet")

    elif "se3bounded" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "se3" in normalized_name or "se3linear" in normalized_name:

        def extract_mean(filter_state):
            return filter_state.hybrid_mean()

    elif "euclidean" in normalized_name and not is_mtt_scenario:

        def extract_mean(filter_state):
            return _point_estimate_or_mean(filter_state)

    elif "euclidean" in normalized_name and is_mtt_scenario:

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
