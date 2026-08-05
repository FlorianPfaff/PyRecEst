"""Compatibility fix for Gaussian marginal dimension inputs."""

from __future__ import annotations

from numbers import Integral

import numpy as np


_DIMENSION_ERROR = "dimensions must contain valid zero-based integer indices"


def _normalize_marginal_dimensions(dimensions):
    """Return validation candidates and an iterable accepted by the core method."""

    if np.ma.isMaskedArray(dimensions) and bool(np.ma.getmaskarray(dimensions).any()):
        raise ValueError(f"{_DIMENSION_ERROR}; got {dimensions}.")

    if isinstance(dimensions, (np.datetime64, np.timedelta64, Integral)):
        return [dimensions], dimensions

    if getattr(dimensions, "ndim", None) == 0:
        try:
            scalar = dimensions.item()
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"{_DIMENSION_ERROR}; got {dimensions}.") from exc
        # Keep the original scalar-array candidate so temporal dtype information
        # is not lost when ``item()`` converts datetime64/timedelta64 values.
        return [dimensions, scalar], [scalar]

    try:
        normalized_dimensions = list(dimensions)
    except TypeError as exc:
        raise ValueError(f"{_DIMENSION_ERROR}; got {dimensions}.") from exc
    return normalized_dimensions, normalized_dimensions


def patch_gaussian_marginal_temporal_indices() -> None:
    """Normalize scalar dimension arrays and reject temporal indices."""

    from .gaussian_distribution import GaussianDistribution

    original = GaussianDistribution.marginalize_out
    if getattr(original, "_rejects_temporal_indices", False):
        return

    def marginalize_out(self, dimensions):
        candidates, normalized_dimensions = _normalize_marginal_dimensions(dimensions)

        if any(
            isinstance(dim, (np.datetime64, np.timedelta64))
            or getattr(getattr(dim, "dtype", None), "kind", None) in {"M", "m"}
            for dim in candidates
        ):
            raise ValueError(f"{_DIMENSION_ERROR}; got {dimensions}.")
        return original(self, normalized_dimensions)

    marginalize_out._rejects_temporal_indices = True
    GaussianDistribution.marginalize_out = marginalize_out
