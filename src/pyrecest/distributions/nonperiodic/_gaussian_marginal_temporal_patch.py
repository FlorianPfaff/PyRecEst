"""Compatibility fix for temporal Gaussian marginal indices."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def patch_gaussian_marginal_temporal_indices() -> None:
    """Reject NumPy temporal scalars before integer-index coercion."""

    from .gaussian_distribution import GaussianDistribution

    original = GaussianDistribution.marginalize_out
    if getattr(original, "_rejects_temporal_indices", False):
        return

    def marginalize_out(self, dimensions):
        if isinstance(dimensions, (np.datetime64, np.timedelta64)):
            candidates = [dimensions]
            normalized_dimensions = dimensions
        elif isinstance(dimensions, Integral):
            candidates = [dimensions]
            normalized_dimensions = dimensions
        else:
            normalized_dimensions = list(dimensions)
            candidates = normalized_dimensions

        if any(
            isinstance(dim, (np.datetime64, np.timedelta64))
            or getattr(getattr(dim, "dtype", None), "kind", None) in {"M", "m"}
            for dim in candidates
        ):
            raise ValueError(
                "dimensions must contain valid zero-based integer indices; "
                f"got {dimensions}."
            )
        return original(self, normalized_dimensions)

    marginalize_out._rejects_temporal_indices = True
    GaussianDistribution.marginalize_out = marginalize_out
