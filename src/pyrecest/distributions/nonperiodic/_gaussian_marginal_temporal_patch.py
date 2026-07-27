"""Compatibility fix for temporal Gaussian marginal indices."""

from __future__ import annotations

import numpy as np


def patch_gaussian_marginal_temporal_indices() -> None:
    """Reject NumPy temporal scalars before integer-index coercion."""

    from .gaussian_distribution import GaussianDistribution

    original = GaussianDistribution.marginalize_out
    if getattr(original, "_rejects_temporal_indices", False):
        return

    def marginalize_out(self, dimensions):
        candidates = [dimensions] if np.ndim(dimensions) == 0 else dimensions
        try:
            temporal = any(np.asarray(dim).dtype.kind in {"M", "m"} for dim in candidates)
        except (TypeError, ValueError):
            temporal = False
        if temporal:
            raise ValueError(
                "dimensions must contain valid zero-based integer indices; "
                f"got {dimensions}."
            )
        return original(self, dimensions)

    marginalize_out._rejects_temporal_indices = True
    GaussianDistribution.marginalize_out = marginalize_out
