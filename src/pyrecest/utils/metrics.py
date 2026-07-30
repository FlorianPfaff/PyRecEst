"""Estimator, finite-set, and extended-object performance metrics."""

from __future__ import annotations

import numpy as np

from . import _metrics_impl as _impl

ArrayLike = _impl.ArrayLike
DistanceFunction = _impl.DistanceFunction


def _pairwise_distances(
    estimated: np.ndarray,
    reference: np.ndarray,
    distance_fn: DistanceFunction | None,
) -> np.ndarray:
    """Return pairwise distances without overflowing representable Euclidean norms."""
    if distance_fn is None:
        differences = estimated[:, None, :] - reference[None, :, :]
        scales = np.max(np.abs(differences), axis=-1)
        distances = scales.copy()
        finite_nonzero = np.isfinite(scales) & (scales > 0.0)
        if np.any(finite_nonzero):
            normalized = np.zeros_like(differences)
            np.divide(
                differences,
                scales[..., None],
                out=normalized,
                where=finite_nonzero[..., None],
            )
            normalized_norms = np.linalg.norm(normalized, axis=-1)
            distances[finite_nonzero] = (
                scales[finite_nonzero] * normalized_norms[finite_nonzero]
            )
        distances[scales == 0.0] = 0.0
        return distances

    distances = np.empty((estimated.shape[0], reference.shape[0]), dtype=float)
    for row, estimate in enumerate(estimated):
        for col, truth in enumerate(reference):
            distances[row, col] = float(distance_fn(estimate, truth))
    return distances


_impl._pairwise_distances = _pairwise_distances
__all__ = _impl.__all__
for _name in __all__:
    globals()[_name] = getattr(_impl, _name)
