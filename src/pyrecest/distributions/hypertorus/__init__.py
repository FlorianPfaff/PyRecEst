"""Hypertoroidal distribution exports."""

from numbers import Integral

import numpy as np

from . import low_rank_hypertoroidal_fourier_distribution as _low_rank
from .low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)


def _as_odd_coefficient_shape(shape):
    if isinstance(shape, (bool, np.bool_)):
        raise ValueError("Fourier coefficient side lengths must be positive and odd.")
    if isinstance(shape, Integral):
        normalized = (int(shape),)
    else:
        try:
            normalized = tuple(int(n) for n in shape)
        except TypeError as exc:
            raise TypeError(
                "shape must be an integer or a sequence of integers."
            ) from exc
    if not normalized:
        raise ValueError("shape must contain at least one dimension.")
    if any(n < 1 or n % 2 != 1 for n in normalized):
        raise ValueError("Fourier coefficient side lengths must be positive and odd.")
    return normalized


_low_rank._as_shape = _as_odd_coefficient_shape

__all__ = ("LowRankHypertoroidalFourierDistribution",)
