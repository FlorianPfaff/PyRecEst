"""Hypertoroidal distribution exports."""

from . import low_rank_hypertoroidal_fourier_distribution as _low_rank
from .low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)

_low_rank_as_shape = _low_rank._as_shape


def _as_odd_coefficient_shape(shape, *, dim=None):
    return _low_rank_as_shape(shape, dim=dim)


_low_rank._as_shape = _as_odd_coefficient_shape

__all__ = ("LowRankHypertoroidalFourierDistribution",)
