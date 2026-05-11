"""Backward-compatible name for the SO(3) product block particle filter."""

from .so3_product_block_particle_filter import SO3ProductBlockParticleFilter


class PartitionedSO3ProductParticleFilter(SO3ProductBlockParticleFilter):
    """Deprecated alias for :class:`SO3ProductBlockParticleFilter`."""
