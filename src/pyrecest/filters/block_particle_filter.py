"""Generic block particle-filter mechanics for product-state particle filters."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol

# pylint: disable=no-name-in-module,no-member,too-many-positional-arguments
from pyrecest.backend import (
    all,
    any,
    array,
    exp,
    isfinite,
    isnan,
    log,
    max,
    ndim,
    random,
    reshape,
    stack,
    sum,
    to_numpy,
)


class _WeightedParticleState(Protocol):
    d: Any
    w: Any


class BlockParticleFilter:
    """Reusable block particle-filter base for product-state particle filters.

    The class implements block weights, block-wise updates, and block-wise
    systematic resampling independently of the particle manifold. It assumes
    particles can be viewed as product states with shape
    ``(n_particles, n_components, ...)``. Subclasses should provide
    ``n_particles`` and ``filter_state.w`` and usually inherit from a concrete
    particle filter that implements ``set_particles``.
    """

    filter_state: _WeightedParticleState
    _n_block_components: int
    partition: tuple[tuple[int, ...], ...]
    _component_to_block: tuple[int, ...]
    _block_weights: Any

    def _initialize_block_particle_filter(
        self,
        n_components: int | None = None,
        partition=None,
        block_weights=None,
        weights=None,
    ) -> None:
        if n_components is None:
            particles = self._block_particle_array()
            if ndim(particles) < 2:
                raise ValueError(
                    "Block particle filters require particles shaped "
                    "(n_particles, n_components, ...)."
                )
            n_components = int(particles.shape[1])
        if int(n_components) <= 0:
            raise ValueError("n_components must be positive.")

        self._n_block_components = int(n_components)
        self.partition = self._validate_partition(partition, self._n_block_components)
        self._component_to_block = self._build_component_to_block(
            self.partition, self._n_block_components
        )

        if block_weights is None:
            base_weights = self._normalize_weights(
                weights if weights is not None else self._global_weights()
            )
            block_weights = stack(
                [base_weights for _ in range(len(self.partition))], axis=0
            )
        self._block_weights = self._normalize_block_weights(block_weights)
        self._sync_global_weights()

    def _block_particle_array(self):
        if hasattr(self, "particles"):
            return self.particles
        return self.filter_state.d

    def _global_weights(self):
        if hasattr(self, "weights"):
            return self.weights
        return self.filter_state.w

    @property
    def n_particles(self) -> int:
        """Return the number of weighted product-state particles."""
        return int(self._global_weights().shape[0])

    @property
    def n_blocks(self) -> int:
        """Return the number of partition blocks."""
        return len(self.partition)

    @staticmethod
    def _validate_partition(
        partition, n_components: int
    ) -> tuple[tuple[int, ...], ...]:
        if partition is None:
            return (tuple(range(n_components)),)
        if isinstance(partition, str):
            name = partition.strip().lower().replace("-", "_")
            if name in {"", "global", "full", "none"}:
                return (tuple(range(n_components)),)
            if name in {
                "component",
                "components",
                "singleton",
                "singletons",
                "factorized",
            }:
                return tuple((idx,) for idx in range(n_components))
            raise ValueError(
                "partition must be 'global', 'singleton', or an explicit sequence "
                "of component-index sequences."
            )

        normalized = []
        seen = set()
        for block_idx, raw_block in enumerate(partition):
            block = tuple(int(component_idx) for component_idx in raw_block)
            if not block:
                raise ValueError(f"partition block {block_idx} is empty.")
            for component_idx in block:
                if component_idx < 0 or component_idx >= n_components:
                    raise ValueError(
                        f"partition block {block_idx} contains component "
                        f"{component_idx}, but valid components are "
                        f"0..{n_components - 1}."
                    )
                if component_idx in seen:
                    raise ValueError(
                        f"component {component_idx} appears in more than one "
                        "partition block."
                    )
                seen.add(component_idx)
            normalized.append(block)

        missing = sorted(set(range(n_components)) - seen)
        if missing:
            raise ValueError(
                "partition must cover every component exactly once; "
                f"missing components {missing}."
            )
        return tuple(normalized)

    @staticmethod
    def _build_component_to_block(partition, n_components: int) -> tuple[int, ...]:
        component_to_block = [0 for _ in range(n_components)]
        for block_idx, block in enumerate(partition):
            for component_idx in block:
                component_to_block[component_idx] = block_idx
        return tuple(component_to_block)

    @staticmethod
    def _normalize_weights(weights):
        weights = array(weights, dtype=float)
        if ndim(weights) != 1:
            weights = reshape(weights, (-1,))
        if not all(weights >= 0.0):
            raise ValueError("Particle weights must be nonnegative.")
        weight_sum = sum(weights)
        if weight_sum <= 0.0:
            raise ValueError("At least one particle weight must be positive.")
        return weights / weight_sum

    @staticmethod
    def _normalize_log_weights(log_weights):
        log_weights = array(log_weights, dtype=float)
        if ndim(log_weights) != 1:
            log_weights = reshape(log_weights, (-1,))
        if any(isnan(log_weights)):
            raise ValueError("log weights must not contain NaN.")
        max_log_weight = max(log_weights)
        if not isfinite(max_log_weight):
            raise ValueError(
                "At least one log weight must be finite and no log weight may be +inf."
            )
        return BlockParticleFilter._normalize_weights(exp(log_weights - max_log_weight))

    def _normalize_block_weights(self, block_weights):
        block_weights = array(block_weights, dtype=float)
        if ndim(block_weights) == 1:
            if block_weights.shape[0] != self.n_particles:
                raise ValueError("block_weights must contain one value per particle.")
            normalized = self._normalize_weights(block_weights)
            return stack([normalized for _ in range(self.n_blocks)], axis=0)
        if ndim(block_weights) != 2:
            raise ValueError(
                "block_weights must have shape (n_particles,) or "
                "(n_blocks, n_particles)."
            )
        if block_weights.shape != (self.n_blocks, self.n_particles):
            raise ValueError(
                "block_weights must have shape "
                f"({self.n_blocks}, {self.n_particles})."
            )
        return stack(
            [self._normalize_weights(block_weights[i]) for i in range(self.n_blocks)],
            axis=0,
        )

    def _sync_global_weights(self) -> None:
        self.filter_state.w = self._normalize_weights(sum(self._block_weights, axis=0))

    @property
    def block_weights(self):
        """Return normalized block weights with shape ``(n_blocks, n_particles)``."""
        return self._block_weights

    def set_block_weights(self, block_weights) -> None:
        """Replace block weights and refresh compatibility weights."""
        self._block_weights = self._normalize_block_weights(block_weights)
        self._sync_global_weights()

    def set_particles(self, particles, weights=None, block_weights=None):
        """Replace particles and optionally global or block weights."""
        parent_set_particles = getattr(super(), "set_particles", None)
        if parent_set_particles is not None:
            parent_set_particles(particles, weights=weights)
        else:
            particles = array(particles, dtype=float)
            if particles.shape != self._block_particle_array().shape:
                raise ValueError(
                    "New particles must match the existing particle shape."
                )
            self.filter_state.d = particles
            if weights is not None:
                weights = self._normalize_weights(weights)
                if weights.shape[0] != self.n_particles:
                    raise ValueError("weights must match the particle count.")
                self.filter_state.w = weights

        if not hasattr(self, "_block_weights"):
            return
        if block_weights is not None:
            self.set_block_weights(block_weights)
        elif weights is not None:
            self.set_block_weights(weights)
        else:
            self._sync_global_weights()

    def component_weights(self, component_idx: int):
        """Return the weight vector used for one product-state component."""
        component_idx = int(component_idx)
        if component_idx < 0 or component_idx >= self._n_block_components:
            raise ValueError("component_idx is out of range.")
        return self._block_weights[self._component_to_block[component_idx]]

    def block_effective_sample_size(self):
        """Return one effective sample size per partition block."""
        return array(
            [
                1.0 / sum(self._block_weights[block_idx] ** 2)
                for block_idx in range(self.n_blocks)
            ]
        )

    def effective_sample_size(self):
        """Return the mean block effective sample size."""
        return sum(self.block_effective_sample_size()) / self.n_blocks

    def mode(self):
        """Return a block-wise modal product particle."""
        particles = to_numpy(self._block_particle_array())
        block_weights = to_numpy(self._block_weights)
        mode_particle = particles[0].copy()
        for block_idx, block in enumerate(self.partition):
            source_idx = int(block_weights[block_idx].argmax())
            mode_particle[list(block)] = particles[source_idx, list(block)]
        return array(mode_particle)

    @staticmethod
    def _systematic_indices(weights):
        weights_list = [float(weight) for weight in to_numpy(weights).reshape(-1)]
        n_particles = len(weights_list)
        start = float(to_numpy(random.rand(1)).reshape(-1)[0]) / n_particles
        positions = [start + i / n_particles for i in range(n_particles)]
        indices = []
        cumulative_weight = weights_list[0]
        source_index = 0
        for position in positions:
            while position > cumulative_weight and source_index < n_particles - 1:
                source_index += 1
                cumulative_weight += weights_list[source_index]
            indices.append(source_index)
        return array(indices)

    def resample_block_systematic(self, block_index: int):
        """Systematically resample one partition block and reset its weights."""
        block_index = int(block_index)
        if block_index < 0 or block_index >= self.n_blocks:
            raise ValueError("block_index is out of range.")
        indices = self._systematic_indices(
            self._normalize_weights(self._block_weights[block_index])
        )
        index_array = to_numpy(indices).astype(int)
        particles = to_numpy(self._block_particle_array()).copy()
        block = list(self.partition[block_index])
        particles[:, block, ...] = particles[index_array][:, block, ...]
        block_weights = to_numpy(self._block_weights).copy()
        block_weights[block_index] = 1.0 / self.n_particles
        self.set_particles(array(particles), block_weights=array(block_weights))
        return indices

    def resample_blocks_systematic(self, block_indices=None):
        """Systematically resample selected blocks and reset their weights."""
        if block_indices is None:
            block_indices = range(self.n_blocks)
        return stack(
            [self.resample_block_systematic(block_idx) for block_idx in block_indices],
            axis=0,
        )

    def resample_systematic(self):
        """Systematically resample all blocks and reset their weights."""
        return self.resample_blocks_systematic()

    def resample(self):
        """Resample all blocks and return the filter."""
        self.resample_blocks_systematic()
        return self

    def update_with_likelihood(
        self,
        likelihood: Callable | Sequence,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update all block weights from one likelihood per particle."""
        if callable(likelihood):
            values = (
                likelihood(self._block_particle_array())
                if measurement is None
                else likelihood(measurement, self._block_particle_array())
            )
        else:
            values = likelihood
        values = array(values, dtype=float)
        if values.shape != (self.n_particles,):
            raise ValueError("likelihood must return one value per particle.")
        if not all(values >= 0.0):
            raise ValueError("likelihood values must be nonnegative.")
        return self.update_with_log_likelihood(
            log(values), resample=resample, ess_threshold=ess_threshold
        )

    def update_with_log_likelihood(
        self,
        log_likelihood: Callable | Sequence,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update all block weights from one log-likelihood per particle."""
        if callable(log_likelihood):
            values = (
                log_likelihood(self._block_particle_array())
                if measurement is None
                else log_likelihood(measurement, self._block_particle_array())
            )
        else:
            values = log_likelihood
        values = array(values, dtype=float)
        if values.shape != (self.n_particles,):
            raise ValueError("log_likelihood must return one value per particle.")
        return self.update_with_block_log_likelihoods(
            stack([values for _ in range(self.n_blocks)], axis=0),
            resample=resample,
            ess_threshold=ess_threshold,
        )

    def update_with_block_likelihoods(
        self,
        likelihood: Callable | Sequence,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update block weights from nonnegative block likelihoods."""
        if callable(likelihood):
            values = (
                likelihood(self._block_particle_array())
                if measurement is None
                else likelihood(measurement, self._block_particle_array())
            )
        else:
            values = likelihood
        values = array(values, dtype=float)
        if values.shape != (self.n_blocks, self.n_particles):
            raise ValueError(
                "block likelihoods must have shape "
                f"({self.n_blocks}, {self.n_particles})."
            )
        if not all(values >= 0.0):
            raise ValueError("likelihood values must be nonnegative.")
        return self.update_with_block_log_likelihoods(
            log(values), resample=resample, ess_threshold=ess_threshold
        )

    def _resampling_thresholds(self, ess_threshold):
        if ess_threshold is None:
            return [self.n_particles / 2.0 for _ in range(self.n_blocks)]
        values = to_numpy(array(ess_threshold, dtype=float)).reshape(-1)
        if values.shape[0] == 1:
            return [float(values[0]) for _ in range(self.n_blocks)]
        if values.shape[0] != self.n_blocks:
            raise ValueError("ess_threshold must be scalar or one value per block.")
        return [float(value) for value in values]

    def update_with_block_log_likelihoods(
        self,
        log_likelihood: Callable | Sequence,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update block weights from block log-likelihoods."""
        if callable(log_likelihood):
            values = (
                log_likelihood(self._block_particle_array())
                if measurement is None
                else log_likelihood(measurement, self._block_particle_array())
            )
        else:
            values = log_likelihood
        values = array(values, dtype=float)
        if values.shape != (self.n_blocks, self.n_particles):
            raise ValueError(
                "block log-likelihoods must have shape "
                f"({self.n_blocks}, {self.n_particles})."
            )
        self._block_weights = stack(
            [
                self._normalize_log_weights(log(self._block_weights[i]) + values[i])
                for i in range(self.n_blocks)
            ],
            axis=0,
        )
        self._sync_global_weights()
        ess = self.block_effective_sample_size()
        if resample:
            thresholds = self._resampling_thresholds(ess_threshold)
            to_resample = [
                i
                for i, value in enumerate(to_numpy(ess).reshape(-1))
                if float(value) < thresholds[i]
            ]
            if to_resample:
                self.resample_blocks_systematic(to_resample)
        return ess

    def update_with_component_likelihoods(
        self,
        component_likelihoods,
        *,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update from per-component likelihoods shaped ``(n_particles, K)``."""
        values = array(component_likelihoods, dtype=float)
        if values.shape != (self.n_particles, self._n_block_components):
            raise ValueError(
                "component_likelihoods must have shape "
                f"({self.n_particles}, {self._n_block_components})."
            )
        if not all(values >= 0.0):
            raise ValueError("likelihood values must be nonnegative.")
        return self.update_with_component_log_likelihoods(
            log(values), resample=resample, ess_threshold=ess_threshold
        )

    def update_with_component_log_likelihoods(
        self,
        component_log_likelihoods,
        *,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update from per-component log-likelihoods shaped ``(n_particles, K)``."""
        values = array(component_log_likelihoods, dtype=float)
        if values.shape != (self.n_particles, self._n_block_components):
            raise ValueError(
                "component_log_likelihoods must have shape "
                f"({self.n_particles}, {self._n_block_components})."
            )
        block_logs = []
        for block in self.partition:
            block_logs.append(
                sum(
                    stack(
                        [values[:, component_idx] for component_idx in block], axis=1
                    ),
                    axis=1,
                )
            )
        return self.update_with_block_log_likelihoods(
            stack(block_logs, axis=0), resample=resample, ess_threshold=ess_threshold
        )
