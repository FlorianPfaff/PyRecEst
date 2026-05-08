"""Partitioned particle filter for Cartesian products of SO(3)."""

from collections.abc import Callable, Sequence

# pylint: disable=no-name-in-module,no-member,too-many-positional-arguments
from pyrecest.backend import all, array, exp, log, ndim, ones, random, stack, sum, to_numpy
from pyrecest.distributions import SO3DiracDistribution
from pyrecest.distributions._so3_helpers import geodesic_distance

from .so3_product_particle_filter import SO3ProductParticleFilter


class PartitionedSO3ProductParticleFilter(SO3ProductParticleFilter):
    """Particle filter for ``SO(3)^K`` with independent partition weights.

    The filter approximates the product-state posterior by a product over a
    user-supplied partition of the ``K`` SO(3) components. Each block keeps its
    own particle weights and may be resampled independently. This preserves
    correlations inside each block while avoiding the degeneracy of a single
    global weight vector in high-dimensional product spaces.

    The inherited ``weights`` property remains available as a normalized average
    of the block weights for API compatibility. Point estimates, block ESS, and
    block resampling use the block-specific weights.
    """

    def __init__(
        self,
        n_particles: int,
        num_rotations: int,
        partition=None,
        initial_particles=None,
        weights=None,
        block_weights=None,
    ) -> None:
        super().__init__(
            n_particles=n_particles,
            num_rotations=num_rotations,
            initial_particles=initial_particles,
            weights=weights,
        )
        self.partition = self._validate_partition(partition, self.num_rotations)
        self._component_to_block = self._build_component_to_block(
            self.partition, self.num_rotations
        )

        if block_weights is None:
            base_weights = (
                self.weights
                if weights is not None
                else ones(self.n_particles) / self.n_particles
            )
            block_weights = stack(
                [base_weights for _ in range(len(self.partition))], axis=0
            )
        self._block_weights = self._normalize_block_weights(block_weights)
        self._sync_global_weights()

    @staticmethod
    def _validate_partition(
        partition, num_rotations: int
    ) -> tuple[tuple[int, ...], ...]:
        if partition is None:
            return (tuple(range(num_rotations)),)

        if isinstance(partition, str):
            name = partition.strip().lower().replace("-", "_")
            if name in {"", "global", "full", "none"}:
                return (tuple(range(num_rotations)),)
            if name in {
                "component",
                "components",
                "singleton",
                "singletons",
                "factorized",
            }:
                return tuple((idx,) for idx in range(num_rotations))
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
                if component_idx < 0 or component_idx >= num_rotations:
                    raise ValueError(
                        f"partition block {block_idx} contains component "
                        f"{component_idx}, but valid components are "
                        f"0..{num_rotations - 1}."
                    )
                if component_idx in seen:
                    raise ValueError(
                        f"component {component_idx} appears in more than one "
                        "partition block."
                    )
                seen.add(component_idx)
            normalized.append(block)

        missing = sorted(set(range(num_rotations)) - seen)
        if missing:
            raise ValueError(
                "partition must cover every component exactly once; "
                f"missing components {missing}."
            )
        return tuple(normalized)

    @staticmethod
    def _build_component_to_block(partition, num_rotations: int) -> tuple[int, ...]:
        component_to_block = [0 for _ in range(num_rotations)]
        for block_idx, block in enumerate(partition):
            for component_idx in block:
                component_to_block[component_idx] = block_idx
        return tuple(component_to_block)

    def _normalize_block_weights(self, block_weights):
        block_weights = array(block_weights, dtype=float)
        if ndim(block_weights) == 1:
            if block_weights.shape[0] != self.n_particles:
                raise ValueError("block_weights must contain one value per particle.")
            normalized = self._normalize_weights(block_weights)
            return stack([normalized for _ in range(len(self.partition))], axis=0)
        if ndim(block_weights) != 2:
            raise ValueError(
                "block_weights must have shape (n_particles,) or "
                "(n_blocks, n_particles)."
            )
        if block_weights.shape != (len(self.partition), self.n_particles):
            raise ValueError(
                "block_weights must have shape "
                f"({len(self.partition)}, {self.n_particles})."
            )
        return stack(
            [
                self._normalize_weights(block_weights[i])
                for i in range(len(self.partition))
            ],
            axis=0,
        )

    def _sync_global_weights(self) -> None:
        self.filter_state.w = self._normalize_weights(sum(self._block_weights, axis=0))

    @property
    def block_weights(self):
        """Return normalized block weights with shape ``(n_blocks, n_particles)``."""
        return self._block_weights

    def set_block_weights(self, block_weights) -> None:
        """Replace the block weights and refresh the compatibility weights."""
        self._block_weights = self._normalize_block_weights(block_weights)
        self._sync_global_weights()

    def set_particles(self, particles, weights=None, block_weights=None):
        """Replace particles and optionally global or block weights."""
        super().set_particles(particles, weights=weights)
        if not hasattr(self, "_block_weights"):
            return
        if block_weights is not None:
            self.set_block_weights(block_weights)
        elif weights is not None:
            self.set_block_weights(weights)
        else:
            self._sync_global_weights()

    def component_weights(self, component_idx: int):
        """Return the weight vector used for one SO(3) component."""
        if component_idx < 0 or component_idx >= self.num_rotations:
            raise ValueError("component_idx is out of range.")
        return self._block_weights[self._component_to_block[int(component_idx)]]

    def block_effective_sample_size(self):
        """Return one effective sample size per partition block."""
        return array(
            [
                1.0 / sum(self._block_weights[block_idx] ** 2)
                for block_idx in range(len(self.partition))
            ]
        )

    def effective_sample_size(self):
        """Return the mean block effective sample size."""
        return sum(self.block_effective_sample_size()) / len(self.partition)

    def mean(self):
        """Return the component-wise chordal mean using block-specific weights."""
        means = [
            SO3DiracDistribution(
                self.particles[:, component_idx, :],
                self.component_weights(component_idx),
            ).mean()
            for component_idx in range(self.num_rotations)
        ]
        return stack(means, axis=0)

    def mode(self):
        """Return a block-wise modal product particle.

        Since the posterior is represented as a product over blocks, the returned
        point may be a hybrid assembled from different source particles.
        """
        particles = to_numpy(self.particles)
        block_weights = to_numpy(self._block_weights)
        mode_particle = particles[0].copy()
        for block_idx, block in enumerate(self.partition):
            source_idx = int(block_weights[block_idx].argmax())
            mode_particle[list(block)] = particles[source_idx, list(block)]
        return array(mode_particle)

    def get_point_estimate(self):
        """Return the component-wise SO(3) mean."""
        return self.mean()

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
        if block_index < 0 or block_index >= len(self.partition):
            raise ValueError("block_index is out of range.")

        weights = self._normalize_weights(self._block_weights[block_index])
        indices = self._systematic_indices(weights)
        index_array = to_numpy(indices).astype(int)
        particles = to_numpy(self.particles).copy()
        block = list(self.partition[block_index])
        particles[:, block, :] = particles[index_array][:, block, :]

        block_weights = to_numpy(self._block_weights).copy()
        block_weights[block_index] = 1.0 / self.n_particles
        super().set_particles(array(particles))
        self._block_weights = array(block_weights)
        self._sync_global_weights()
        return indices

    def resample_blocks_systematic(self, block_indices=None):
        """Systematically resample selected blocks and reset their weights."""
        if block_indices is None:
            block_indices = range(len(self.partition))
        indices = [
            self.resample_block_systematic(block_idx) for block_idx in block_indices
        ]
        return stack(indices, axis=0)

    def update_with_block_likelihoods(
        self,
        likelihood: Callable | Sequence,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update block weights from nonnegative block likelihoods.

        The likelihood must evaluate to an array shaped
        ``(n_blocks, n_particles)``. Each row updates the corresponding block's
        weights independently.
        """
        if callable(likelihood):
            if measurement is None:
                likelihood_values = likelihood(self.particles)
            else:
                likelihood_values = likelihood(measurement, self.particles)
        else:
            likelihood_values = likelihood
        likelihood_values = array(likelihood_values, dtype=float)
        if likelihood_values.shape != (len(self.partition), self.n_particles):
            raise ValueError(
                "block likelihoods must have shape "
                f"({len(self.partition)}, {self.n_particles})."
            )
        if not all(likelihood_values >= 0.0):
            raise ValueError("likelihood values must be nonnegative.")

        self._block_weights = stack(
            [
                self._normalize_weights(
                    self._block_weights[block_idx] * likelihood_values[block_idx]
                )
                for block_idx in range(len(self.partition))
            ],
            axis=0,
        )
        self._sync_global_weights()
        ess = self.block_effective_sample_size()
        threshold = self.n_particles / 2.0 if ess_threshold is None else ess_threshold
        if resample:
            ess_values = to_numpy(ess).reshape(-1)
            resample_blocks = [
                block_idx
                for block_idx, block_ess in enumerate(ess_values)
                if float(block_ess) < threshold
            ]
            if resample_blocks:
                self.resample_blocks_systematic(resample_blocks)
        return ess

    def update_with_block_log_likelihoods(
        self,
        log_likelihood,
        measurement=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update block weights from block log-likelihoods.

        The log-likelihood must evaluate to an array shaped
        ``(n_blocks, n_particles)``. Each row updates the corresponding block's
        weights independently in log space, avoiding likelihood underflow.
        """
        if callable(log_likelihood):
            if measurement is None:
                log_likelihood_values = log_likelihood(self.particles)
            else:
                log_likelihood_values = log_likelihood(measurement, self.particles)
        else:
            log_likelihood_values = log_likelihood
        log_likelihood_values = array(log_likelihood_values, dtype=float)
        if log_likelihood_values.shape != (len(self.partition), self.n_particles):
            raise ValueError(
                "block log-likelihoods must have shape "
                f"({len(self.partition)}, {self.n_particles})."
            )

        self._block_weights = stack(
            [
                self._normalize_log_weights(
                    log(self._normalize_weights(self._block_weights[block_idx]))
                    + log_likelihood_values[block_idx]
                )
                for block_idx in range(len(self.partition))
            ],
            axis=0,
        )
        self._sync_global_weights()
        ess = self.block_effective_sample_size()
        threshold = self.n_particles / 2.0 if ess_threshold is None else ess_threshold
        if resample:
            ess_values = to_numpy(ess).reshape(-1)
            resample_blocks = [
                block_idx
                for block_idx, block_ess in enumerate(ess_values)
                if float(block_ess) < threshold
            ]
            if resample_blocks:
                self.resample_blocks_systematic(resample_blocks)
        return ess

    def update_with_component_likelihoods(
        self,
        component_likelihoods,
        *,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update from per-component likelihoods shaped ``(n_particles, K)``."""
        component_likelihoods = array(component_likelihoods, dtype=float)
        if component_likelihoods.shape != (self.n_particles, self.num_rotations):
            raise ValueError(
                "component_likelihoods must have shape "
                f"({self.n_particles}, {self.num_rotations})."
            )
        if not all(component_likelihoods >= 0.0):
            raise ValueError("likelihood values must be nonnegative.")

        block_likelihoods = []
        for block in self.partition:
            block_likelihood = ones(self.n_particles)
            for component_idx in block:
                block_likelihood = (
                    block_likelihood * component_likelihoods[:, component_idx]
                )
            block_likelihoods.append(block_likelihood)
        return self.update_with_block_likelihoods(
            stack(block_likelihoods, axis=0),
            resample=resample,
            ess_threshold=ess_threshold,
        )

    def update_with_component_log_likelihoods(
        self,
        component_log_likelihoods,
        *,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update from per-component log-likelihoods shaped ``(n_particles, K)``."""
        component_log_likelihoods = array(component_log_likelihoods, dtype=float)
        if component_log_likelihoods.shape != (
            self.n_particles,
            self.num_rotations,
        ):
            raise ValueError(
                "component_log_likelihoods must have shape "
                f"({self.n_particles}, {self.num_rotations})."
            )

        block_log_likelihoods = []
        for block in self.partition:
            block_log_likelihood = component_log_likelihoods[:, block[0]]
            for component_idx in block[1:]:
                block_log_likelihood = (
                    block_log_likelihood + component_log_likelihoods[:, component_idx]
                )
            block_log_likelihoods.append(block_log_likelihood)
        return self.update_with_block_log_likelihoods(
            stack(block_log_likelihoods, axis=0),
            resample=resample,
            ess_threshold=ess_threshold,
        )

    def update_with_geodesic_likelihood(
        self,
        measurement,
        noise_std,
        *,
        mask=None,
        resample: bool = True,
        ess_threshold=None,
    ):
        """Update with isotropic masked geodesic likelihoods per partition block."""
        if noise_std <= 0.0:
            raise ValueError("noise_std must be positive.")

        measurement = self._as_product_point(measurement, self.num_rotations)
        if mask is None:
            mask = ones(self.num_rotations)
        else:
            mask = array(mask, dtype=float)
            if mask.shape != (self.num_rotations,):
                raise ValueError("mask must have shape (num_rotations,).")

        distances = stack(
            [
                geodesic_distance(self.particles[:, i, :], measurement[i, :])
                for i in range(self.num_rotations)
            ],
            axis=1,
        )
        block_likelihoods = []
        for block in self.partition:
            quadratic_terms = stack(
                [mask[i] * distances[:, i] ** 2 for i in block],
                axis=1,
            )
            quadratic = sum(quadratic_terms, axis=1) / (noise_std**2)
            block_likelihoods.append(exp(-0.5 * quadratic))
        return self.update_with_block_likelihoods(
            stack(block_likelihoods, axis=0),
            resample=resample,
            ess_threshold=ess_threshold,
        )
