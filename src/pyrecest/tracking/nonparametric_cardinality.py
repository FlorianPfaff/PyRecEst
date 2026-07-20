"""Bayesian nonparametric cardinality priors for tracking association.

The classes in this module expose Dirichlet-process and Pitman--Yor-process
Chinese-restaurant-process predictive probabilities in a small, deterministic
form that can be used by multitarget trackers for birth/association proposals or
for prior predictive diagnostics of the number of target-generated clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, lgamma, log
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class PitmanYorCardinalityPrior:
    """Pitman--Yor Chinese-restaurant prior over target-generated clusters.

    Parameters
    ----------
    strength : float, optional
        Pitman--Yor strength parameter ``theta``. It must satisfy
        ``strength > -discount``. For the Dirichlet-process special case
        ``discount == 0``, this means ``strength > 0``.
    discount : float, optional
        Pitman--Yor discount parameter ``d`` with ``0 <= d < 1``. Positive
        values encourage heavier-tailed cluster counts than a Dirichlet process.

    Notes
    -----
    Given current cluster sizes ``n_1, ..., n_K`` and ``n = sum_k n_k``, the
    predictive probabilities are

    ``P(existing k) = (n_k - d) / (theta + n)``

    and

    ``P(new cluster) = (theta + d K) / (theta + n)``.

    The returned probabilities are intended as *prior* association weights; a
    tracker should still multiply them by measurement likelihoods, survival
    probabilities, and clutter alternatives.
    """

    strength: float = 1.0
    discount: float = 0.0

    def __post_init__(self) -> None:
        strength = float(self.strength)
        discount = float(self.discount)
        if not isfinite(strength):
            raise ValueError("strength must be finite.")
        if not isfinite(discount):
            raise ValueError("discount must be finite.")
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must satisfy 0 <= discount < 1.")
        if strength <= -discount:
            raise ValueError("strength must satisfy strength > -discount.")
        object.__setattr__(self, "strength", strength)
        object.__setattr__(self, "discount", discount)

    @property
    def is_dirichlet_process(self) -> bool:
        """Return true if the prior is the Dirichlet-process special case."""

        return self.discount == 0.0

    def predictive_weights(self, cluster_sizes: Sequence[int]) -> tuple[float, ...]:
        """Return unnormalized existing-cluster and new-cluster weights.

        The final returned entry is always the weight for a new target-generated
        cluster. The preceding entries correspond to the supplied cluster sizes
        in order.
        """

        sizes = _validate_cluster_sizes(cluster_sizes)
        if not sizes:
            return (1.0,)
        existing_weights = tuple(float(size) - self.discount for size in sizes)
        new_weight = self.strength + self.discount * len(sizes)
        return existing_weights + (float(new_weight),)

    def predictive_probabilities(
        self, cluster_sizes: Sequence[int]
    ) -> tuple[float, ...]:
        """Return CRP predictive probabilities for existing clusters and a new cluster."""

        sizes = _validate_cluster_sizes(cluster_sizes)
        if not sizes:
            return (1.0,)
        denominator = self.strength + float(sum(sizes))
        return tuple(weight / denominator for weight in self.predictive_weights(sizes))

    def predictive_log_probabilities(
        self, cluster_sizes: Sequence[int]
    ) -> tuple[float, ...]:
        """Return log predictive probabilities for existing clusters and a new cluster."""

        return tuple(
            log(probability)
            for probability in self.predictive_probabilities(cluster_sizes)
        )

    def log_exchangeable_partition_probability(
        self, cluster_sizes: Sequence[int]
    ) -> float:
        """Return the EPPF log probability of one partition with given block sizes.

        The value is the probability of one exchangeable partition whose block
        sizes are ``cluster_sizes``. It is not the marginal probability of the
        unordered size histogram, because the latter also depends on the number
        of set partitions with those sizes.
        """

        sizes = _validate_cluster_sizes(cluster_sizes)
        if not sizes:
            return 0.0

        num_observations = sum(sizes)
        num_clusters = len(sizes)
        log_probability = 0.0

        for cluster_index in range(1, num_clusters):
            log_probability += log(self.strength + self.discount * cluster_index)
        for observation_index in range(1, num_observations):
            log_probability -= log(self.strength + observation_index)
        for size in sizes:
            log_probability += _log_rising_factorial(1.0 - self.discount, size - 1)
        return float(log_probability)

    def exchangeable_partition_probability(self, cluster_sizes: Sequence[int]) -> float:
        """Return the EPPF probability of one partition with given block sizes."""

        return float(exp(self.log_exchangeable_partition_probability(cluster_sizes)))

    def cluster_count_pmf(self, num_observations: int) -> tuple[float, ...]:
        """Return ``P(K_n = k)`` for ``k = 0, ..., num_observations``.

        ``K_n`` is the number of occupied target-generated clusters after ``n``
        exchangeable observations under the CRP prior. Entry zero is one only
        when ``num_observations == 0``.
        """

        n = _validate_nonnegative_int(num_observations, "num_observations")
        pmf = [1.0]
        for occupied_observations in range(n):
            next_pmf = [0.0] * (occupied_observations + 2)
            if occupied_observations == 0:
                next_pmf[1] = 1.0
                pmf = next_pmf
                continue
            denominator = self.strength + float(occupied_observations)
            for num_clusters, probability in enumerate(pmf):
                if probability == 0.0 or num_clusters == 0:
                    continue
                stay_weight = float(occupied_observations) - self.discount * float(
                    num_clusters
                )
                new_weight = self.strength + self.discount * float(num_clusters)
                next_pmf[num_clusters] += probability * stay_weight / denominator
                next_pmf[num_clusters + 1] += probability * new_weight / denominator
            pmf = next_pmf
        return tuple(float(probability) for probability in pmf)

    def expected_number_of_clusters(self, num_observations: int) -> float:
        """Return the prior expected number of occupied clusters after ``n`` observations."""

        pmf = self.cluster_count_pmf(num_observations)
        return float(
            sum(
                num_clusters * probability
                for num_clusters, probability in enumerate(pmf)
            )
        )

    def cluster_count_tail_probability(
        self, num_observations: int, min_clusters: int
    ) -> float:
        """Return ``P(K_n >= min_clusters)`` under the prior predictive PMF."""

        min_clusters = _validate_nonnegative_int(min_clusters, "min_clusters")
        pmf = self.cluster_count_pmf(num_observations)
        if min_clusters >= len(pmf):
            return 0.0
        return float(sum(pmf[min_clusters:]))


class DirichletProcessCardinalityPrior(PitmanYorCardinalityPrior):
    """Dirichlet-process cardinality prior with zero Pitman--Yor discount."""

    def __init__(self, strength: float = 1.0) -> None:
        super().__init__(strength=strength, discount=0.0)


def _validate_cluster_sizes(cluster_sizes: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(cluster_sizes)
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError("cluster sizes must be positive integers.")
        if int(size) <= 0:
            raise ValueError("cluster sizes must be positive integers.")
    return tuple(int(size) for size in sizes)


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _log_rising_factorial(start: float, length: int) -> float:
    if length <= 0:
        return 0.0
    return float(lgamma(start + length) - lgamma(start))
