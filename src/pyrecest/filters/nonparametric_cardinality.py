"""Bayesian nonparametric cardinality utilities for multi-target tracking.

The classes in this module are deliberately small building blocks rather than
complete multi-target Bayes filters.  They expose the predictive cardinality and
partition behavior of Dirichlet-process and Pitman--Yor-process priors so that
trackers can use them for birth modeling, measurement partitioning, or posterior
predictive diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from numbers import Integral, Real

from scipy.special import gammaln


def _as_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validate_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_probability(value, name):
    value = _as_float(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _clip_probability(value, lower, upper):
    return min(max(float(value), lower), upper)


def _log_rising_factorial(value, count):
    count = _validate_nonnegative_integer(count, "count")
    if count == 0:
        return 0.0
    return float(gammaln(value + count) - gammaln(value))


@dataclass(frozen=True)
class PitmanYorProcessCardinalityPrior:
    """Predictive partition/cardinality prior induced by a Pitman--Yor process.

    Parameters
    ----------
    discount : float, optional
        Discount parameter ``d``.  Must satisfy ``0 <= d < 1``.
    strength : float, optional
        Strength/concentration parameter ``theta``.  Must satisfy
        ``theta > -d``.  For ``d == 0``, this is the ordinary Dirichlet-process
        concentration parameter and must be positive.

    Notes
    -----
    For occupied cluster sizes ``n_1, ..., n_K`` and ``n = sum_j n_j``, the
    predictive probabilities are

    ``P(existing k) = (n_k - d) / (theta + n)``

    and

    ``P(new) = (theta + d K) / (theta + n)``.

    The Dirichlet-process case is recovered by setting ``discount=0``.
    """

    discount: float = 0.0
    strength: float = 1.0

    def __post_init__(self):
        discount = _as_float(self.discount, "discount")
        strength = _as_float(self.strength, "strength")

        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must satisfy 0 <= discount < 1.")
        if strength <= -discount:
            raise ValueError("strength must satisfy strength > -discount.")
        if discount == 0.0 and strength <= 0.0:
            raise ValueError("strength must be positive when discount is zero.")

        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "strength", strength)

    @property
    def is_dirichlet_process(self):
        """Return whether this prior reduces to a Dirichlet process."""
        return self.discount == 0.0

    @staticmethod
    def _coerce_cluster_sizes(cluster_sizes):
        try:
            raw_sizes = tuple(cluster_sizes)
        except TypeError as exc:
            raise TypeError(
                "cluster_sizes must be an iterable of positive integers."
            ) from exc

        sizes = tuple(
            _validate_positive_integer(size, "cluster_sizes") for size in raw_sizes
        )
        return sizes

    @staticmethod
    def _validate_count_state(num_observations, num_clusters):
        num_observations = _validate_nonnegative_integer(
            num_observations, "num_observations"
        )
        num_clusters = _validate_nonnegative_integer(num_clusters, "num_clusters")
        if num_observations == 0:
            if num_clusters != 0:
                raise ValueError(
                    "num_clusters must be zero when num_observations is zero."
                )
        elif not 1 <= num_clusters <= num_observations:
            raise ValueError("num_clusters must be between one and num_observations.")
        return num_observations, num_clusters

    def predictive_existing_cluster_probabilities(self, cluster_sizes):
        """Return predictive assignment probabilities for existing clusters.

        Parameters
        ----------
        cluster_sizes : iterable of int
            Positive occupied cluster sizes.

        Returns
        -------
        tuple of float
            Probabilities for assigning the next observation to each existing
            cluster, in the same order as ``cluster_sizes``.
        """
        sizes = self._coerce_cluster_sizes(cluster_sizes)
        num_observations = sum(sizes)
        if num_observations == 0:
            return ()

        denominator = self.strength + num_observations
        return tuple((size - self.discount) / denominator for size in sizes)

    def predictive_new_cluster_probability(self, cluster_sizes):
        """Return the predictive probability that the next observation starts a new cluster."""
        sizes = self._coerce_cluster_sizes(cluster_sizes)
        return self.predictive_new_cluster_probability_from_counts(
            sum(sizes), len(sizes)
        )

    def predictive_new_cluster_probability_from_counts(
        self, num_observations, num_clusters
    ):
        """Return the next-observation new-cluster probability from count summaries.

        Parameters
        ----------
        num_observations : int
            Number of previous observations assigned to non-clutter clusters.
        num_clusters : int
            Number of occupied clusters represented by those observations.
        """
        num_observations, num_clusters = self._validate_count_state(
            num_observations, num_clusters
        )
        if num_observations == 0:
            return 1.0
        return (self.strength + self.discount * num_clusters) / (
            self.strength + num_observations
        )

    def predictive_assignment_probabilities(self, cluster_sizes):
        """Return existing-cluster probabilities followed by the new-cluster probability."""
        existing_probabilities = self.predictive_existing_cluster_probabilities(
            cluster_sizes
        )
        return existing_probabilities + (
            self.predictive_new_cluster_probability(cluster_sizes),
        )

    def expected_number_of_clusters(self, num_observations):
        """Return ``E[K_n]`` after ``num_observations`` exchangeable observations."""
        num_observations = _validate_nonnegative_integer(
            num_observations, "num_observations"
        )
        return self.expected_additional_clusters(num_observations)

    def expected_additional_clusters(
        self, additional_observations, initial_observations=0, initial_clusters=0
    ):
        """Return the expected number of new clusters in a future batch.

        The recurrence is exact for the Pitman--Yor predictive rule because the
        new-cluster probability is linear in the current number of clusters.
        """
        additional_observations = _validate_nonnegative_integer(
            additional_observations, "additional_observations"
        )
        initial_observations, initial_clusters = self._validate_count_state(
            initial_observations, initial_clusters
        )

        observations = initial_observations
        expected_total_clusters = float(initial_clusters)
        for _ in range(additional_observations):
            if observations == 0:
                new_cluster_probability = 1.0
            else:
                new_cluster_probability = (
                    self.strength + self.discount * expected_total_clusters
                ) / (self.strength + observations)
            expected_total_clusters += new_cluster_probability
            observations += 1
        return expected_total_clusters - initial_clusters

    def log_eppf(self, cluster_sizes):
        """Return the log exchangeable partition probability for cluster sizes.

        ``cluster_sizes`` are treated as the occupied block sizes of an unordered
        exchangeable partition.  The empty partition has log probability zero.
        """
        sizes = self._coerce_cluster_sizes(cluster_sizes)
        num_observations = sum(sizes)
        num_clusters = len(sizes)
        if num_observations == 0:
            return 0.0

        log_probability = 0.0
        for cluster_index in range(1, num_clusters):
            log_probability += log(self.strength + cluster_index * self.discount)
        log_probability -= _log_rising_factorial(
            self.strength + 1.0, num_observations - 1
        )
        for size in sizes:
            log_probability += _log_rising_factorial(1.0 - self.discount, size - 1)
        return float(log_probability)

    def eppf(self, cluster_sizes):
        """Return the exchangeable partition probability for cluster sizes."""
        return exp(self.log_eppf(cluster_sizes))


class DirichletProcessCardinalityPrior(PitmanYorProcessCardinalityPrior):
    """Dirichlet-process cardinality prior.

    This is a convenience wrapper around
    :class:`PitmanYorProcessCardinalityPrior` with ``discount=0``.
    """

    def __init__(self, concentration=1.0):
        super().__init__(discount=0.0, strength=concentration)

    @property
    def concentration(self):
        """Return the Dirichlet-process concentration parameter."""
        return self.strength


@dataclass(frozen=True)
class PitmanYorBirthProbability:
    """Callable Pitman--Yor birth-existence heuristic for lightweight MTT trackers.

    The callable maps the Pitman--Yor new-cluster predictive probability to a
    Bernoulli birth existence probability.  It is intentionally a calibrated
    heuristic, not a complete BNP multi-object posterior: each active Bernoulli
    component and each already-created birth in the current update contributes
    one pseudo-observation and one occupied cluster.

    Parameters
    ----------
    discount : float, optional
        Pitman--Yor discount parameter.  Larger values make successive births in
        a burst less suppressed relative to a Dirichlet process.
    strength : float, optional
        Pitman--Yor strength/concentration parameter.
    base_birth_existence_probability : float, optional
        Birth existence probability used when the new-cluster probability is one.
    prior_observation_count : int, optional
        Number of pseudo-observations carried into the call before counting live
        components.  Use this to encode historical target evidence.
    prior_cluster_count : int, optional
        Number of occupied pseudo-clusters carried into the call before counting
        live components.
    minimum_probability, maximum_probability : float, optional
        Clipping bounds for numerical stability and compatibility with Bernoulli
        component validation.
    """

    discount: float = 0.5
    strength: float = 1.0
    base_birth_existence_probability: float = 0.8
    prior_observation_count: int = 0
    prior_cluster_count: int = 0
    minimum_probability: float = 1e-12
    maximum_probability: float = 1.0 - 1e-12
    cardinality_prior: PitmanYorProcessCardinalityPrior = field(init=False)

    def __post_init__(self):
        cardinality_prior = PitmanYorProcessCardinalityPrior(
            self.discount, self.strength
        )
        base_birth_existence_probability = _validate_probability(
            self.base_birth_existence_probability, "base_birth_existence_probability"
        )
        prior_observation_count, prior_cluster_count = (
            cardinality_prior._validate_count_state(
                self.prior_observation_count, self.prior_cluster_count
            )
        )
        minimum_probability = _validate_probability(
            self.minimum_probability, "minimum_probability"
        )
        maximum_probability = _validate_probability(
            self.maximum_probability, "maximum_probability"
        )
        if minimum_probability > maximum_probability:
            raise ValueError("minimum_probability must not exceed maximum_probability.")

        object.__setattr__(self, "cardinality_prior", cardinality_prior)
        object.__setattr__(
            self, "base_birth_existence_probability", base_birth_existence_probability
        )
        object.__setattr__(self, "prior_observation_count", prior_observation_count)
        object.__setattr__(self, "prior_cluster_count", prior_cluster_count)
        object.__setattr__(self, "minimum_probability", minimum_probability)
        object.__setattr__(self, "maximum_probability", maximum_probability)

    def __call__(
        self,
        *,
        measurement=None,
        measurement_index=None,
        num_measurements=None,
        num_existing_components=0,
        num_new_births=0,
    ):
        """Return a context-dependent birth existence probability.

        The measurement-valued keyword arguments are accepted so instances can be
        used directly as ``birth_existence_probability`` callables in
        :class:`~pyrecest.filters.MultiBernoulliTracker`.  The current heuristic
        only uses ``num_existing_components`` and ``num_new_births``.
        """
        del measurement, measurement_index, num_measurements

        num_existing_components = _validate_nonnegative_integer(
            num_existing_components, "num_existing_components"
        )
        num_new_births = _validate_nonnegative_integer(num_new_births, "num_new_births")
        num_observations = (
            self.prior_observation_count + num_existing_components + num_new_births
        )
        num_clusters = (
            self.prior_cluster_count + num_existing_components + num_new_births
        )
        new_cluster_probability = (
            self.cardinality_prior.predictive_new_cluster_probability_from_counts(
                num_observations, num_clusters
            )
        )
        birth_probability = (
            self.base_birth_existence_probability * new_cluster_probability
        )
        return _clip_probability(
            birth_probability, self.minimum_probability, self.maximum_probability
        )
