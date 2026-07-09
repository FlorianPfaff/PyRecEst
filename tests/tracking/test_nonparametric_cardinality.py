import math

import pytest
from pyrecest.tracking.nonparametric_cardinality import (
    DirichletProcessCardinalityPrior,
    PitmanYorCardinalityPrior,
)


def test_dirichlet_process_predictive_probability_matches_crp_rule():
    prior = DirichletProcessCardinalityPrior(strength=2.0)

    probabilities = prior.predictive_probabilities((3, 1))

    assert probabilities == pytest.approx((3.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0))
    assert sum(probabilities) == pytest.approx(1.0)


def test_pitman_yor_predictive_probability_uses_discounted_counts():
    prior = PitmanYorCardinalityPrior(strength=2.0, discount=0.25)

    probabilities = prior.predictive_probabilities((3, 1))

    assert probabilities == pytest.approx((2.75 / 6.0, 0.75 / 6.0, 2.5 / 6.0))
    assert prior.predictive_weights((3, 1)) == pytest.approx((2.75, 0.75, 2.5))


def test_first_observation_always_creates_a_cluster():
    prior = PitmanYorCardinalityPrior(strength=0.0, discount=0.5)

    assert prior.predictive_probabilities(()) == (1.0,)
    assert prior.cluster_count_pmf(1) == (0.0, 1.0)


def test_cluster_count_pmf_matches_small_dirichlet_process_case():
    prior = DirichletProcessCardinalityPrior(strength=1.0)

    pmf = prior.cluster_count_pmf(3)

    assert pmf == pytest.approx((0.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 6.0))
    assert prior.expected_number_of_clusters(3) == pytest.approx(11.0 / 6.0)


def test_zero_discount_pitman_yor_matches_dirichlet_process_cluster_count_pmf():
    pitman_yor = PitmanYorCardinalityPrior(strength=1.7, discount=0.0)
    dirichlet = DirichletProcessCardinalityPrior(strength=1.7)

    assert pitman_yor.cluster_count_pmf(6) == pytest.approx(
        dirichlet.cluster_count_pmf(6)
    )


def test_pitman_yor_has_heavier_cluster_count_tail_than_matching_dirichlet_process():
    num_observations = 10
    min_clusters = 6
    pitman_yor = PitmanYorCardinalityPrior(strength=1.0, discount=0.5)
    dirichlet = DirichletProcessCardinalityPrior(strength=1.0)

    assert pitman_yor.cluster_count_tail_probability(
        num_observations,
        min_clusters,
    ) > dirichlet.cluster_count_tail_probability(num_observations, min_clusters)


def test_eppf_probability_matches_dirichlet_process_reference_case():
    prior = DirichletProcessCardinalityPrior(strength=1.0)

    probability = prior.exchangeable_partition_probability((2, 1))

    assert probability == pytest.approx(1.0 / 6.0)
    assert prior.log_exchangeable_partition_probability((2, 1)) == pytest.approx(
        math.log(1.0 / 6.0)
    )


def test_parameter_validation():
    with pytest.raises(ValueError):
        PitmanYorCardinalityPrior(strength=1.0, discount=1.0)
    with pytest.raises(ValueError):
        PitmanYorCardinalityPrior(strength=-0.5, discount=0.25)
    with pytest.raises(ValueError):
        DirichletProcessCardinalityPrior(strength=0.0)


def test_cluster_size_validation():
    prior = PitmanYorCardinalityPrior()

    with pytest.raises(ValueError):
        prior.predictive_probabilities((0, 1))
    with pytest.raises(TypeError):
        prior.predictive_probabilities((True, 1))
