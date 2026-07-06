import math
import unittest

import numpy.testing as npt

from pyrecest.filters.nonparametric_cardinality import (
    DirichletProcessCardinalityPrior,
    PitmanYorBirthProbability,
    PitmanYorProcessCardinalityPrior,
)


class NonparametricCardinalityPriorTest(unittest.TestCase):
    def test_pitman_yor_predictive_probabilities_are_heavier_tailed_than_dp(self):
        pitman_yor_prior = PitmanYorProcessCardinalityPrior(discount=0.5, strength=2.0)
        dirichlet_prior = DirichletProcessCardinalityPrior(concentration=2.0)

        pitman_yor_probabilities = pitman_yor_prior.predictive_assignment_probabilities([3, 2])
        dirichlet_probabilities = dirichlet_prior.predictive_assignment_probabilities([3, 2])

        npt.assert_allclose(
            pitman_yor_probabilities,
            (2.5 / 7.0, 1.5 / 7.0, 3.0 / 7.0),
        )
        self.assertAlmostEqual(sum(pitman_yor_probabilities), 1.0)
        self.assertGreater(
            pitman_yor_prior.predictive_new_cluster_probability([3, 2]),
            dirichlet_prior.predictive_new_cluster_probability([3, 2]),
        )

    def test_discount_zero_matches_dirichlet_process(self):
        pitman_yor_prior = PitmanYorProcessCardinalityPrior(discount=0.0, strength=1.7)
        dirichlet_prior = DirichletProcessCardinalityPrior(concentration=1.7)

        npt.assert_allclose(
            pitman_yor_prior.predictive_assignment_probabilities([4, 1, 1]),
            dirichlet_prior.predictive_assignment_probabilities([4, 1, 1]),
        )
        self.assertTrue(pitman_yor_prior.is_dirichlet_process)
        self.assertEqual(dirichlet_prior.concentration, 1.7)

    def test_expected_cluster_count_grows_faster_with_positive_discount(self):
        pitman_yor_prior = PitmanYorProcessCardinalityPrior(discount=0.4, strength=1.0)
        dirichlet_prior = DirichletProcessCardinalityPrior(concentration=1.0)

        self.assertGreater(
            pitman_yor_prior.expected_number_of_clusters(50),
            dirichlet_prior.expected_number_of_clusters(50),
        )
        self.assertAlmostEqual(dirichlet_prior.expected_number_of_clusters(0), 0.0)
        self.assertAlmostEqual(dirichlet_prior.expected_number_of_clusters(1), 1.0)

    def test_log_eppf_matches_simple_dirichlet_process_value(self):
        dirichlet_prior = DirichletProcessCardinalityPrior(concentration=2.0)

        self.assertAlmostEqual(math.exp(dirichlet_prior.log_eppf([2, 1])), 1.0 / 6.0)
        self.assertAlmostEqual(dirichlet_prior.eppf([]), 1.0)

    def test_expected_additional_clusters_uses_initial_count_state(self):
        prior = PitmanYorProcessCardinalityPrior(discount=0.5, strength=1.0)

        self.assertAlmostEqual(
            prior.expected_additional_clusters(
                additional_observations=1,
                initial_observations=5,
                initial_clusters=3,
            ),
            (1.0 + 0.5 * 3.0) / (1.0 + 5.0),
        )

    def test_pitman_yor_birth_probability_encourages_bursty_births_relative_to_dp(self):
        pitman_yor_birth = PitmanYorBirthProbability(
            discount=0.5,
            strength=1.0,
            base_birth_existence_probability=0.8,
        )
        dirichlet_birth = PitmanYorBirthProbability(
            discount=0.0,
            strength=1.0,
            base_birth_existence_probability=0.8,
        )

        self.assertAlmostEqual(
            pitman_yor_birth(num_existing_components=0, num_new_births=0),
            0.8,
        )
        self.assertAlmostEqual(
            pitman_yor_birth(num_existing_components=0, num_new_births=1),
            0.6,
        )
        self.assertGreater(
            pitman_yor_birth(num_existing_components=5, num_new_births=0),
            dirichlet_birth(num_existing_components=5, num_new_births=0),
        )

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            PitmanYorProcessCardinalityPrior(discount=-0.1, strength=1.0)
        with self.assertRaises(ValueError):
            PitmanYorProcessCardinalityPrior(discount=1.0, strength=1.0)
        with self.assertRaises(ValueError):
            PitmanYorProcessCardinalityPrior(discount=0.0, strength=0.0)
        with self.assertRaises(ValueError):
            PitmanYorBirthProbability(prior_observation_count=0, prior_cluster_count=1)
        with self.assertRaises(ValueError):
            PitmanYorBirthProbability(minimum_probability=0.9, maximum_probability=0.1)

    def test_invalid_cluster_sizes_are_rejected(self):
        prior = PitmanYorProcessCardinalityPrior(discount=0.2, strength=1.0)

        with self.assertRaises(ValueError):
            prior.predictive_assignment_probabilities([2, 0])
        with self.assertRaises(ValueError):
            prior.predictive_new_cluster_probability_from_counts(2, 3)


if __name__ == "__main__":
    unittest.main()
