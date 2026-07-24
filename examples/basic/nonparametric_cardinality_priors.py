"""Compare Dirichlet-process and Pitman--Yor cardinality priors.

This example focuses on the prior used for target/cardinality structure rather
than a complete tracker.  A positive Pitman--Yor discount keeps more probability
mass on additional new clusters than a Dirichlet process with the same strength,
which is useful for prototyping bursty-birth or many-short-track regimes.
"""

from pyrecest.filters.nonparametric_cardinality import (
    DirichletProcessCardinalityPrior,
    PitmanYorBirthProbability,
    PitmanYorProcessCardinalityPrior,
)


def main():
    cluster_sizes = [3, 2]
    dirichlet_prior = DirichletProcessCardinalityPrior(concentration=2.0)
    pitman_yor_prior = PitmanYorProcessCardinalityPrior(discount=0.5, strength=2.0)

    print("cluster sizes:", cluster_sizes)
    print(
        "DP assignment probabilities: ",
        dirichlet_prior.predictive_assignment_probabilities(cluster_sizes),
    )
    print(
        "PYP assignment probabilities:",
        pitman_yor_prior.predictive_assignment_probabilities(cluster_sizes),
    )
    print(
        "DP expected clusters after 50 observations: ",
        round(dirichlet_prior.expected_number_of_clusters(50), 3),
    )
    print(
        "PYP expected clusters after 50 observations:",
        round(pitman_yor_prior.expected_number_of_clusters(50), 3),
    )

    birth_probability = PitmanYorBirthProbability(
        discount=0.5,
        strength=1.0,
        base_birth_existence_probability=0.8,
    )
    print(
        "first unassigned-measurement birth probability:",
        birth_probability(num_existing_components=0, num_new_births=0),
    )
    print(
        "second burst birth probability:             ",
        birth_probability(num_existing_components=0, num_new_births=1),
    )


if __name__ == "__main__":
    main()
