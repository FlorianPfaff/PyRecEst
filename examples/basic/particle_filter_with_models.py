"""Particle filter example using reusable transition and likelihood models."""

# pylint: disable=import-error,no-name-in-module,no-member
from _filter_example_output import print_position_velocity_estimates
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import EuclideanParticleFilter
from pyrecest.models import (
    LikelihoodMeasurementModel,
    SampleableTransitionModel,
)


def make_constant_velocity_sampler(dt, process_noise):
    """Create a sampler for a constant-velocity transition model."""

    def sample_next(states):
        propagated_states = states @ array([[1.0, 0.0], [dt, 1.0]])
        return propagated_states + process_noise.sample(states.shape[0])

    return sample_next


def make_position_likelihood(measurement_noise_cov):
    """Create a position-only likelihood function."""

    def likelihood(measurement, states):
        noise_model = GaussianDistribution(
            array([measurement[0]]), measurement_noise_cov
        )
        return noise_model.pdf(states[..., 0])

    return likelihood


def weighted_point_estimate(particle_filter):
    """Return the particle-filter point estimate."""
    return particle_filter.get_point_estimate()


def run_filter():
    """Run a Euclidean particle filter with model objects."""
    dt = 1.0
    num_particles = 500
    process_noise = GaussianDistribution(
        array([0.0, 0.0]),
        diag(array([0.05, 0.01])),
    )
    initial_distribution = GaussianDistribution(
        array([0.0, 1.0]),
        diag(array([1.0, 1.0])),
    )

    transition_model = SampleableTransitionModel(
        sample_next=make_constant_velocity_sampler(dt, process_noise),
    )
    measurement_model = LikelihoodMeasurementModel(
        likelihood=make_position_likelihood(array([[0.25]])),
    )

    measurements = [0.9, 2.0, 3.1, 3.9, 5.2]
    particle_filter = EuclideanParticleFilter(num_particles, dim=2)
    particle_filter.filter_state = initial_distribution
    estimates = []

    for measurement in measurements:
        particle_filter.predict_model(transition_model)
        particle_filter.update_model(measurement_model, array([measurement]))
        estimates.append(weighted_point_estimate(particle_filter))

    return measurements, estimates


def main():
    """Print the posterior estimates produced by the particle-filter loop."""
    measurements, estimates = run_filter()

    print_position_velocity_estimates(measurements, estimates)


if __name__ == "__main__":
    main()
