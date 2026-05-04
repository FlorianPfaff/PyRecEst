"""Unscented Kalman filter example using additive-noise model objects."""

# pylint: disable=import-error,no-name-in-module,no-member
from _filter_example_output import print_position_velocity_estimates
from pyrecest.backend import array, diag
from pyrecest.filters import UnscentedKalmanFilter
from pyrecest.models import (
    AdditiveNoiseMeasurementModel,
    AdditiveNoiseTransitionModel,
)


def constant_velocity_transition(state, dt):
    """Propagate a constant-velocity state by one time step."""
    position, velocity = state
    return array([position + dt * velocity, velocity])


def position_measurement(state):
    """Measure position from a `[position, velocity]` state."""
    position, _velocity = state
    return array([position])


def run_filter():
    """Run a UKF with reusable transition and measurement models."""
    dt = 1.0
    transition_model = AdditiveNoiseTransitionModel(
        transition_function=constant_velocity_transition,
        noise_distribution=diag(array([0.05, 0.01])),
    )
    measurement_model = AdditiveNoiseMeasurementModel(
        measurement_function=position_measurement,
        noise_distribution=array([[0.25]]),
    )

    measurements = [0.9, 2.0, 3.1, 3.9, 5.2]
    unscented_filter = UnscentedKalmanFilter(
        (array([0.0, 1.0]), diag(array([1.0, 1.0]))),
        dt=dt,
    )
    estimates = []

    for measurement in measurements:
        unscented_filter.predict_model(transition_model, dt=dt)
        unscented_filter.update_model(measurement_model, array([measurement]))
        estimates.append(unscented_filter.get_point_estimate())

    return measurements, estimates, unscented_filter.filter_state.C


def main():
    """Print the posterior estimates produced by the UKF loop."""
    measurements, estimates, final_covariance = run_filter()

    print_position_velocity_estimates(
        measurements, estimates, final_covariance=final_covariance
    )


if __name__ == "__main__":
    main()
