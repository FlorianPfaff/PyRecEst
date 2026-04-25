"""Basic Kalman filter example for a one-dimensional constant-velocity model."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter


def run_filter():
    dt = 1.0
    system_matrix = array([[1.0, dt], [0.0, 1.0]])
    measurement_matrix = array([[1.0, 0.0]])
    system_noise_cov = diag(array([0.05, 0.01]))
    measurement_noise_cov = array([[0.25]])

    measurements = [0.9, 2.0, 3.1, 3.9, 5.2]
    kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))
    estimates = []

    for measurement in measurements:
        kalman_filter.predict_linear(system_matrix, system_noise_cov)
        kalman_filter.update_linear(
            array([measurement]), measurement_matrix, measurement_noise_cov
        )
        estimates.append(kalman_filter.get_point_estimate())

    return measurements, estimates, kalman_filter.filter_state.C


def main():
    measurements, estimates, final_covariance = run_filter()

    print("step  measurement  position  velocity")
    for step, (measurement, estimate) in enumerate(
        zip(measurements, estimates), start=1
    ):
        position, velocity = estimate
        print(
            f"{step:>4}  {measurement:>11.2f}  "
            f"{float(position):>8.3f}  {float(velocity):>8.3f}"
        )

    print("\nFinal covariance:")
    print(final_covariance)


if __name__ == "__main__":
    main()
