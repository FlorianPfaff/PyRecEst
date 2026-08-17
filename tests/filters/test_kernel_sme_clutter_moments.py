import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array, to_numpy
from pyrecest.filters.kernel_sme_filter import KernelSMEFilter

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="KernelSMEFilter moments are not supported on this backend",
)


def _normal_pdf(value, mean, covariance):
    delta = value - mean
    dimension = delta.size
    normalizer = np.sqrt((2.0 * np.pi) ** dimension * np.linalg.det(covariance))
    exponent = -0.5 * delta @ np.linalg.solve(covariance, delta)
    return np.exp(exponent) / normalizer


def test_clutter_only_covariance_does_not_double_count_clutter_products():
    false_alarm_rate = 2.0
    kernel_width = 0.7
    clutter_covariance = np.array([[1.5, 0.2], [0.2, 1.2]])
    test_points_np = np.array([[0.25, -0.5], [-0.1, 0.4]])
    identity = np.eye(2)

    mu_s, sigma_s, sigma_xs = KernelSMEFilter.calc_moments(
        x_prior=array([0.0, 0.0]),
        C_prior=array(identity),
        measurement_matrix=array(identity),
        covMatMeas=array(0.2 * identity),
        testPoints=array(test_points_np),
        kernel_width=kernel_width,
        n_targets=1,
        falseAlarmRate=false_alarm_rate,
        clutterCov=array(clutter_covariance),
        lambdaMultimeas=0.0,
    )

    clutter_kernel_covariance = clutter_covariance + kernel_width * identity
    clutter_pdf = np.array(
        [
            _normal_pdf(
                test_points_np[:, index],
                np.zeros(2),
                clutter_kernel_covariance,
            )
            for index in range(test_points_np.shape[1])
        ]
    )
    expected_mu = false_alarm_rate * clutter_pdf
    expected_sigma = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            point_i = test_points_np[:, i]
            point_j = test_points_np[:, j]
            kernel_between = _normal_pdf(
                point_i,
                point_j,
                2.0 * kernel_width * identity,
            )
            clutter_mid_pdf = _normal_pdf(
                0.5 * (point_i + point_j),
                np.zeros(2),
                clutter_kernel_covariance,
            )
            expected_sigma[i, j] = false_alarm_rate * kernel_between * clutter_mid_pdf

    npt.assert_allclose(to_numpy(mu_s), expected_mu, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(to_numpy(sigma_s), expected_sigma, rtol=1e-7, atol=1e-10)
    npt.assert_allclose(to_numpy(sigma_xs), np.zeros((2, 2)), atol=1e-12)
