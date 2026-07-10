import unittest
from unittest.mock import patch

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, diag, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.kernel_sme_filter import KernelSMEFilter


class TestKernelSMEFilterKernelCovariance(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Kernel SME test-point generation is not supported on this backend",
    )
    def test_test_points_treat_kernel_width_as_covariance(self):
        test_points = KernelSMEFilter.gen_test_points(array([[10.0]]), 4.0)

        npt.assert_allclose(test_points, array([[10.0, 12.0, 8.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Kernel SME updates are not supported on this backend",
    )
    def test_update_uses_mean_measurement_variance_as_kernel_covariance(self):
        tracker = KernelSMEFilter(
            [GaussianDistribution(array([0.0]), eye(1))], log_estimates=False
        )
        measurements = array([[0.0]])
        measurement_covariance = diag(array([4.0]))
        test_points = array([[0.0]])

        with (
            patch.object(
                KernelSMEFilter,
                "gen_test_points",
                return_value=test_points,
            ) as gen_test_points,
            patch.object(
                KernelSMEFilter,
                "calc_pseudo_meas",
                return_value=zeros(1),
            ) as calc_pseudo_meas,
            patch.object(
                KernelSMEFilter,
                "calc_moments",
                return_value=(zeros(1), eye(1), zeros((1, 1))),
            ) as calc_moments,
        ):
            tracker.update_linear(
                measurements,
                eye(1),
                measurement_covariance,
            )

        npt.assert_allclose(gen_test_points.call_args.args[1], 4.0)
        npt.assert_allclose(calc_pseudo_meas.call_args.args[2], 4.0)
        npt.assert_allclose(calc_moments.call_args.args[5], 4.0)


if __name__ == "__main__":
    unittest.main()
