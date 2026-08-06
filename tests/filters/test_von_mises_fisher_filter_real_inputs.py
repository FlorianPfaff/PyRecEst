"""Regression tests for real-valued von Mises-Fisher filter inputs."""

import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters.von_mises_fisher_filter import VonMisesFisherFilter


class VonMisesFisherFilterRealInputTest(unittest.TestCase):
    def test_update_rejects_complex_measurement_without_mutating_state(self):
        current_filter = VonMisesFisherFilter()
        current_filter.filter_state = VonMisesFisherDistribution(array([1.0, 0.0]), 0.7)
        measurement_noise = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)
        original_mean = copy.deepcopy(current_filter.filter_state.mu)
        original_kappa = current_filter.filter_state.kappa

        with self.assertRaisesRegex(ValueError, "real-valued"):
            current_filter.update_identity(
                measurement_noise,
                array([1.0j, 0.0]),
            )

        npt.assert_allclose(current_filter.filter_state.mu, original_mean)
        self.assertEqual(current_filter.filter_state.kappa, original_kappa)

    def test_filter_state_rejects_complex_mean_direction(self):
        current_filter = VonMisesFisherFilter()
        invalid_state = VonMisesFisherDistribution(array([1.0, 0.0]), 0.7)
        invalid_state.mu = array([1.0j, 0.0])

        with self.assertRaisesRegex(ValueError, "real-valued"):
            current_filter.filter_state = invalid_state


if __name__ == "__main__":
    unittest.main()
