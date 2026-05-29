import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, cos, sin
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters.von_mises_fisher_filter import VonMisesFisherFilter


class VMFFilterTest(unittest.TestCase):
    def setUp(self):
        """Initial setup for each test."""
        self.filter = VonMisesFisherFilter()
        self.phi = 0.3
        self.mu = array([cos(self.phi), sin(self.phi)])
        self.kappa = 0.7
        self.vmf = VonMisesFisherDistribution(self.mu, self.kappa)

    def test_VMFFilter2d(self):
        """Test VonMisesFisherFilter in 2d."""
        self.filter.filter_state = self.vmf
        vmf_result = self.filter.filter_state
        self.assertIsInstance(vmf_result, VonMisesFisherDistribution)
        self.assertTrue(allclose(self.vmf.mu, vmf_result.mu))
        self.assertEqual(self.vmf.kappa, vmf_result.kappa)

    def test_set_state_validation_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "VonMisesFisherDistribution"):
            self.filter.filter_state = object()

        invalid_mu = copy.deepcopy(self.vmf)
        invalid_mu.mu = array([float("nan"), 0.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.filter.filter_state = invalid_mu

        nonunit_mu = copy.deepcopy(self.vmf)
        nonunit_mu.mu = array([2.0, 0.0])
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.filter.filter_state = nonunit_mu

        invalid_kappa = copy.deepcopy(self.vmf)
        invalid_kappa.kappa = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.filter.filter_state = invalid_kappa

        negative_kappa = copy.deepcopy(self.vmf)
        negative_kappa.kappa = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.filter.filter_state = negative_kappa

    def test_prediction_identity(self):
        """Test prediction identity."""
        self.filter.filter_state = self.vmf
        noise_distribution = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)
        self.filter.predict_identity(noise_distribution)
        self.assertLess(self.filter.filter_state.kappa, 0.5)
        # Add other assertions and logic here

    def test_predict_identity_rejects_invalid_noise(self):
        self.filter.filter_state = self.vmf
        with self.assertRaisesRegex(ValueError, "system noise"):
            self.filter.predict_identity(object())
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.filter.predict_identity(
                VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 0.9)
            )
        with self.assertRaisesRegex(ValueError, "zonal"):
            self.filter.predict_identity(
                VonMisesFisherDistribution(array([1.0, 0.0]), 0.9)
            )

    def test_update_identity(self):
        """Test update identity."""
        self.filter.filter_state = self.vmf
        noise_distribution = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)
        self.filter.update_identity(noise_distribution, self.vmf.mu)
        vmf_updated_identity = self.filter.filter_state
        self.assertIsInstance(vmf_updated_identity, VonMisesFisherDistribution)
        npt.assert_allclose(self.vmf.mu, vmf_updated_identity.mu, rtol=5e-7)
        self.assertGreaterEqual(vmf_updated_identity.kappa, self.vmf.kappa)

    def test_update_identity_accepts_array_like_measurement_without_mutating_noise(
        self,
    ):
        self.filter.filter_state = self.vmf
        noise_distribution = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)
        original_noise_mu = copy.deepcopy(noise_distribution.mu)

        self.filter.update_identity(
            noise_distribution,
            [float(cos(self.phi)), float(sin(self.phi))],
        )

        npt.assert_allclose(noise_distribution.mu, original_noise_mu)
        npt.assert_allclose(self.filter.filter_state.mu, self.vmf.mu, rtol=5e-7)

    def test_update_identity_rejects_invalid_inputs(self):
        self.filter.filter_state = self.vmf
        with self.assertRaisesRegex(ValueError, "measurement noise"):
            self.filter.update_identity(object(), [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.filter.update_identity(
                VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 0.9),
                [1.0, 0.0],
            )
        with self.assertRaisesRegex(ValueError, "zonal"):
            self.filter.update_identity(
                VonMisesFisherDistribution(array([1.0, 0.0]), 0.9),
                [1.0, 0.0],
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            self.filter.update_identity(
                VonMisesFisherDistribution(array([0.0, 1.0]), 0.9),
                [1.0, 0.0, 0.0],
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            self.filter.update_identity(
                VonMisesFisherDistribution(array([0.0, 1.0]), 0.9),
                [float("nan"), 0.0],
            )
        with self.assertRaisesRegex(ValueError, "unit vector"):
            self.filter.update_identity(
                VonMisesFisherDistribution(array([0.0, 1.0]), 0.9),
                [2.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
