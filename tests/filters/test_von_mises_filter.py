import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import VonMisesDistribution
from pyrecest.filters.von_mises_filter import VonMisesFilter


class TestVonMisesFilter(unittest.TestCase):
    def setUp(self):
        self.curr_filter = VonMisesFilter()
        self.vm_prior = VonMisesDistribution(2.1, 1.3)

    def test_setting_state(self):
        vm = self.vm_prior
        self.curr_filter.filter_state = vm
        vm1 = self.curr_filter.filter_state
        self.assertIsInstance(vm1, VonMisesDistribution)
        self.assertEqual(vm.mu, vm1.mu)
        self.assertEqual(vm.kappa, vm1.kappa)

    def test_setting_state_validation_errors_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "VonMisesDistribution"):
            self.curr_filter.filter_state = object()

        invalid_mu = copy.deepcopy(self.vm_prior)
        invalid_mu.mu = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.curr_filter.filter_state = invalid_mu

        invalid_kappa = copy.deepcopy(self.vm_prior)
        invalid_kappa.kappa = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.curr_filter.filter_state = invalid_kappa

        negative_kappa = copy.deepcopy(self.vm_prior)
        negative_kappa.kappa = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.curr_filter.filter_state = negative_kappa

    def test_prediction(self):
        sysnoise = VonMisesDistribution(0, 0.3)

        self.curr_filter.filter_state = self.vm_prior
        self.curr_filter.predict_identity(sysnoise)
        self.assertIsInstance(self.curr_filter.filter_state, VonMisesDistribution)
        self.assertEqual(self.curr_filter.filter_state.mu, 2.1)
        self.assertLess(self.curr_filter.filter_state.kappa, 1.3)

    def test_prediction_rejects_invalid_noise(self):
        self.curr_filter.filter_state = self.vm_prior
        with self.assertRaisesRegex(ValueError, "system noise"):
            self.curr_filter.predict_identity(object())

        invalid_noise = copy.deepcopy(self.vm_prior)
        invalid_noise.kappa = -1.0
        with self.assertRaisesRegex(ValueError, "system noise"):
            self.curr_filter.predict_identity(invalid_noise)

    def test_update(self):
        meas_noise = VonMisesDistribution(0.0, 1.3)
        meas = array(1.1)

        self.curr_filter.filter_state = self.vm_prior
        self.curr_filter.update_identity(meas_noise, meas)
        self.assertIsInstance(self.curr_filter.filter_state, VonMisesDistribution)
        npt.assert_allclose(
            self.curr_filter.get_point_estimate(), (self.vm_prior.mu + meas) / 2.0
        )
        self.assertGreater(self.curr_filter.filter_state.kappa, 1.3)

    def test_update_accepts_python_float_measurement(self):
        meas_noise = VonMisesDistribution(0.0, 1.3)

        self.curr_filter.filter_state = self.vm_prior
        self.curr_filter.update_identity(meas_noise, 1.1)

        self.assertIsInstance(self.curr_filter.filter_state, VonMisesDistribution)

    def test_update_accepts_default_measurement(self):
        meas_noise = VonMisesDistribution(0.0, 1.3)

        self.curr_filter.filter_state = self.vm_prior
        self.curr_filter.update_identity(meas_noise)

        self.assertIsInstance(self.curr_filter.filter_state, VonMisesDistribution)

    def test_update_rejects_invalid_inputs(self):
        self.curr_filter.filter_state = self.vm_prior
        with self.assertRaisesRegex(ValueError, "measurement noise"):
            self.curr_filter.update_identity(object(), 1.1)
        with self.assertRaisesRegex(ValueError, "scalar"):
            self.curr_filter.update_identity(VonMisesDistribution(0.0, 1.3), [1.1, 2.2])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.curr_filter.update_identity(
                VonMisesDistribution(0.0, 1.3), float("nan")
            )

        invalid_measurements = (
            True,
            False,
            1.0 + 0.0j,
            "1.1",
            b"1.1",
            [True],
            array(True),
            array(1.0 + 0.0j),
        )
        for measurement in invalid_measurements:
            with self.subTest(measurement=measurement):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    self.curr_filter.update_identity(
                        VonMisesDistribution(0.0, 1.3), measurement
                    )


if __name__ == "__main__":
    unittest.main()
