import unittest

import numpy as np
from pyrecest.filters.multisensor_hdp_association import multisensor_hdp_association


class MultisensorHDPSensorPriorLogDomainTest(unittest.TestCase):
    def test_large_finite_counts_and_concentration_preserve_prior_ratio(self):
        largest = np.finfo(float).max

        with np.errstate(over="raise", invalid="raise"):
            result = multisensor_hdp_association(
                {"radar": np.zeros((1, 2))},
                global_target_weights=np.ones(2),
                global_birth_weight=0.0,
                sensor_target_counts={"radar": np.array([largest, 0.0])},
                sensor_concentrations={"radar": largest},
                clutter_weights=0.0,
            )["radar"]

        np.testing.assert_allclose(
            result.probabilities,
            np.array([[0.75, 0.25, 0.0, 0.0]]),
            rtol=0.0,
            atol=64 * np.finfo(float).eps,
        )
        self.assertTrue(np.all(np.isfinite(result.log_weights[0, :2])))

    def test_tiny_concentration_and_base_mass_do_not_erase_target(self):
        small = 1e-200
        target_log_likelihood = -np.log(small)

        with np.errstate(under="raise", invalid="raise"):
            result = multisensor_hdp_association(
                {"radar": np.array([[target_log_likelihood]])},
                global_target_weights=np.array([small]),
                global_birth_weight=1.0,
                sensor_target_counts=0.0,
                sensor_concentrations=small,
                clutter_weights=0.0,
            )["radar"]

        np.testing.assert_allclose(
            result.probabilities,
            np.array([[0.5, 0.5, 0.0]]),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertTrue(np.isfinite(result.log_weights[0, 0]))


if __name__ == "__main__":
    unittest.main()
