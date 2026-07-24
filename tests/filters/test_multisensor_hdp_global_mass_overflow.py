import unittest

import numpy as np
from pyrecest.filters.multisensor_hdp_association import multisensor_hdp_association


class MultisensorHDPGlobalMassOverflowTest(unittest.TestCase):
    def test_maximum_finite_global_masses_preserve_relative_probability(self):
        largest = np.finfo(float).max

        with np.errstate(over="raise", invalid="raise"):
            result = multisensor_hdp_association(
                {"radar": np.zeros((1, 2))},
                global_target_weights=np.array([largest, largest]),
                global_birth_weight=largest,
                clutter_weights=0.0,
            )["radar"]

        np.testing.assert_allclose(
            result.probabilities,
            np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]]),
            rtol=0.0,
            atol=1e-15,
        )


if __name__ == "__main__":
    unittest.main()
