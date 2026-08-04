import unittest

import numpy as np

# pylint: disable=no-name-in-module
from pyrecest.filters import adaptive_position_proposal_probability


class TestReplayGridProbabilityRealControls(unittest.TestCase):
    def test_rejects_complex_probability_controls(self):
        invalid_probabilities = (
            np.complex64(0.5 + 1.0j),
            np.complex128(0.5 + 0.0j),
            np.array(0.5 + 1.0j, dtype=np.complex64),
            np.array(np.complex64(0.5 + 1.0j), dtype=object),
            np.array(complex(0.5, 1.0), dtype=object),
        )

        for probability in invalid_probabilities:
            with self.subTest(field="base_probability", value=repr(probability)):
                with self.assertRaisesRegex(ValueError, "base_probability"):
                    adaptive_position_proposal_probability(
                        [1.0], probability, None
                    )
            with self.subTest(field="ess_threshold", value=repr(probability)):
                with self.assertRaisesRegex(ValueError, "ess_threshold"):
                    adaptive_position_proposal_probability(
                        [1.0], 0.5, probability
                    )

    def test_accepts_real_numpy_probability_scalars(self):
        probability, ess_fraction = adaptive_position_proposal_probability(
            [1.0, 0.0], np.float32(0.5), np.float64(0.75)
        )

        self.assertAlmostEqual(probability, 0.5)
        self.assertAlmostEqual(ess_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
