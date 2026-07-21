import types
import unittest

import numpy as np
from pyrecest.filters import adaptive_position_proposal_probability


class TestReplayGridEssGating(unittest.TestCase):
    def test_probability_one_still_respects_ess_threshold(self):
        high_ess_filter = types.SimpleNamespace(
            filter_state=types.SimpleNamespace(w=np.ones(4))
        )
        probability, ess_fraction = adaptive_position_proposal_probability(
            high_ess_filter, 1.0, 0.5
        )
        self.assertEqual(probability, 0.0)
        self.assertAlmostEqual(ess_fraction, 1.0)

        low_ess_filter = types.SimpleNamespace(
            filter_state=types.SimpleNamespace(w=np.array([1.0, 0.0, 0.0, 0.0]))
        )
        probability, ess_fraction = adaptive_position_proposal_probability(
            low_ess_filter, 1.0, 0.5
        )
        self.assertEqual(probability, 1.0)
        self.assertAlmostEqual(ess_fraction, 0.25)


if __name__ == "__main__":
    unittest.main()
