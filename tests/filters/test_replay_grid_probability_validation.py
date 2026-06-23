import unittest

import numpy as np

from pyrecest.filters import adaptive_position_proposal_probability


class TestReplayGridProposalProbabilityValidation(unittest.TestCase):
    def test_adaptive_position_proposal_probability_rejects_malformed_probabilities(
        self,
    ):
        weights = np.ones(4)
        cases = (
            ("base_probability", True, None),
            ("base_probability", np.asarray([0.5]), None),
            ("ess_threshold", 0.5, False),
            ("ess_threshold", 0.5, np.asarray([0.5])),
        )

        for expected_name, base_probability, ess_threshold in cases:
            with self.subTest(
                base_probability=base_probability,
                ess_threshold=ess_threshold,
            ):
                with self.assertRaisesRegex(ValueError, expected_name):
                    adaptive_position_proposal_probability(
                        weights,
                        base_probability,
                        ess_threshold,
                    )


if __name__ == "__main__":
    unittest.main()
