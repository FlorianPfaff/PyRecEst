import unittest

from pyrecest.filters import (
    AssociationHypothesis,
    CostThresholdGate,
    ProbabilityThresholdGate,
    TopKGate,
    filter_hypotheses,
)


class AssociationGateMonotonicityTest(unittest.TestCase):
    def test_later_scalar_gate_does_not_reaccept_prior_rejection(self):
        hypotheses = [
            AssociationHypothesis(0, 0, cost=1.0, probability=0.0),
            AssociationHypothesis(0, 1, cost=2.0, probability=1.0),
        ]

        gated = filter_hypotheses(
            hypotheses,
            [ProbabilityThresholdGate(0.5), CostThresholdGate(3.0)],
            accepted_only=False,
        )

        self.assertFalse(gated[0].accepted)
        self.assertEqual(gated[0].reason, "ProbabilityThresholdGate")
        self.assertTrue(gated[1].accepted)

    def test_later_top_k_gate_only_ranks_active_hypotheses(self):
        hypotheses = [
            AssociationHypothesis(0, 0, cost=1.0, probability=0.0),
            AssociationHypothesis(0, 1, cost=10.0, probability=1.0),
        ]

        accepted = filter_hypotheses(
            hypotheses,
            [ProbabilityThresholdGate(0.5), TopKGate(1, mode="track")],
        )

        self.assertEqual(
            [
                (hypothesis.track_index, hypothesis.measurement_index)
                for hypothesis in accepted
            ],
            [(0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
