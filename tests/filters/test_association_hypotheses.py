import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array
from pyrecest.filters import (
    AssociationHypothesis,
    CostThresholdGate,
    KalmanFilter,
    NISGate,
    ProbabilityThresholdGate,
    TopKGate,
    association_result_from_hypotheses,
    build_linear_gaussian_hypothesis_associator,
    filter_hypotheses,
    hypotheses_to_cost_matrix,
    linear_gaussian_association_hypotheses,
)


class AssociationHypothesesTest(unittest.TestCase):
    def test_linear_gaussian_hypotheses_store_nis_and_log_likelihood(self):
        tracks = [
            KalmanFilter((array([0.0]), array([[1.0]]))),
            KalmanFilter((array([10.0]), array([[1.0]]))),
        ]
        hypotheses = linear_gaussian_association_hypotheses(
            tracks,
            [array([0.5]), array([9.0])],
            array([[1.0]]),
            array([[1.0]]),
        )

        self.assertEqual(len(hypotheses), 4)
        close_hypothesis = next(
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.track_index == 0 and hypothesis.measurement_index == 0
        )
        self.assertTrue(allclose(close_hypothesis.innovation, array([0.5])))
        self.assertTrue(
            allclose(close_hypothesis.innovation_covariance, array([[2.0]]))
        )
        self.assertAlmostEqual(close_hypothesis.normalized_innovation_squared, 0.125)
        self.assertAlmostEqual(close_hypothesis.cost, 0.125)
        self.assertIsNotNone(close_hypothesis.log_likelihood)

    def test_nis_gate_filters_distant_gaussian_hypotheses(self):
        hypotheses = self._simple_hypotheses()
        gated = filter_hypotheses(hypotheses, NISGate(threshold=1.0))
        cost_matrix = hypotheses_to_cost_matrix(
            gated,
            num_tracks=2,
            num_measurements=2,
            missing_cost=99.0,
        )

        self.assertEqual(cost_matrix.shape, (2, 2))
        self.assertAlmostEqual(cost_matrix[0, 0], 0.125)
        self.assertEqual(cost_matrix[0, 1], 99.0)
        self.assertEqual(cost_matrix[1, 0], 99.0)
        self.assertAlmostEqual(cost_matrix[1, 1], 0.5)

    def test_nis_gate_rejects_invalid_threshold(self):
        for threshold in (np.nan, True, np.array([1.0])):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    NISGate(threshold=threshold)

    def test_nis_gate_validates_confidence_and_measurement_dimension(self):
        invalid_cases = [
            {"measurement_dim": 1, "confidence": np.nan, "message": "confidence"},
            {"measurement_dim": 1, "confidence": 1.0, "message": "confidence"},
            {"measurement_dim": True, "confidence": 0.95, "message": "measurement_dim"},
            {"measurement_dim": 1.5, "confidence": 0.95, "message": "measurement_dim"},
            {
                "measurement_dim": np.array([1]),
                "confidence": 0.95,
                "message": "measurement_dim",
            },
        ]
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, case["message"]):
                    NISGate(
                        measurement_dim=case["measurement_dim"],
                        confidence=case["confidence"],
                    )

    def test_cost_threshold_and_top_k_gates_are_generic(self):
        hypotheses = [
            AssociationHypothesis(0, 0, cost=1.0),
            AssociationHypothesis(0, 1, cost=2.0),
            AssociationHypothesis(1, 0, cost=4.0),
            AssociationHypothesis(1, 1, cost=3.0),
        ]

        thresholded = filter_hypotheses(hypotheses, CostThresholdGate(3.0))
        self.assertEqual(
            [(hyp.track_index, hyp.measurement_index) for hyp in thresholded],
            [(0, 0), (0, 1), (1, 1)],
        )

        top_per_track = filter_hypotheses(hypotheses, TopKGate(1, mode="track"))
        self.assertEqual(
            [(hyp.track_index, hyp.measurement_index) for hyp in top_per_track],
            [(0, 0), (1, 1)],
        )

    def test_cost_and_probability_gates_reject_nan_thresholds(self):
        for gate_factory in (CostThresholdGate, ProbabilityThresholdGate):
            with self.subTest(gate_factory=gate_factory):
                with self.assertRaisesRegex(ValueError, "threshold"):
                    gate_factory(np.nan)

        with self.assertRaisesRegex(ValueError, "threshold"):
            ProbabilityThresholdGate(np.nan, use_likelihood=True)

    def test_top_k_gate_rejects_invalid_k(self):
        for k in (True, 1.5, np.nan, np.inf, np.array([1])):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "k must"):
                    TopKGate(k)

    def test_top_k_gate_accepts_integer_scalar_k(self):
        gate = TopKGate(np.array(2.0))

        self.assertEqual(gate.k, 2)

    def test_probability_likelihood_gate_rejects_zero_threshold_and_allows_zero_probability_threshold(
        self,
    ):
        with self.assertRaises(ValueError):
            ProbabilityThresholdGate(0.0, use_likelihood=True)

        hypothesis = AssociationHypothesis(0, 0, probability=0.0)
        self.assertTrue(ProbabilityThresholdGate(0.0).accepts(hypothesis))

    def test_hypothesis_accepted_flag_must_be_boolean(self):
        rejected = AssociationHypothesis(0, 0, cost=1.0, accepted=np.bool_(False))

        self.assertFalse(rejected.accepted)
        self.assertEqual(filter_hypotheses([rejected]), [])

        with self.assertRaisesRegex(ValueError, "accepted"):
            AssociationHypothesis(0, 0, cost=1.0, accepted="False")

        with self.assertRaisesRegex(ValueError, "accepted"):
            AssociationHypothesis(0, 0, cost=1.0).with_acceptance("False")

    def test_callable_gate_result_must_be_boolean(self):
        hypotheses = [AssociationHypothesis(0, 0, cost=1.0)]

        with self.assertRaisesRegex(ValueError, "gate result"):
            filter_hypotheses(hypotheses, lambda _hypothesis: "False")

    def test_measurement_axis_auto_uses_measurement_dimension(self):
        state_covariance = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        measurement_matrix = array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tracks = [KalmanFilter((array([0.0, 0.0, 0.0]), state_covariance))]
        measurements = array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        hypotheses = linear_gaussian_association_hypotheses(
            tracks, measurements, measurement_matrix, state_covariance
        )

        self.assertEqual(len(hypotheses), 2)
        self.assertEqual(hypotheses[1].measurement_index, 1)
        self.assertTrue(allclose(hypotheses[1].innovation, array([1.0, 1.0, 1.0])))

        with self.assertRaises(ValueError):
            linear_gaussian_association_hypotheses(
                tracks, state_covariance, measurement_matrix, state_covariance
            )

    def test_hypotheses_solve_global_nearest_neighbor_assignment(self):
        hypotheses = filter_hypotheses(self._simple_hypotheses(), NISGate(1.0))
        association = association_result_from_hypotheses(
            hypotheses,
            num_tracks=2,
            num_measurements=2,
            unassigned_track_cost=10.0,
            missing_cost=99.0,
        )

        self.assertEqual(sorted(association.matches), [(0, 0), (1, 1)])
        self.assertEqual(association.unmatched_track_indices, [])
        self.assertEqual(association.unmatched_measurement_indices, [])

    def test_probability_likelihood_gate_rejects_zero_threshold(self):
        with self.assertRaises(ValueError):
            ProbabilityThresholdGate(0.0, use_likelihood=True)

    def test_default_hypothesis_assignment_does_not_select_missing_pair(self):
        hypotheses = [
            AssociationHypothesis(0, 0, cost=1.0),
        ]

        association = association_result_from_hypotheses(
            hypotheses,
            num_tracks=2,
            num_measurements=2,
        )

        self.assertEqual(association.matches, [(0, 0)])
        self.assertIn(1, association.unmatched_track_indices)
        self.assertIn(1, association.unmatched_measurement_indices)

    def test_linear_gaussian_hypothesis_associator_matches_expected_pairs(self):
        tracks = [
            KalmanFilter((array([0.0]), array([[1.0]]))),
            KalmanFilter((array([10.0]), array([[1.0]]))),
        ]
        associator = build_linear_gaussian_hypothesis_associator(
            array([[1.0]]),
            array([[1.0]]),
            unassigned_track_cost=10.0,
            gates=NISGate(1.0),
            missing_cost=99.0,
        )

        association = associator(tracks, [array([0.5]), array([9.0])])

        self.assertEqual(sorted(association.matches), [(0, 0), (1, 1)])
        self.assertEqual(association.unmatched_track_indices, [])
        self.assertEqual(association.unmatched_measurement_indices, [])

    @staticmethod
    def _simple_hypotheses():
        tracks = [
            KalmanFilter((array([0.0]), array([[1.0]]))),
            KalmanFilter((array([10.0]), array([[1.0]]))),
        ]
        return linear_gaussian_association_hypotheses(
            tracks,
            [array([0.5]), array([9.0])],
            array([[1.0]]),
            array([[1.0]]),
        )


if __name__ == "__main__":
    unittest.main()
