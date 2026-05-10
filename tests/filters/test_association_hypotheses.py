import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array
from pyrecest.filters import (
    AssociationHypothesis,
    CostThresholdGate,
    KalmanFilter,
    NISGate,
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
