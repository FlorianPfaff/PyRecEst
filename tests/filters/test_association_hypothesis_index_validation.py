import unittest

from pyrecest.filters import (
    AssociationHypothesis,
    hypotheses_to_cost_matrix,
    hypotheses_to_log_likelihood_matrix,
    hypotheses_to_probability_matrix,
)


class AssociationHypothesisIndexValidationTest(unittest.TestCase):
    def test_dense_converters_reject_negative_track_indices(self):
        hypotheses = [
            AssociationHypothesis(
                -1,
                0,
                cost=1.0,
                log_likelihood=0.0,
                probability=0.5,
            )
        ]

        for converter in (
            hypotheses_to_cost_matrix,
            hypotheses_to_log_likelihood_matrix,
            hypotheses_to_probability_matrix,
        ):
            with self.subTest(converter=converter.__name__):
                with self.assertRaisesRegex(ValueError, "hypothesis.track_index"):
                    converter(hypotheses, num_tracks=2, num_measurements=1)

    def test_dense_converters_reject_negative_measurement_indices(self):
        hypotheses = [
            AssociationHypothesis(
                0,
                -1,
                cost=1.0,
                log_likelihood=0.0,
                probability=0.5,
            )
        ]

        for converter in (
            hypotheses_to_cost_matrix,
            hypotheses_to_log_likelihood_matrix,
            hypotheses_to_probability_matrix,
        ):
            with self.subTest(converter=converter.__name__):
                with self.assertRaisesRegex(ValueError, "hypothesis.measurement_index"):
                    converter(hypotheses, num_tracks=1, num_measurements=2)


if __name__ == "__main__":
    unittest.main()
