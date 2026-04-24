import unittest

import pyrecest.backend
from pyrecest.filters.multi_hypothesis_tracker import MultiHypothesisTracker


class MultiHypothesisTrackerEnumerationTest(unittest.TestCase):
    @staticmethod
    def _brute_force_candidate_assignments(candidate_measurements, base_log_score):
        all_measurements = sorted(
            {
                measurement_index
                for track_candidates in candidate_measurements
                for measurement_index, _ in track_candidates
            }
        )
        gain_lookup = [
            dict(track_candidates) for track_candidates in candidate_measurements
        ]
        solutions = []

        def recurse(track_index, used_measurements, current_assignment, current_gain):
            if track_index == len(candidate_measurements):
                solutions.append(
                    (base_log_score + current_gain, tuple(current_assignment))
                )
                return

            recurse(
                track_index + 1,
                used_measurements,
                current_assignment + [-1],
                current_gain,
            )
            for measurement_index in all_measurements:
                if measurement_index in used_measurements:
                    continue
                if measurement_index not in gain_lookup[track_index]:
                    continue
                recurse(
                    track_index + 1,
                    used_measurements | {measurement_index},
                    current_assignment + [measurement_index],
                    current_gain + gain_lookup[track_index][measurement_index],
                )

        recurse(0, set(), [], 0.0)
        unique_solutions = []
        seen_assignments = set()
        for solution in sorted(solutions, key=lambda item: (-item[0], item[1])):
            if solution[1] in seen_assignments:
                continue
            seen_assignments.add(solution[1])
            unique_solutions.append(solution)
        return unique_solutions

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_enumeration_matches_bruteforce(self):
        tracker = MultiHypothesisTracker.__new__(MultiHypothesisTracker)
        tracker.association_param = {"max_hypotheses_per_global_hypothesis": 5}
        candidate_measurements = [
            [(0, 3.0), (1, 0.4)],
            [(0, 2.2), (2, 1.1)],
            [(1, 1.3)],
        ]
        base_log_score = -4.5

        actual = tracker._enumerate_candidate_assignments(  # pylint: disable=protected-access
            candidate_measurements,
            base_log_score,
        )
        expected = self._brute_force_candidate_assignments(
            candidate_measurements,
            base_log_score,
        )[: len(actual)]

        self.assertEqual(actual, expected)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Not supported on the JAX backend",
    )
    def test_enumeration_without_candidates_returns_all_missed_detections(self):
        tracker = MultiHypothesisTracker.__new__(MultiHypothesisTracker)
        tracker.association_param = {"max_hypotheses_per_global_hypothesis": 3}

        actual = tracker._enumerate_candidate_assignments(
            [[], []], -2.0
        )  # pylint: disable=protected-access

        self.assertEqual(actual, [(-2.0, (-1, -1))])


if __name__ == "__main__":
    unittest.main()
