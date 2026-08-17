import unittest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.filters import GoalConditionedReplayIMMFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestGoalConditionedReplayIMMProbabilityValidation(unittest.TestCase):
    @staticmethod
    def _construct(**kwargs):
        return GoalConditionedReplayIMMFilter(
            initial_state=(array([0.0]), eye(1)),
            candidate_goals=array([-1.0, 1.0]),
            **kwargs,
        )

    def test_rejects_nonfinite_transition_matrix_entries(self):
        invalid_matrices = {
            "goal_transition_matrix": array([[1.0, 0.0], [float("nan"), 1.0]]),
            "mode_transition_matrix": array([[1.0, 0.0], [float("nan"), 1.0]]),
        }

        for name, matrix in invalid_matrices.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{name} must contain only finite entries",
                ):
                    self._construct(**{name: matrix})

    def test_rejects_nonfinite_prior_entries(self):
        invalid_priors = (
            ("goal_prior", array([float("nan"), 1.0])),
            ("goal_prior", array([float("inf"), 1.0])),
            ("mode_prior", array([float("nan"), 1.0])),
            ("mode_prior", array([float("inf"), 1.0])),
        )

        for name, prior in invalid_priors:
            with self.subTest(name=name, prior=prior):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{name} must contain only finite entries",
                ):
                    self._construct(**{name: prior})
