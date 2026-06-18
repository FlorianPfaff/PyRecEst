import unittest
from unittest.mock import patch

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, isinf, isnan, random, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GoalConditionedReplayIMMFilter

from .test_goal_conditioned_replay_common import (
    assert_association_likelihood_linear_positive,
)


class TestGoalConditionedReplayIMMFilter(unittest.TestCase):
    def test_constructor_rejects_jax_backend(self):
        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "JAX backend"):
                GoalConditionedReplayIMMFilter(
                    initial_state=(array([0.0, 0.0]), 0.01 * eye(2)),
                    candidate_goals=array([[1.0, 0.0]]),
                )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_replay_moves_velocity_toward_goal(self):
        random.seed(0)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(array([0.0, 0.0]), 0.01 * eye(2)),
            candidate_goals=array([[1.0, 0.0]]),
            dt=1.0,
            attraction_strength=1.0,
            velocity_decay=0.5,
            smooth_sys_noise_cov=0.01 * eye(4),
            jump_sys_noise_cov=0.10 * eye(4),
            jump_probability=0.0,
        )

        filt.predict_replay()
        velocity_estimate = filt.get_velocity_estimate()

        self.assertEqual(velocity_estimate.shape, (2,))
        self.assertGreater(velocity_estimate[0], 0.2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_position_update_returns_log_marginal_and_pulls_mean_to_measurement(self):
        random.seed(1)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(array([0.0, 0.0]), 0.02 * eye(2)),
            candidate_goals=array([[1.0, 0.0]]),
            dt=1.0,
            attraction_strength=1.0,
            velocity_decay=0.0,
            smooth_sys_noise_cov=0.01 * eye(4),
            jump_sys_noise_cov=0.10 * eye(4),
            jump_probability=0.0,
        )

        filt.predict_replay()
        log_marginal = filt.update_position(
            array([1.0, 0.0]),
            GaussianDistribution(zeros(2), 0.05 * eye(2)),
            return_log_marginal=True,
        )
        position_estimate = filt.get_position_estimate()

        self.assertFalse(isnan(log_marginal))
        self.assertFalse(isinf(log_marginal))
        self.assertGreater(position_estimate[0], 0.55)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_goal_update_reweights_goal_posterior(self):
        random.seed(2)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(array([0.0, 0.0]), 0.01 * eye(2)),
            candidate_goals=array([[-1.0, 0.0], [1.0, 0.0]]),
            goal_prior=array([0.5, 0.5]),
            dt=1.0,
            smooth_sys_noise_cov=0.01 * eye(4),
            jump_sys_noise_cov=0.10 * eye(4),
            jump_probability=0.0,
        )

        log_marginal = filt.update_goal(
            array([1.0, 0.0]),
            GaussianDistribution(zeros(2), 0.01 * eye(2)),
            return_log_marginal=True,
        )
        goal_probabilities = filt.goal_probabilities

        self.assertFalse(isnan(log_marginal))
        self.assertFalse(isinf(log_marginal))
        self.assertGreater(goal_probabilities[1], 0.95)
        self.assertGreater(goal_probabilities[1], goal_probabilities[0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_linear_association_likelihood_is_positive(self):
        random.seed(3)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(array([0.5, 0.0, 0.2, 0.0]), 0.05 * eye(4)),
            candidate_goals=array([[1.0, 0.0]]),
            dt=1.0,
            smooth_sys_noise_cov=0.01 * eye(4),
            jump_sys_noise_cov=0.10 * eye(4),
            jump_probability=0.0,
        )

        assert_association_likelihood_linear_positive(self, filt, state_dim=4)
