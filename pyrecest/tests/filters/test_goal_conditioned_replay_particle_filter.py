import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, isinf, isnan, logical_or, random, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GoalConditionedReplayParticleFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
)
class TestGoalConditionedReplayParticleFilter(unittest.TestCase):
    def test_predict_replay_moves_velocity_toward_goal(self):
        random.seed(0)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=256,
            spatial_dim=2,
            dt=1.0,
            alpha=0.5,
            beta=1.0,
            process_noise=GaussianDistribution(zeros(2), 0.01 * eye(2)),
            goal_noise=GaussianDistribution(zeros(2), 0.001 * eye(2)),
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )

        filt.predict_replay()
        velocity_estimate = filt.get_velocity_estimate()

        self.assertEqual(velocity_estimate.shape, (2,))
        self.assertGreater(float(velocity_estimate[0]), 0.2)

    def test_position_update_returns_log_marginal_and_pulls_particles_to_measurement(
        self,
    ):
        random.seed(1)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=512,
            spatial_dim=2,
            dt=1.0,
            alpha=0.0,
            beta=1.0,
            process_noise=GaussianDistribution(zeros(2), 0.02 * eye(2)),
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )
        filt.predict_replay(use_semi_implicit_position_update=True)

        meas_noise = GaussianDistribution(zeros(2), 0.05 * eye(2))
        log_marginal = filt.update_position(
            meas_noise,
            array([1.0, 0.0]),
            return_log_marginal=True,
        )
        position_estimate = filt.get_position_estimate()

        self.assertFalse(logical_or(isnan(log_marginal), isinf(log_marginal)))
        self.assertGreater(float(position_estimate[0]), 0.5)

    def test_linear_association_likelihood_is_positive(self):
        random.seed(2)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=128,
            spatial_dim=2,
            process_noise=GaussianDistribution(zeros(2), 0.01 * eye(2)),
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.5, 0.0]),
            velocity_prior=array([0.2, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )

        H_vel = zeros((2, 6))
        H_vel[:, 2:4] = eye(2)
        meas_noise = GaussianDistribution(zeros(2), 0.05 * eye(2))

        assoc = filt.association_likelihood_linear(
            array([0.2, 0.0]),
            H_vel,
            meas_noise,
        )

        self.assertGreater(float(assoc), 0.0)


if __name__ == "__main__":
    unittest.main()
