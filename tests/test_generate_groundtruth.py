import unittest
from unittest.mock import Mock

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import generate_groundtruth


def _zero_noise():
    noise = Mock()
    noise.sample.return_value = zeros(2)
    return noise


class TestGenerateGroundtruth(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_custom_noisy_transition_reads_previous_object_entry(self):
        x0 = array([[0.0, 1.0], [2.0, 3.0]])
        step = array([1.0, -1.0])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 2,
            "n_timesteps": 3,
            "gen_next_state_with_noise": lambda state: state + step,
        }

        groundtruth = generate_groundtruth(simulation_param, x0)

        npt.assert_allclose(groundtruth[0], x0)
        npt.assert_allclose(groundtruth[1], x0 + step)
        npt.assert_allclose(groundtruth[2], x0 + 2.0 * step)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_custom_noiseless_transition_reads_previous_object_entry(self):
        x0 = array([[0.0, 1.0], [2.0, 3.0]])
        step = array([0.5, 1.0])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 2,
            "n_timesteps": 3,
            "sys_noise": _zero_noise(),
            "gen_next_state_without_noise": lambda state: state + step,
        }

        groundtruth = generate_groundtruth(simulation_param, x0)

        npt.assert_allclose(groundtruth[0], x0)
        npt.assert_allclose(groundtruth[1], x0 + step)
        npt.assert_allclose(groundtruth[2], x0 + 2.0 * step)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_custom_transition_with_inputs_uses_column_controls(self):
        x0 = array([[0.0, 1.0], [2.0, 3.0]])
        inputs = array([[1.0, 0.5, -0.25], [-2.0, 1.5, 0.75]])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 2,
            "n_timesteps": 4,
            "inputs": inputs,
            "sys_noise": _zero_noise(),
            "gen_next_state_without_noise": lambda state, control: state + control,
        }

        groundtruth = generate_groundtruth(simulation_param, x0)
        expected_first_step = x0 + inputs[:, 0]
        expected_second_step = expected_first_step + inputs[:, 1]
        expected_third_step = expected_second_step + inputs[:, 2]

        npt.assert_allclose(groundtruth[0], x0)
        npt.assert_allclose(groundtruth[1], expected_first_step)
        npt.assert_allclose(groundtruth[2], expected_second_step)
        npt.assert_allclose(groundtruth[3], expected_third_step)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_preserves_matrix_shape_for_one_dimensional_multiple_targets(self):
        x0 = array([[0.0], [2.0]])
        step = array([1.0])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(1), eye(1)),
            "n_targets": 2,
            "n_timesteps": 2,
            "gen_next_state_with_noise": lambda state: state + step,
        }

        groundtruth = generate_groundtruth(simulation_param, x0)

        self.assertEqual(groundtruth[0].shape, (2, 1))
        self.assertEqual(groundtruth[1].shape, (2, 1))
        npt.assert_allclose(groundtruth[0], x0)
        npt.assert_allclose(groundtruth[1], array([[1.0], [3.0]]))

    def test_rejects_vector_initial_state_for_multiple_targets(self):
        x0 = array([0.0, 1.0])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 2,
            "n_timesteps": 2,
            "gen_next_state_with_noise": lambda state: state,
        }

        with self.assertRaisesRegex(ValueError, "number of targets"):
            generate_groundtruth(simulation_param, x0)

    def test_rejects_wrong_number_of_input_columns(self):
        x0 = array([[0.0, 1.0], [2.0, 3.0]])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 2,
            "n_timesteps": 4,
            "inputs": array([[1.0, 0.5], [-2.0, 1.5]]),
            "sys_noise": _zero_noise(),
            "gen_next_state_without_noise": lambda state, control: state + control,
        }

        with self.assertRaisesRegex(ValueError, "number of timesteps"):
            generate_groundtruth(simulation_param, x0)

    def test_rejects_inputs_for_identity_system_model(self):
        x0 = array([0.0, 1.0])
        simulation_param = {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 1,
            "n_timesteps": 2,
            "inputs": array([[1.0], [-2.0]]),
            "sys_noise": _zero_noise(),
        }

        with self.assertRaisesRegex(ValueError, "identity system model"):
            generate_groundtruth(simulation_param, x0)


if __name__ == "__main__":
    unittest.main()
