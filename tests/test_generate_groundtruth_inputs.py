# pylint: disable=no-name-in-module,no-member,too-few-public-methods
import unittest

import numpy as np
import pyrecest.backend
from pyrecest.backend import array, eye, to_numpy
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import generate_groundtruth


class _ZeroNoise:
    def sample(self, _n):
        return array([0.0, 0.0])


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Groundtruth generation mutates arrays in-place.",
)
class TestGenerateGroundtruthInputs(unittest.TestCase):
    def _base_simulation_param(self):
        return {
            "n_timesteps": 4,
            "n_targets": 1,
            "initial_prior": GaussianDistribution(array([0.0, 0.0]), eye(2)),
            "inputs": array(
                [
                    [1.0, 2.0, 3.0],
                    [10.0, 20.0, 30.0],
                ]
            ),
        }

    def _assert_input_driven_groundtruth(self, simulation_param):
        groundtruth = generate_groundtruth(simulation_param, x0=array([0.0, 0.0]))
        actual = np.array([to_numpy(state) for state in groundtruth])
        np.testing.assert_allclose(
            actual,
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 10.0],
                    [3.0, 30.0],
                    [6.0, 60.0],
                ]
            ),
        )

    def test_gen_next_state_with_noise_uses_input_columns(self):
        simulation_param = self._base_simulation_param()
        simulation_param["gen_next_state_with_noise"] = (
            lambda previous_state, current_input: previous_state + current_input
        )

        self._assert_input_driven_groundtruth(simulation_param)

    def test_gen_next_state_without_noise_uses_input_columns(self):
        simulation_param = self._base_simulation_param()
        simulation_param["sys_noise"] = _ZeroNoise()
        simulation_param["gen_next_state_without_noise"] = (
            lambda previous_state, current_input: previous_state + current_input
        )

        self._assert_input_driven_groundtruth(simulation_param)


if __name__ == "__main__":
    unittest.main()
