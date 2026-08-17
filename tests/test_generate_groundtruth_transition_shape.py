import unittest
from unittest.mock import Mock

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import generate_groundtruth


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="Groundtruth generation mutates arrays in-place.",
)
class TestGenerateGroundtruthTransitionShape(unittest.TestCase):
    @staticmethod
    def _base_simulation_param():
        return {
            "initial_prior": GaussianDistribution(zeros(2), eye(2)),
            "n_targets": 1,
            "n_timesteps": 2,
        }

    def test_rejects_scalar_noisy_transition_for_multidimensional_state(self):
        simulation_param = self._base_simulation_param()
        simulation_param["gen_next_state_with_noise"] = lambda _state: 5.0

        with self.assertRaisesRegex(
            ValueError,
            r"gen_next_state_with_noise.*shape \(2,\)",
        ):
            generate_groundtruth(simulation_param, x0=array([0.0, 1.0]))

    def test_rejects_singleton_noiseless_transition_for_multidimensional_state(
        self,
    ):
        simulation_param = self._base_simulation_param()
        simulation_param["gen_next_state_without_noise"] = lambda _state: array([5.0])
        simulation_param["sys_noise"] = Mock()
        simulation_param["sys_noise"].sample.return_value = zeros(2)

        with self.assertRaisesRegex(
            ValueError,
            r"gen_next_state_without_noise.*shape \(2,\)",
        ):
            generate_groundtruth(simulation_param, x0=array([0.0, 1.0]))

    def test_preserves_supported_transition_result_shapes(self):
        row_param = self._base_simulation_param()
        row_param["gen_next_state_with_noise"] = lambda _state: array([[2.0, 3.0]])
        row_groundtruth = generate_groundtruth(
            row_param,
            x0=array([0.0, 1.0]),
        )
        npt.assert_allclose(row_groundtruth[1], array([2.0, 3.0]))

        scalar_param = {
            "initial_prior": GaussianDistribution(zeros(1), eye(1)),
            "n_targets": 1,
            "n_timesteps": 2,
            "gen_next_state_with_noise": lambda _state: 4.0,
        }
        scalar_groundtruth = generate_groundtruth(
            scalar_param,
            x0=array([1.0]),
        )
        npt.assert_allclose(scalar_groundtruth[1], array([4.0]))


if __name__ == "__main__":
    unittest.main()
