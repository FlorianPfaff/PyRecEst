import unittest

import numpy as np
import numpy.testing as npt
import pytest

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, eye, get_backend_name, zeros
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.evaluation import (
    check_and_fix_config,
    configure_for_filter,
    generate_simulated_scenarios,
    perform_predict_update_cycles,
)
from pyrecest.evaluation.configure_for_filter import register_filter_factory
from pyrecest.filters import EuclideanParticleFilter

IS_JAX_BACKEND = get_backend_name() == "jax"


class _LikelihoodRecordingFilter:
    def __init__(self):
        self.applied_likelihood_values = []

    @property
    def filter_state(self):
        return self

    def update_nonlinear_using_likelihood(self, likelihood, measurement=None):
        likelihood_values = likelihood(measurement, array([[0.0]]))
        self.applied_likelihood_values.append(likelihood_values)

    def get_point_estimate(self):
        if not self.applied_likelihood_values:
            return zeros(1)
        return array([self.applied_likelihood_values[-1][0]])


class _PredictOnlyFilter:
    def __init__(self):
        self.predicted = False

    @property
    def filter_state(self):
        return self

    def get_point_estimate(self):
        return array([1.0 if self.predicted else 0.0])


def _likelihood_test_factory(_filter_config, scenario_config, _precalculated_params):
    filter_obj = _LikelihoodRecordingFilter()

    def prediction_routine():
        return None

    return (
        filter_obj,
        prediction_routine,
        scenario_config.get("likelihood"),
        scenario_config.get("meas_noise"),
    )


def _predict_only_test_factory(_filter_config, _scenario_config, _precalculated_params):
    filter_obj = _PredictOnlyFilter()

    def prediction_routine():
        filter_obj.predicted = True

    return filter_obj, prediction_routine, None, None


class TestEvaluationControlFlowRegressions(unittest.TestCase):
    def test_configured_kf_with_inputs_passes_process_covariance(self):
        scenario_config = {
            "initial_prior": GaussianDistribution(array([0.0]), eye(1)),
            "sys_noise": GaussianDistribution(array([0.0]), 0.5 * eye(1)),
            "meas_noise": GaussianDistribution(array([0.0]), eye(1)),
            "inputs": array([[2.0]]),
        }

        filter_obj, prediction_routine, _, _ = configure_for_filter(
            {"name": "kf", "parameter": None},
            scenario_config,
        )

        prediction_routine(array([2.0]))

        npt.assert_allclose(filter_obj.filter_state.mean(), array([2.0]))
        npt.assert_allclose(filter_obj.filter_state.covariance(), 1.5 * eye(1))

    def test_configured_euclidean_pf_initializes_from_prior(self):
        scenario_config = {
            "initial_prior": GaussianDistribution(array([5.0, -3.0]), 1.0e-12 * eye(2)),
            "sys_noise": GaussianDistribution(array([0.0, 0.0]), eye(2)),
            "meas_noise": GaussianDistribution(array([0.0, 0.0]), eye(2)),
            "inputs": None,
            "manifold": "euclidean",
        }

        filter_obj, prediction_routine, _, _ = configure_for_filter(
            {"name": "pf", "parameter": 20},
            scenario_config,
        )

        self.assertIsInstance(filter_obj, EuclideanParticleFilter)
        self.assertIsNotNone(prediction_routine)
        npt.assert_allclose(
            filter_obj.filter_state.mean(),
            scenario_config["initial_prior"].mean(),
            atol=1.0e-4,
        )

    def test_configured_euclidean_pf_honors_noise_free_transition(self):
        particle_count = 4

        def transition(particles):
            if particles.ndim != 2:
                raise AssertionError("transition must receive the particle batch")
            return particles + 2.0

        scenario_config = {
            "initial_prior": LinearDiracDistribution(zeros((particle_count, 1))),
            "sys_noise": LinearDiracDistribution(zeros((1, 1))),
            "inputs": None,
            "manifold": "euclidean",
            "gen_next_state_without_noise": transition,
            "gen_next_state_without_noise_is_vectorized": True,
        }

        filter_obj, prediction_routine, _, _ = configure_for_filter(
            {"name": "pf", "parameter": particle_count},
            scenario_config,
        )

        prediction_routine()

        npt.assert_allclose(
            filter_obj.filter_state.d,
            np.full((particle_count, 1), 2.0),
        )

    def test_configured_euclidean_pf_honors_input_transition_with_noise(self):
        particle_count = 4
        scenario_config = {
            "initial_prior": LinearDiracDistribution(zeros((particle_count, 1))),
            "inputs": array([[3.0]]),
            "manifold": "euclidean",
            "gen_next_state_with_noise": (
                lambda particles, curr_input: particles + curr_input
            ),
            "gen_next_state_with_noise_is_vectorized": True,
        }

        filter_obj, prediction_routine, _, _ = configure_for_filter(
            {"name": "pf", "parameter": particle_count},
            scenario_config,
        )

        prediction_routine(array([3.0]))

        npt.assert_allclose(
            filter_obj.filter_state.d,
            np.full((particle_count, 1), 3.0),
        )

    def test_configure_periodic_particle_filter_rejects_inputs(self):
        scenario_config = {
            "initial_prior": GaussianDistribution(array([0.0]), eye(1)),
            "sys_noise": GaussianDistribution(array([0.0]), eye(1)),
            "inputs": array([[1.0]]),
            "manifold": "circle",
        }

        with self.assertRaisesRegex(ValueError, "Inputs currently not supported"):
            configure_for_filter({"name": "pf", "parameter": 20}, scenario_config)

    def test_perform_cycles_honors_callable_likelihood_updates(self):
        filter_name = "likelihood_control_flow_regression"
        register_filter_factory(filter_name, _likelihood_test_factory)
        calls = []

        def likelihood(measurement, particles):
            calls.append((measurement, particles))
            return array([float(measurement) + 1.0])

        scenario_config = {
            "n_timesteps": 1,
            "n_meas_at_individual_time_step": [1],
            "apply_sys_noise_times": [False],
            "use_likelihood": True,
            "likelihood": likelihood,
            "mtt": False,
            "eot": False,
        }
        groundtruth = np.array([[0.0]])
        measurements = np.empty(1, dtype=object)
        measurements[0] = np.array([[2.0]])

        _, _, last_estimate, _ = perform_predict_update_cycles(
            scenario_config,
            {"name": filter_name, "parameter": None},
            groundtruth,
            measurements,
        )

        self.assertEqual(len(calls), 1)
        npt.assert_allclose(last_estimate, array([3.0]))

    def test_plotting_prediction_guard_uses_n_timesteps_key(self):
        filter_name = "plotting_timestep_guard_regression"
        register_filter_factory(filter_name, _predict_only_test_factory)
        scenario_config = {
            "n_timesteps": 1,
            "n_meas_at_individual_time_step": [0],
            "apply_sys_noise_times": [True],
            "plot": True,
            "mtt": False,
            "eot": False,
        }
        groundtruth = np.array([[0.0]])
        measurements = np.empty(1, dtype=object)
        measurements[0] = np.empty((0, 1))

        _, _, last_estimate, _ = perform_predict_update_cycles(
            scenario_config,
            {"name": filter_name, "parameter": None},
            groundtruth,
            measurements,
        )

        npt.assert_allclose(last_estimate, array([1.0]))

    def test_extract_all_estimates_handles_object_groundtruth_arrays(self):
        filter_name = "object_groundtruth_estimate_storage_regression"
        register_filter_factory(filter_name, _predict_only_test_factory)
        scenario_config = {
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": [0, 0],
            "apply_sys_noise_times": [False, False],
            "mtt": False,
            "eot": False,
        }
        groundtruth = np.empty(2, dtype=object)
        groundtruth[0] = np.array([0.0])
        groundtruth[1] = np.array([0.0])
        measurements = np.empty(2, dtype=object)
        measurements[0] = np.empty((0, 1))
        measurements[1] = np.empty((0, 1))

        _, _, last_estimate, all_estimates = perform_predict_update_cycles(
            scenario_config,
            {"name": filter_name, "parameter": None},
            groundtruth,
            measurements,
            extract_all_estimates=True,
        )

        npt.assert_allclose(last_estimate, array([0.0]))
        npt.assert_allclose(all_estimates[0], array([0.0]))
        npt.assert_allclose(all_estimates[1], array([0.0]))

    def test_plain_config_defaults_to_non_mtt_non_eot(self):
        config = check_and_fix_config(
            {
                "n_timesteps": 2,
                "initial_prior": GaussianDistribution(array([0.0]), eye(1)),
            }
        )

        self.assertFalse(config["mtt"])
        self.assertFalse(config["eot"])
        self.assertEqual(config["n_meas_at_individual_time_step"], [1, 1])

    def test_eot_meas_per_step_is_normalized_to_per_step_counts(self):
        config = check_and_fix_config(
            {
                "mtt": False,
                "eot": True,
                "n_timesteps": 3,
                "meas_per_step": 2,
                "initial_prior": GaussianDistribution(array([0.0, 0.0]), eye(2)),
            }
        )

        self.assertNotIn("meas_per_step", config)
        self.assertEqual(config["n_meas_at_individual_time_step"], [2, 2, 2])

    @pytest.mark.skipif(
        IS_JAX_BACKEND, reason="MTT measurement generation is unsupported with JAX."
    )
    def test_simulated_single_target_mtt_measurements_accept_object_groundtruth(self):
        simulation_config = {
            "mtt": True,
            "eot": False,
            "all_seeds": [1],
            "n_timesteps": 2,
            "n_targets": 1,
            "initial_prior": GaussianDistribution(array([0.0, 0.0]), eye(2)),
            "sys_noise": GaussianDistribution(array([0.0, 0.0]), 1.0e-6 * eye(2)),
            "meas_matrix_for_each_target": eye(2),
            "meas_noise": GaussianDistribution(array([0.0, 0.0]), 1.0e-6 * eye(2)),
            "detection_probability": 1,
            "clutter_rate": 0,
        }

        groundtruths, measurements = generate_simulated_scenarios(simulation_config)

        self.assertEqual(groundtruths.shape, (1, 2))
        self.assertEqual(measurements.shape, (1, 2))
        self.assertEqual(measurements[0, 0].shape, (1, 2))
        self.assertEqual(measurements[0, 1].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
