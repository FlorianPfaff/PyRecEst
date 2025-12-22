import os
import tempfile
import unittest
from typing import Optional

import matplotlib
import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import all, array, eye, sqrt, zeros
from pyrecest.distributions import (
    GaussianDistribution,
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.evaluation import (
    check_and_fix_config,
    configure_for_filter,
    determine_all_deviations,
    evaluate_for_file,
    evaluate_for_simulation_config,
    generate_groundtruth,
    generate_measurements,
    generate_simulated_scenarios,
    get_axis_label,
    get_distance_function,
    get_extract_mean,
    iterate_configs_and_runs,
    perform_predict_update_cycles,
    simulation_database,
    summarize_filter_results,
)
from pyrecest.filters import HypertoroidalParticleFilter, KalmanFilter
from shapely.geometry import Polygon


class TestEvalationBase(unittest.TestCase):
    scenario_name: Optional[str] = None

    def setUp(self):
        assert (
            self.scenario_name is not None
        ), "scenario_name must be set in child class"

        simulation_config = simulation_database(self.scenario_name)
        self.simulation_param = check_and_fix_config(simulation_config)
        self.n_timesteps_default = 10
        self.n_runs_default = 8
        self.tmpdirname = tempfile.TemporaryDirectory()  # pylint: disable=R1732

    def tearDown(self):
        self.tmpdirname.cleanup()


class TestEvalationBasics(TestEvalationBase):
    scenario_name: Optional[str] = "R2randomWalk"

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plot_results(self):
        from pyrecest.evaluation.plot_results import plot_results

        matplotlib.pyplot.close("all")  # Ensure all previous plots are closed

        matplotlib.use("SVG")  # Set the backend to SVG for better compatibility
        # To generate some results
        self.test_evaluate_for_simulation_config_R2_random_walk()
        files = os.listdir(self.tmpdirname.name)
        filename = os.path.join(self.tmpdirname.name, files[0])

        figs, _ = plot_results(
            filename=filename,
            plot_log=False,
            plot_stds=False,
        )

        for fig in figs:
            fig.savefig(f"test_plot_{fig.number}.png")

    @parameterized.expand(
        [
            (np.zeros(2),),
            (None,),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_generate_gt_R2(self, x0):
        groundtruth = generate_groundtruth(self.simulation_param, x0)

        # Test if groundtruth and its content is as expected
        self.assertEqual(np.shape(groundtruth), (self.n_timesteps_default,))
        self.assertEqual(
            np.shape(groundtruth[0]), (self.simulation_param["initial_prior"].dim,)
        )

    @parameterized.expand([(1,), (3,)])
    def test_generate_measurements(self, n_meas):
        self.simulation_param["n_meas_at_individual_time_step"] = n_meas * np.ones(
            self.n_timesteps_default, dtype=int
        )
        measurements = generate_measurements(
            np.zeros(
                (self.n_timesteps_default, self.simulation_param["initial_prior"].dim)
            ),
            self.simulation_param,
        )

        self.assertEqual(np.size(measurements), self.n_timesteps_default)
        for i in range(self.n_timesteps_default):
            self.assertEqual(
                np.atleast_2d(measurements[i]).shape,
                (
                    self.simulation_param["n_meas_at_individual_time_step"][i],
                    self.simulation_param["initial_prior"].dim,
                ),
            )

    @parameterized.expand([("boundary",), ("within",)])
    def test_generate_measurements_eot(self, eot_sample_from: str):
        np.random.seed(0)
        simulation_param = {
            "eot": True,
            "intensity_lambda": 0.2,
            "target_shape": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "eot_sample_from": eot_sample_from,
            "n_timesteps": self.n_timesteps_default,
        }
        state_dim = 2

        measurements = generate_measurements(
            np.zeros((self.n_timesteps_default, state_dim)),
            simulation_param,
        )

        self.assertEqual(np.size(measurements), self.n_timesteps_default)
        n_meas_at_individual_time_step = array(
            [meas_at_timestep.shape[0] for meas_at_timestep in measurements]
        )
        # If one measurement at every timestep, then the number is apparently not stochastic
        self.assertFalse(all(n_meas_at_individual_time_step == 1))
        state_dim_at_individual_time_step = array(
            [meas_at_timestep.shape[-1] for meas_at_timestep in measurements]
        )
        has_state_dim_all = state_dim_at_individual_time_step == state_dim
        has_dim_zero_all = state_dim_at_individual_time_step == 0
        self.assertTrue(
            np.all(
                [
                    state_dim or dim_zero
                    for state_dim, dim_zero in zip(has_state_dim_all, has_dim_zero_all)
                ]
            )
        )

    def _generate_simulated_scenario_data(self):
        """Helper that actually generates the data and returns it."""
        self.simulation_param["all_seeds"] = range(self.n_runs_default)
        groundtruths, measurements = generate_simulated_scenarios(self.simulation_param)
        return groundtruths, measurements

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_generate_simulated_scenario(self):
        groundtruths, measurements = self._generate_simulated_scenario_data()

        self.assertEqual(
            np.shape(groundtruths), (self.n_runs_default, self.n_timesteps_default)
        )
        self.assertEqual(
            np.shape(measurements), (self.n_runs_default, self.n_timesteps_default)
        )

    def test_determine_all_deviations(self):
        def dummy_extract_mean(x):
            return x

        def dummy_distance_function(x, y):
            return np.linalg.norm(x - y)

        # Initialize the outer array with object type
        groundtruths = np.empty((3, 4), dtype=object)

        # Populate each entry with (2,) arrays
        for i in range(3):
            for j in range(4):
                groundtruths[i, j] = array([i + j, i - j])

        results = np.array([groundtruths[:, -1], groundtruths[:, -1] + 1])

        # Run the function and get the deviations matrix
        all_deviations = determine_all_deviations(
            results,
            dummy_extract_mean,
            dummy_distance_function,
            groundtruths,
        )

        # Check the shape of the output matrices
        assert len(all_deviations) == len(results)

        # Validate some of the results
        np.testing.assert_allclose(
            # Should be zeros as the lastEstimates match groundtruths
            all_deviations[0],
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            # Should be np.sqrt(2) away from groundtruths
            all_deviations[1],
            [sqrt(2), sqrt(2), sqrt(2)],
        )

    def test_configure_kf(self):
        filterParam = {"name": "kf", "parameter": None}
        scenarioParam = {
            "initial_prior": GaussianDistribution(array([0, 0]), eye(2)),
            "inputs": None,
            "manifold_type": "Euclidean",
            "meas_noise": GaussianDistribution(array([0, 0]), eye(2)),
        }

        (
            configured_filter,
            predictionRoutine,
            _,
            meas_noise_for_filter,
        ) = configure_for_filter(filterParam, scenarioParam)

        self.assertIsInstance(configured_filter, KalmanFilter)
        self.assertIsNotNone(predictionRoutine)
        self.assertTrue(meas_noise_for_filter.shape == (2, 2))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_configure_pf(self):
        filter_config = {"name": "pf", "parameter": 100}
        scenario_config = {
            "initial_prior": HypertoroidalWrappedNormalDistribution(
                array([0.0, 0.0]), eye(2)
            ),
            "inputs": None,
            "manifold": "hypertorus",
            "gen_next_state_with_noise": lambda x: x,
        }

        configured_filter, predictionRoutine, *_ = configure_for_filter(
            filter_config, scenario_config
        )

        self.assertIsInstance(configured_filter, HypertoroidalParticleFilter)
        self.assertIsNotNone(predictionRoutine)

    def test_configure_unsupported_filter(self):
        filterParam = {"name": "unsupported_filter", "parameter": 10}
        scenario_config = {
            "initial_prior": "some_initial_prior",
            "inputs": None,
            "manifold": "Euclidean",
        }

        with self.assertRaises(ValueError):
            configure_for_filter(filterParam, scenario_config)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",
    )
    def test_perform_predict_update_cycles(self):
        scenario_name = "R2randomWalk"
        scenario_param = simulation_database(scenario_name)
        scenario_param = check_and_fix_config(scenario_param)

        meas = generate_measurements(
            np.zeros((self.n_timesteps_default, 2)), scenario_param
        )

        (
            last_filter_state,
            time_elapsed,
            last_estimate,
            all_estimates,
        ) = perform_predict_update_cycles(
            scenario_param,
            {"name": "kf", "parameter": None},
            np.zeros((self.n_timesteps_default, 2)),
            measurements=meas,
        )

        self.assertIsInstance(time_elapsed, float)
        self.assertGreater(time_elapsed, 0)
        self.assertIsNotNone(last_filter_state)
        self.assertIsInstance(last_estimate, np.ndarray)
        self.assertEqual(last_estimate.shape, (2,))
        self.assertIsNone(all_estimates)

    def test_get_distance_function(self):
        distance_function = get_distance_function("hypertorus")

        self.assertTrue(
            callable(distance_function),
            f"Expected distanceFunction to be callable, but got {type(distance_function)}",
        )
        self.assertEqual(distance_function(array([0, 0]), array([0, 0])), 0)

    def test_get_mean_calc(self):
        extract_mean = get_extract_mean("hypertorus")

        self.assertTrue(
            callable(extract_mean),
            f"Expected extractMean to be callable, but got {type(extract_mean)}",
        )

    def test_get_axis_label(self):
        error_label = get_axis_label("hypertorus")

        self.assertTrue(
            isinstance(error_label, str),
            f"Expected errorLabel to be a string, but got {type(error_label)}",
        )

    @parameterized.expand(
        [
            ([{"name": "kf", "parameter": None}],),
            (
                [
                    {"name": "kf", "parameter": None},
                    {"name": "pf", "parameter": 51},
                    {"name": "pf", "parameter": 81},
                ],
            ),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_iterate_configs_and_runs(self, filter_configs):
        groundtruths, measurements = self._generate_simulated_scenario_data()
        evaluation_config = {
            "plot_each_step": False,
            "convert_to_point_estimate_during_runtime": False,
            "extract_all_point_estimates": False,
            "tolerate_failure": False,
            "auto_warning_on_off": False,
        }

        iterate_configs_and_runs(
            groundtruths,
            measurements,
            self.simulation_param,
            filter_configs,
            evaluation_config,
        )

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def _validate_eval_data(
        self,
        scenario_config,
        filter_configs,
        evaluation_config,
        last_filter_states,
        runtimes,
        run_failed,
        groundtruths,
        measurements,
        **_,
    ):
        n_configs = len(filter_configs)

        self.assertIsInstance(scenario_config, dict)
        self.assertIsInstance(scenario_config["manifold"], str)

        self.assertEqual(len(filter_configs), n_configs)
        self.assertDictEqual(filter_configs[0], {"name": "kf", "parameter": None})
        self.assertDictEqual(filter_configs[1], {"name": "pf", "parameter": 51})
        self.assertDictEqual(filter_configs[2], {"name": "pf", "parameter": 81})

        self.assertIsInstance(evaluation_config, dict)

        self.assertEqual(np.shape(last_filter_states), (n_configs, self.n_runs_default))
        self.assertTrue(np.all(last_filter_states != None))  # noqa

        self.assertEqual(np.shape(runtimes), (n_configs, self.n_runs_default))
        print(runtimes)
        self.assertTrue(np.all(runtimes > 0))

        self.assertEqual(np.shape(run_failed), (n_configs, self.n_runs_default))
        self.assertTrue(not np.any(run_failed))

        self.assertEqual(np.ndim(groundtruths), 2)
        self.assertIsInstance(groundtruths[0, 0], np.ndarray)
        self.assertIn(np.ndim(groundtruths[0, 0]), (1, 2))

        self.assertEqual(np.ndim(measurements), 2)
        self.assertIsInstance(measurements[0, 0], np.ndarray)
        self.assertIn(np.ndim(measurements[0, 0]), (1, 2))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_evaluate_for_simulation_config_R2_random_walk(self):
        filters_configs_input = [
            {"name": "kf", "parameter": None},
            {"name": "pf", "parameter": [51, 81]},
        ]

        (
            last_filter_states,  # pylint: disable=R0801
            runtimes,  # pylint: disable=R0801
            run_failed,  # pylint: disable=R0801
            groundtruths,  # pylint: disable=R0801
            measurements,  # pylint: disable=R0801
            scenario_config,  # pylint: disable=R0801
            filter_configs,  # pylint: disable=R0801
            evaluation_config,  # pylint: disable=R0801
        ) = evaluate_for_simulation_config(
            self.scenario_name,
            filters_configs_input,
            n_runs=self.n_runs_default,
            n_timesteps=self.n_timesteps_default,
            initial_seed=1,
            auto_warning_on_off=False,
            save_folder=self.tmpdirname.name,
        )
        self._validate_eval_data(
            scenario_config,
            filter_configs,
            evaluation_config,
            last_filter_states,
            runtimes,
            run_failed,
            groundtruths,
            measurements,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_evaluate_for_file_R2_random_walk(self):
        self.simulation_param["all_seeds"] = range(self.n_runs_default)
        groundtruths, measurements = generate_simulated_scenarios(self.simulation_param)

        filters_configs_input = [
            {"name": "kf", "parameter": None},
            {"name": "pf", "parameter": [51, 81]},
        ]

        filename = "tmp.npy"
        np.save(filename, {"groundtruths": groundtruths, "measurements": measurements})

        scenario_config = {
            "manifold": "Euclidean",
            "initial_prior": GaussianDistribution(zeros(2), 0.5 * eye(2)),
            "meas_noise": GaussianDistribution(zeros(2), 0.5 * eye(2)),
            "sys_noise": GaussianDistribution(zeros(2), 0.5 * eye(2)),
        }

        (
            last_filter_states,  # pylint: disable=R0801
            runtimes,  # pylint: disable=R0801
            run_failed,  # pylint: disable=R0801
            groundtruths,  # pylint: disable=R0801
            measurements,  # pylint: disable=R0801
            scenario_config,  # pylint: disable=R0801
            filter_configs,  # pylint: disable=R0801
            evaluation_config,  # pylint: disable=R0801
        ) = evaluate_for_file(
            filename,
            filters_configs_input,
            scenario_config,
            save_folder=self.tmpdirname.name,
        )

        self._validate_eval_data(
            scenario_config,
            filter_configs,
            evaluation_config,
            last_filter_states,
            runtimes,
            run_failed,
            groundtruths,
            measurements,
        )

    def _load_evaluation_data(self):
        self.test_evaluate_for_simulation_config_R2_random_walk()
        files = os.listdir(self.tmpdirname.name)
        filename = os.path.join(self.tmpdirname.name, files[0])
        return np.load(filename, allow_pickle=True).item()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_file_content(self):
        data = self._load_evaluation_data()
        self._validate_eval_data(**data)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_group_results_by_filter(self):
        from pyrecest.evaluation.group_results_by_filter import group_results_by_filter

        # Dummy data
        data1 = [
            {
                "name": "kf",
                "parameter": 41,
                "error_mean": 1.17,
                "error_std": 0.75,
                "time_mean": 0.009,
                "failure_rate": 0.0,
            },
            {
                "name": "kf",
                "parameter": 61,
                "error_mean": 1.27,
                "error_std": 0.85,
                "time_mean": 0.109,
                "failure_rate": 0.0,
            },
            {
                "name": "pf",
                "parameter": 51,
                "error_mean": 1.21,
                "error_std": 0.77,
                "time_mean": 0.031,
                "failure_rate": 0.0,
            },
            {
                "name": "pf",
                "parameter": 81,
                "error_mean": 1.18,
                "error_std": 0.71,
                "time_mean": 0.030,
                "failure_rate": 0.0,
            },
        ]
        data2 = [data1[3], data1[1], data1[0], data1[2]]

        repackaged_data1 = group_results_by_filter(data1)
        repackaged_data2 = group_results_by_filter(data2)

        self.assertEqual(repackaged_data1, repackaged_data2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",
    )
    def test_summarize_filter_results(self):
        data = self._load_evaluation_data()
        results_summarized = summarize_filter_results(**data)

        for result in results_summarized:
            error_mean = result["error_mean"]
            error_std = result["error_std"]
            time_mean = result["time_mean"]
            failure_rate = result["failure_rate"]

            self.assertGreaterEqual(error_mean, 0.0)
            self.assertLessEqual(error_mean, 2.0)

            self.assertGreaterEqual(error_std, 0.0)
            self.assertLessEqual(error_std, 1.0)

            self.assertGreaterEqual(time_mean, 0.0)
            self.assertLessEqual(time_mean, 1.0)

            self.assertEqual(failure_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
