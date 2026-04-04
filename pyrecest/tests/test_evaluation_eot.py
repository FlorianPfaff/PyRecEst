from pyrecest.evaluation import (
    evaluate_for_simulation_config,
)
from pyrecest.tests.test_evaluation_basic import TestEvalationBase

class TestEvalationEOT(TestEvalationBase):
    scenario_name = "R2randomWalkEOT"

    def test_evaluate_for_simulation_config_R2_random_walk(self):
        filters_configs_input = [
            {"name": "random_matrix", "parameter": None},
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
        
