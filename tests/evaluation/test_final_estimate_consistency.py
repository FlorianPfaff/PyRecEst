import numpy as np
import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.evaluation import perform_predict_update_cycles
from pyrecest.evaluation.configure_for_filter import register_filter_factory


class _PredictingFilter:
    def __init__(self):
        self.value = 0.0

    @property
    def filter_state(self):
        return self

    def get_point_estimate(self):
        return array([self.value])


def _predicting_filter_factory(_filter_config, _scenario_config, _precalculated_params):
    filter_obj = _PredictingFilter()

    def prediction_routine():
        filter_obj.value += 1.0

    return filter_obj, prediction_routine, None, None


def _run_cycle(*, extract_all_estimates):
    scenario_config = {
        "n_timesteps": 1,
        "n_meas_at_individual_time_step": [0],
        "apply_sys_noise_times": [True],
        "mtt": False,
        "eot": False,
    }
    groundtruth = np.empty(1, dtype=object)
    groundtruth[0] = np.array([0.0])
    measurements = np.empty(1, dtype=object)
    measurements[0] = np.empty((0, 1))

    return perform_predict_update_cycles(
        scenario_config,
        {"name": "final_estimate_consistency_regression", "parameter": None},
        groundtruth,
        measurements,
        extract_all_estimates=extract_all_estimates,
    )


def test_extracting_history_does_not_change_returned_final_estimate():
    register_filter_factory(
        "final_estimate_consistency_regression", _predicting_filter_factory
    )

    state_without_history, _, estimate_without_history, _ = _run_cycle(
        extract_all_estimates=False
    )
    state_with_history, _, estimate_with_history, all_estimates = _run_cycle(
        extract_all_estimates=True
    )

    npt.assert_allclose(estimate_without_history, array([1.0]))
    npt.assert_allclose(estimate_with_history, estimate_without_history)
    npt.assert_allclose(estimate_with_history, state_with_history.get_point_estimate())
    npt.assert_allclose(
        estimate_without_history, state_without_history.get_point_estimate()
    )
    npt.assert_allclose(all_estimates[0], array([0.0]))
