import numpy as np
import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.evaluation import perform_predict_update_cycles
from pyrecest.evaluation.configure_for_filter import register_filter_factory


class _NoUpdateFilter:
    @property
    def filter_state(self):
        return self

    def get_point_estimate(self):
        return array([0.0])


def _no_update_filter_factory(_filter_config, _scenario_config, _precalculated_params):
    filter_obj = _NoUpdateFilter()

    def prediction_routine():
        return None

    return filter_obj, prediction_routine, None, None


def test_flat_empty_measurement_array_is_zero_updates():
    filter_name = "flat_empty_measurement_zero_update_regression"
    register_filter_factory(filter_name, _no_update_filter_factory)
    scenario_config = {
        "n_timesteps": 1,
        "n_meas_at_individual_time_step": [0],
        "apply_sys_noise_times": [False],
        "mtt": False,
        "eot": False,
    }
    groundtruth = np.array([[0.0]])
    measurements = np.empty(1, dtype=object)
    measurements[0] = np.array([])

    _, _, last_estimate, _ = perform_predict_update_cycles(
        scenario_config,
        {"name": filter_name, "parameter": None},
        groundtruth,
        measurements,
    )

    npt.assert_allclose(last_estimate, array([0.0]))
