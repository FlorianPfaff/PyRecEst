import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import array, get_backend_name
from pyrecest.evaluation import perform_predict_update_cycles
from pyrecest.evaluation.configure_for_filter import register_filter_factory


class _FractionalEstimateFilter:
    @property
    def filter_state(self):
        return self

    def get_point_estimate(self):
        return array([0.5])


def _fractional_estimate_factory(
    _filter_config, _scenario_config, _precalculated_params
):
    return _FractionalEstimateFilter(), lambda: None, None, None


@pytest.mark.skipif(
    get_backend_name() != "numpy",
    reason="Dense NumPy evaluation storage is covered on the NumPy backend.",
)
def test_integer_groundtruth_preserves_fractional_estimate_history():
    filter_name = "integer_groundtruth_fractional_estimate_history_regression"
    register_filter_factory(filter_name, _fractional_estimate_factory)
    scenario_config = {
        "n_timesteps": 2,
        "n_meas_at_individual_time_step": [0, 0],
        "apply_sys_noise_times": [False, False],
        "mtt": False,
        "eot": False,
    }
    groundtruth = np.array([[0], [1]], dtype=int)
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

    assert all_estimates.dtype.kind == "f"
    npt.assert_allclose(last_estimate, array([0.5]))
    npt.assert_allclose(all_estimates, np.array([[0.5], [0.5]]))
