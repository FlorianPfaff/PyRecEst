import importlib

import numpy as np
import pyrecest.backend as backend
import pytest


def test_iterate_configs_and_runs_uses_config_count_for_vector_parameters(monkeypatch):
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("iterate_configs_and_runs stores object-valued filter states")

    iterate_module = importlib.import_module(
        "pyrecest.evaluation.iterate_configs_and_runs"
    )
    vector_parameter = np.array([1.0, 2.0])
    calls = []

    def fake_predict_update_cycles(
        scenario_config,
        *,
        filter_config,
        groundtruth,
        measurements,
    ):
        calls.append(
            (
                scenario_config,
                filter_config["parameter"],
                groundtruth,
                measurements,
            )
        )
        return object(), 0.25, None, None

    monkeypatch.setattr(
        iterate_module,
        "perform_predict_update_cycles",
        fake_predict_update_cycles,
    )

    groundtruths = np.empty((2, 1), dtype=object)
    measurements = np.empty((2, 1), dtype=object)
    evaluation_config = {
        "plot_each_step": False,
        "convert_to_point_estimate_during_runtime": False,
        "extract_all_point_estimates": False,
        "tolerate_failure": False,
        "auto_warning_on_off": False,
    }

    last_filter_states, runtimes, run_failed, *_ = (
        iterate_module.iterate_configs_and_runs(
            groundtruths,
            measurements,
            {"name": "dummy"},
            [{"name": "dummy_filter", "parameter": vector_parameter}],
            evaluation_config,
        )
    )

    assert np.shape(last_filter_states) == (1, 2)
    assert np.shape(runtimes) == (1, 2)
    assert np.shape(run_failed) == (1, 2)
    assert len(calls) == 2
    for _, parameter, _, _ in calls:
        np.testing.assert_array_equal(parameter, vector_parameter)
