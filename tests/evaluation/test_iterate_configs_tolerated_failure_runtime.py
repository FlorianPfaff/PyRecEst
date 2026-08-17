import importlib

import numpy as np
import pyrecest.backend as backend
import pytest


def test_tolerated_failure_marks_runtime_missing(monkeypatch):
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("iterate_configs_and_runs stores object-valued filter states")

    iterate_module = importlib.import_module(
        "pyrecest.evaluation.iterate_configs_and_runs"
    )

    def failing_predict_update_cycles(*args, **kwargs):
        raise RuntimeError("intentional failure")

    monkeypatch.setattr(
        iterate_module,
        "perform_predict_update_cycles",
        failing_predict_update_cycles,
    )

    groundtruths = np.empty((1, 1), dtype=object)
    measurements = np.empty((1, 1), dtype=object)
    evaluation_config = {
        "plot_each_step": False,
        "convert_to_point_estimate_during_runtime": False,
        "extract_all_point_estimates": False,
        "tolerate_failure": True,
        "auto_warning_on_off": False,
    }

    _, runtimes, run_failed, *_ = iterate_module.iterate_configs_and_runs(
        groundtruths,
        measurements,
        {"name": "dummy"},
        [{"name": "dummy_filter", "parameter": None}],
        evaluation_config,
    )

    assert bool(run_failed[0, 0])
    assert np.isnan(runtimes[0, 0])
