from __future__ import annotations

import importlib

import numpy as np


def test_evaluate_for_file_counts_empty_1d_measurements_as_zero(tmp_path, monkeypatch):
    module = importlib.import_module("pyrecest.evaluation.evaluate_for_file")
    input_file = tmp_path / "scenario.npy"
    measurements = np.empty((1, 3), dtype=object)
    measurements[0, 0] = np.array([], dtype=float)
    measurements[0, 1] = np.array([4.0, 5.0])
    measurements[0, 2] = np.array([[1.0, 2.0], [3.0, 4.0]])
    groundtruths = np.zeros((1, 3, 2), dtype=float)
    np.save(input_file, {"groundtruths": groundtruths, "measurements": measurements})

    captured = {}

    def capture_call(
        groundtruths_arg,
        measurements_arg,
        filter_configs,
        scenario_config,
        **kwargs,
    ):
        captured["scenario_config"] = dict(scenario_config)
        return (
            {},
            [],
            np.array([], dtype=bool),
            groundtruths_arg,
            measurements_arg,
            scenario_config,
            filter_configs,
            kwargs,
        )

    monkeypatch.setattr(module, "evaluate_for_variables", capture_call)

    module.evaluate_for_file(str(input_file), [], {}, save_folder=str(tmp_path))

    np.testing.assert_array_equal(
        captured["scenario_config"]["n_meas_at_individual_time_step"],
        np.array([0, 1, 2], dtype=int),
    )
