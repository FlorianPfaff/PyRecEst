import importlib

import numpy as np
import pyrecest.calibration as calibration


def _summary_row() -> dict[str, float]:
    return {
        "time_offset_s": 0.0,
        "count": 1.0,
        "mean": 1.0,
        "std": 0.0,
        "rmse": 1.0,
        "p95": 1.0,
        "max": 1.0,
    }


def test_calibration_hotfixes_are_reload_idempotent():
    module = calibration

    for _ in range(2):
        module = importlib.reload(module)

        aggregated = module.aggregate_time_offset_sweeps([[_summary_row()]])
        assert aggregated[0]["rmse"] == 1.0

        examples = module.make_bias_training_examples(
            measurement_times_s=np.array([0.0]),
            measurement_values=np.array([[2.0]]),
            reference_times_s=np.array([0.0]),
            reference_values=np.array([[1.0]]),
            max_time_delta_s=None,
        )
        np.testing.assert_array_equal(examples.residual, np.array([[1.0]]))
