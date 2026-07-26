import numpy as np
from pyrecest.calibration.time_offset import time_offset_error_summary


def test_time_offset_error_summary_accepts_none_offset_like_apply_time_offset():
    summary = time_offset_error_summary(
        np.array([0.0, 1.0]),
        np.array([[0.0], [1.0]]),
        np.array([0.0, 1.0]),
        np.array([[0.0], [1.0]]),
        None,
    )

    assert summary["time_offset_s"] == 0.0
    assert summary["count"] == 2.0
    assert summary["rmse"] == 0.0
