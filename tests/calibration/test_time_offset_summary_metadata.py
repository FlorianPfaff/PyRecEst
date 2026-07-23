import numpy as np
from pyrecest.calibration.time_offset import TimeOffsetFitResult


def test_summary_keeps_computed_fields_authoritative():
    result = TimeOffsetFitResult(
        best_offset_s=0.25,
        metric="rmse",
        offsets_s=np.array([0.25]),
        metric_values=np.array([1.5]),
        counts=np.array([4]),
        metadata={
            "metric": "spoofed",
            "best_offset_s": 99.0,
            "evaluated_offsets": 99,
            "best_metric_value": 99.0,
            "best_count": 99,
            "sensor": "camera",
        },
    )

    assert result.summary() == {
        "sensor": "camera",
        "metric": "rmse",
        "best_offset_s": 0.25,
        "evaluated_offsets": 1,
        "best_metric_value": 1.5,
        "best_count": 4,
    }


def test_summary_does_not_fabricate_best_fields_without_valid_fit():
    result = TimeOffsetFitResult(
        best_offset_s=None,
        metric="rmse",
        offsets_s=np.array([0.0]),
        metric_values=np.array([np.nan]),
        counts=np.array([0]),
        metadata={
            "best_metric_value": 0.0,
            "best_count": 100,
            "sensor": "camera",
        },
    )

    summary = result.summary()

    assert summary == {
        "sensor": "camera",
        "metric": "rmse",
        "best_offset_s": None,
        "evaluated_offsets": 1,
    }
