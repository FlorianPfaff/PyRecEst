import math

from pyrecest.calibration import aggregate_time_offset_sweeps
from pyrecest.calibration.time_offset import (
    aggregate_time_offset_sweeps as aggregate_time_offset_sweeps_from_module,
)


def _summary_row(*, count: float, p95: float) -> dict[str, float]:
    return {
        "time_offset_s": 0.0,
        "count": count,
        "mean": 0.0,
        "std": 0.0,
        "rmse": 0.0,
        "p95": p95,
        "max": p95,
    }


def test_aggregate_time_offset_sweeps_does_not_average_percentiles():
    sweeps = [
        [_summary_row(count=100.0, p95=0.0)],
        [_summary_row(count=1.0, p95=100.0)],
    ]

    for aggregate in (
        aggregate_time_offset_sweeps,
        aggregate_time_offset_sweeps_from_module,
    ):
        aggregated = aggregate(sweeps)

        assert math.isnan(aggregated[0]["p95"])


def test_aggregate_time_offset_sweeps_preserves_single_nonempty_percentile():
    sweeps = [
        [_summary_row(count=0.0, p95=float("nan"))],
        [_summary_row(count=4.0, p95=2.5)],
    ]

    for aggregate in (
        aggregate_time_offset_sweeps,
        aggregate_time_offset_sweeps_from_module,
    ):
        aggregated = aggregate(sweeps)

        assert aggregated[0]["p95"] == 2.5
