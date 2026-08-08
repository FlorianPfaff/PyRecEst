from __future__ import annotations

import numpy as np
from pyrecest.tracking import (
    estimate_innovation_covariance_scale,
    summarize_nis_consistency,
)


def test_extreme_finite_statistics_do_not_overflow() -> None:
    large = 0.75 * np.finfo(np.float64).max

    with np.errstate(over="raise", invalid="raise"):
        summary = summarize_nis_consistency(
            [large, large],
            1,
            gate_probabilities=(),
        )
        estimate = estimate_innovation_covariance_scale([large, large], 1)

    assert summary.nis_mean == large
    assert summary.nis_std == 0.0
    assert summary.nis_median == large
    assert summary.nis_p90 == large
    assert summary.nis_p95 == large
    assert summary.nis_p99 == large
    assert summary.nis_max == large
    assert summary.mean_innovation_covariance_scale == large
    assert estimate.statistic == large
    assert estimate.scale == large
