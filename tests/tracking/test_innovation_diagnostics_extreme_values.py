import numpy as np
import pytest
from pyrecest.tracking.innovation_diagnostics import (
    InnovationDiagnostic,
    summarize_innovation_diagnostics,
)


def _summary_for(
    nis_values: list[float], residual_norm_values: list[float] | None = None
):
    if residual_norm_values is None:
        residual_norm_values = nis_values
    diagnostics = [
        InnovationDiagnostic(
            measurement_dim=1,
            nis=nis_value,
            residual_norm=residual_norm_value,
        )
        for nis_value, residual_norm_value in zip(
            nis_values, residual_norm_values, strict=True
        )
    ]
    return summarize_innovation_diagnostics(diagnostics, group_by=None)[0]


def test_summary_means_remain_finite_for_large_finite_values():
    large = np.finfo(float).max * 0.75

    with np.errstate(over="raise", invalid="raise"):
        summary = _summary_for([large, large])

    assert summary.nis_mean == pytest.approx(large)
    assert summary.residual_norm_mean == pytest.approx(large)
    assert np.isfinite(summary.nis_mean)
    assert np.isfinite(summary.residual_norm_mean)


def test_summary_percentiles_remain_finite_across_extreme_signed_values():
    extreme = np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        summary = _summary_for([-extreme, extreme], [0.0, extreme])

    assert summary.nis_median == pytest.approx(0.0)
    assert summary.nis_p95 == pytest.approx(0.9 * extreme)
    assert np.isfinite(summary.nis_median)
    assert np.isfinite(summary.nis_p95)
