from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    estimate_innovation_covariance_scale,
    summarize_nis_consistency,
)
from scipy.stats import chi2


def test_summary_reports_moments_coverage_and_scales() -> None:
    values = np.array([1.0, 2.0, 10.0])
    summary = summarize_nis_consistency(
        values,
        measurement_dim=3,
        gate_probabilities=(0.95,),
    )

    coverage = summary.coverage_for(0.95)
    threshold = float(chi2.ppf(0.95, df=3))
    assert summary.measurement_dim == 3
    assert summary.count == 3
    assert summary.nis_mean == pytest.approx(13.0 / 3.0)
    assert summary.mean_innovation_covariance_scale == pytest.approx(13.0 / 9.0)
    assert summary.nis_max == 10.0
    assert summary.chi2_ks_distance is not None
    assert 0.0 <= summary.chi2_ks_distance <= 1.0
    assert coverage.threshold == pytest.approx(threshold)
    assert coverage.actual_fraction == pytest.approx(2.0 / 3.0)
    assert coverage.coverage_gap == pytest.approx(2.0 / 3.0 - 0.95)
    assert coverage.observed_quantile == pytest.approx(np.quantile(values, 0.95))
    assert coverage.innovation_covariance_scale == pytest.approx(
        np.quantile(values, 0.95) / threshold
    )


def test_summary_ks_distance_uses_two_sided_empirical_cdf() -> None:
    values = np.array([0.5, 2.0, 8.0])
    summary = summarize_nis_consistency(values, 2, gate_probabilities=())

    sorted_values = np.sort(values)
    theoretical = chi2.cdf(sorted_values, df=2)
    upper = np.arange(1, 4, dtype=float) / 3.0
    lower = np.arange(3, dtype=float) / 3.0
    expected = max(np.max(upper - theoretical), np.max(theoretical - lower))
    assert summary.chi2_ks_distance == pytest.approx(expected)


def test_empty_summary_retains_requested_gate_thresholds() -> None:
    summary = summarize_nis_consistency([], 2, gate_probabilities=(0.95,))

    assert summary.count == 0
    assert summary.nis_mean is None
    assert summary.chi2_ks_distance is None
    assert summary.coverage[0].threshold == pytest.approx(chi2.ppf(0.95, df=2))
    assert summary.coverage[0].actual_fraction is None


def test_mean_scale_matches_chi_square_mean() -> None:
    estimate = estimate_innovation_covariance_scale([4.0, 4.0], 2)

    assert estimate.method == "mean"
    assert estimate.statistic == 4.0
    assert estimate.target == 2.0
    assert estimate.scale == 2.0
    assert estimate.quantile is None


def test_quantile_scale_matches_chi_square_quantile() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    estimate = estimate_innovation_covariance_scale(
        values,
        3,
        method="quantile",
        quantile=0.75,
    )

    assert estimate.statistic == pytest.approx(np.quantile(values, 0.75))
    assert estimate.target == pytest.approx(chi2.ppf(0.75, df=3))
    assert estimate.scale == pytest.approx(estimate.statistic / estimate.target)
    assert estimate.quantile == 0.75


@pytest.mark.parametrize(
    "measurement_dim",
    [0, -1, 1.5, True, "2", np.nan, 2**53 + 1, np.int64(2**53 + 1)],
)
def test_rejects_invalid_measurement_dimensions(measurement_dim) -> None:
    with pytest.raises(ValueError, match="measurement_dim"):
        summarize_nis_consistency([1.0], measurement_dim)


@pytest.mark.parametrize(
    "values",
    [[-1.0], [np.nan], [np.inf], [True], ["1.0"], [1.0 + 2.0j]],
)
def test_rejects_invalid_nis_values(values) -> None:
    with pytest.raises(ValueError, match="nis_values"):
        summarize_nis_consistency(values, 1)


@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, np.inf, True, "0.95"])
def test_rejects_invalid_gate_probabilities(probability) -> None:
    with pytest.raises(ValueError, match="gate probability"):
        summarize_nis_consistency([1.0], 1, gate_probabilities=(probability,))


def test_scale_estimator_rejects_empty_samples_and_unknown_method() -> None:
    with pytest.raises(ValueError, match="at least one"):
        estimate_innovation_covariance_scale([], 1)
    with pytest.raises(ValueError, match="method"):
        estimate_innovation_covariance_scale([1.0], 1, method="median")


def test_rejects_masked_nis_values_but_accepts_clear_masks() -> None:
    with pytest.raises(ValueError, match="nis_values"):
        summarize_nis_consistency(np.ma.array([1.0, 100.0], mask=[False, True]), 1)

    summary = summarize_nis_consistency(np.ma.array([1.0, 2.0], mask=False), 1)
    assert summary.count == 2
