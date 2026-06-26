"""Regression tests for CLI expected-result comparison helpers."""

from pyrecest.cli import _check_expected_mapping


def test_expected_mapping_reports_nonnumeric_actual_instead_of_raising():
    errors = _check_expected_mapping(
        "metrics",
        {"rmse": "not-a-number"},
        {"rmse": 1.0},
        tolerance=1e-8,
    )

    assert errors == [
        "metrics.rmse mismatch: expected finite numeric 1.0, got 'not-a-number'"
    ]


def test_expected_mapping_compares_boolean_expected_values_exactly():
    errors = _check_expected_mapping(
        "diagnostics",
        {"converged": 1},
        {"converged": True},
        tolerance=1e-8,
    )

    assert errors == ["diagnostics.converged mismatch: expected True, got 1"]
