from __future__ import annotations

import numpy as np
from pyrecest.cli import _check_expected_mapping


def test_expected_mapping_accepts_numpy_array_equal_to_json_list() -> None:
    errors = _check_expected_mapping(
        "diagnostics",
        {"indices": np.array([2, 0, 1])},
        {"indices": [2, 0, 1]},
        tolerance=0.0,
    )
    assert errors == []


def test_expected_mapping_reports_numpy_array_mismatches() -> None:
    errors = _check_expected_mapping(
        "diagnostics",
        {"indices": np.array([2, 0, 1])},
        {"indices": [2, 1, 0]},
        tolerance=0.0,
    )
    assert len(errors) == 1
    assert errors[0].startswith("diagnostics.indices mismatch:")


def test_expected_mapping_accepts_numpy_boolean_scalars() -> None:
    errors = _check_expected_mapping(
        "diagnostics",
        {"resampled": np.bool_(True)},
        {"resampled": True},
        tolerance=0.0,
    )
    assert errors == []
