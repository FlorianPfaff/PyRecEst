"""Contract tests for backend capability metadata and documentation."""

from __future__ import annotations

from pathlib import Path

from scripts.check_backend_api_matrix import (
    load_capability_module,
    parse_documented_matrix,
    validate_documented_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_api_matrix_documentation_matches_capability_metadata():
    module = load_capability_module(
        REPO_ROOT / "src" / "pyrecest" / "_backend" / "capabilities.py"
    )
    documented = parse_documented_matrix(REPO_ROOT / "docs" / "backend-api-matrix.md")

    errors = validate_documented_matrix(
        documented,
        dict(module.API_BACKEND_CAPABILITIES),
        tuple(module.BACKEND_SUPPORT_LEVELS),
    )

    assert errors == []


def test_backend_api_capability_rows_use_declared_support_levels():
    module = load_capability_module(
        REPO_ROOT / "src" / "pyrecest" / "_backend" / "capabilities.py"
    )
    support_levels = set(module.BACKEND_SUPPORT_LEVELS)

    for api_name, row in module.API_BACKEND_CAPABILITIES.items():
        assert row["numpy"] in support_levels, api_name
        assert row["pytorch"] in support_levels, api_name
        assert row["jax"] in support_levels, api_name
        assert isinstance(row.get("notes"), str), api_name
        assert row.get("notes"), api_name
