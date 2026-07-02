from __future__ import annotations

import json
from pathlib import Path

from pyrecest._backend.capabilities import (
    API_BACKEND_CAPABILITIES,
    BACKEND_SUPPORT_LEVELS,
    iter_api_backend_capabilities,
)
from pyrecest.cli import main as cli_main
from scripts.generate_backend_api_matrix import render_backend_api_matrix


def test_api_backend_capability_rows_have_valid_support_levels() -> None:
    for api_name, support in iter_api_backend_capabilities():
        assert api_name
        for backend in ("numpy", "pytorch", "jax"):
            assert support[backend] in BACKEND_SUPPORT_LEVELS
        assert support.get("notes")


def test_iter_api_backend_capabilities_returns_row_copies() -> None:
    api_name, support = iter_api_backend_capabilities()[0]
    original_notes = API_BACKEND_CAPABILITIES[api_name]["notes"]

    support["notes"] = "mutated by caller"

    assert API_BACKEND_CAPABILITIES[api_name]["notes"] == original_notes


def test_cli_backends_reports_machine_readable_capabilities(capsys) -> None:
    assert cli_main(["backends"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api"] == API_BACKEND_CAPABILITIES


def test_backend_api_matrix_documentation_matches_generator() -> None:
    expected = render_backend_api_matrix()
    actual = Path("docs/backend-api-matrix.md").read_text(encoding="utf-8")
    assert actual == expected
