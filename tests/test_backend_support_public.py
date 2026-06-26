from importlib import import_module

from pyrecest import (
    backend_support,
    format_backend_support_markdown,
    get_backend_support,
)
from pyrecest._backend.capabilities import BACKEND_SUPPORT_LEVELS

backend_support_module = import_module("pyrecest.backend_support")


def test_public_backend_support_lookup():
    assert get_backend_support("KalmanFilter", backend="numpy") == "supported"
    assert (
        backend_support("EvaluationUtilities", backend="pytorch")
        in BACKEND_SUPPORT_LEVELS
    )
    assert get_backend_support("missing-api") is None


def test_backend_support_markdown_contains_expected_rows():
    rendered = format_backend_support_markdown()
    assert "KalmanFilter" in rendered
    assert "BackendFacade" in rendered


def test_backend_support_markdown_handles_table_separators(monkeypatch):
    separator = chr(124)
    replacement = chr(0xFF5C)

    def fake_backend_capabilities():
        return (
            (
                f"Pipe{separator}API",
                {
                    "numpy": "supported",
                    "pytorch": f"partial{separator}bridged",
                    "jax": "unsupported",
                    "notes": f"first {separator} second\ncontinued",
                },
            ),
        )

    monkeypatch.setattr(
        backend_support_module,
        "iter_api_backend_capabilities",
        fake_backend_capabilities,
    )

    rendered = backend_support_module.format_backend_support_markdown()
    data_row = rendered.splitlines()[-1]

    assert data_row.count(separator) == 6
    assert f"Pipe{replacement}API" in data_row
    assert f"partial{replacement}bridged" in data_row
    assert f"first {replacement} second<br>continued" in data_row
