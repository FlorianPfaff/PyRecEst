import pytest
from scripts import check_public_api_registry as registry_script


def test_render_markdown_sanitizes_table_cell_separators(monkeypatch):
    separator = chr(124)
    replacement = chr(0xFF5C)

    monkeypatch.setattr(
        registry_script,
        "_load_registry",
        lambda: (
            {
                f"API{separator}Name": {
                    "module": f"pyrecest.module{separator}part",
                    "category": "stable",
                    "backend_contract": f"Contract{separator}Name",
                    "notes": f"first {separator} second\ncontinued",
                }
            },
            ("stable",),
        ),
    )

    rendered = registry_script.render_markdown()
    data_row = rendered.splitlines()[-1]

    assert data_row.count(separator) == 6
    assert f"API{replacement}Name" in data_row
    assert f"pyrecest.module{replacement}part" in data_row
    assert f"Contract{replacement}Name" in data_row
    assert f"first {replacement} second<br>continued" in data_row


def test_main_help_does_not_load_registry(monkeypatch, capsys):
    def fail_validate_registry():
        raise AssertionError("--help should not require registry validation")

    monkeypatch.setattr(registry_script, "validate_registry", fail_validate_registry)

    with pytest.raises(SystemExit) as exc_info:
        registry_script.main(["--help"])

    assert exc_info.value.code == 0
    assert "--output" in capsys.readouterr().out
