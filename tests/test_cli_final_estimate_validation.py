from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pyrecest.cli import _cmd_run_scenario


class _DummyScenarioResult:
    final_estimate = [1.0, 2.0]
    metrics: dict[str, float] = {}
    diagnostics: dict[str, float] = {}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps({"name": "dummy"}, indent=indent)


@pytest.mark.parametrize(
    "invalid_final_estimate",
    [
        1.0,
        {"first": 1.0, "second": 2.0},
        ["not-a-number", 2.0],
        [True, 2.0],
    ],
)
def test_cmd_run_scenario_reports_malformed_expected_final_estimate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    invalid_final_estimate,
) -> None:
    import pyrecest.scenarios as scenarios

    monkeypatch.setattr(
        scenarios,
        "run_scenario",
        lambda config: _DummyScenarioResult(),
    )
    expected_path = tmp_path / "expected.json"
    expected_path.write_text(
        json.dumps({"final_estimate": invalid_final_estimate}),
        encoding="utf-8",
    )

    result = _cmd_run_scenario(
        SimpleNamespace(config=tmp_path / "config.toml", expected=expected_path, tolerance=None)
    )

    captured = capsys.readouterr()
    assert result == 1
    assert json.loads(captured.out)["name"] == "dummy"
    assert "final_estimate" in captured.err
