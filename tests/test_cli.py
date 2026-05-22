import json
from pathlib import Path

from pyrecest.cli import main


def test_cli_backends_outputs_json(capsys):
    assert main(["backends"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "facade" in payload
    assert "api" in payload


def test_cli_run_scenario_with_expected(capsys):
    assert (
        main(
            [
                "run-scenario",
                "scenarios/linear_gaussian_cv_1d/config.toml",
                "--expected",
                "scenarios/linear_gaussian_cv_1d/expected.json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "linear_gaussian_cv_1d"


def test_cli_run_scenario_rejects_final_estimate_length_mismatch(tmp_path, capsys):
    expected = json.loads(
        Path("scenarios/linear_gaussian_cv_1d/expected.json").read_text(
            encoding="utf-8"
        )
    )
    expected["final_estimate"] = [*expected["final_estimate"], 0.0]
    expected_path = tmp_path / "expected_length_mismatch.json"
    expected_path.write_text(json.dumps(expected), encoding="utf-8")

    assert (
        main(
            [
                "run-scenario",
                "scenarios/linear_gaussian_cv_1d/config.toml",
                "--expected",
                str(expected_path),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "final_estimate length mismatch" in captured.err
