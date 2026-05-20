import json

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
