import json
from pathlib import Path

import pytest
from pyrecest.cli import _values_equal, main


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


def test_cli_run_scenario_rejects_malformed_expected_json(tmp_path, capsys):
    expected_path = tmp_path / "malformed.json"
    expected_path.write_text("{not valid json", encoding="utf-8")

    assert (
        main(
            [
                "run-scenario",
                "scenarios/linear_gaussian_cv_1d/config.toml",
                "--expected",
                str(expected_path),
            ]
        )
        == 2
    )
    assert "failed to read expected results" in capsys.readouterr().err


def test_cli_run_scenario_rejects_nonobject_expected_json(tmp_path, capsys):
    expected_path = tmp_path / "expected_list.json"
    expected_path.write_text("[]", encoding="utf-8")

    assert (
        main(
            [
                "run-scenario",
                "scenarios/linear_gaussian_cv_1d/config.toml",
                "--expected",
                str(expected_path),
            ]
        )
        == 2
    )
    assert "expected results must be a JSON object" in capsys.readouterr().err


@pytest.mark.parametrize("section_name", ["metrics", "diagnostics"])
def test_cli_run_scenario_rejects_nonobject_expected_sections(
    tmp_path, capsys, section_name
):
    expected = json.loads(
        Path("scenarios/linear_gaussian_cv_1d/expected.json").read_text(
            encoding="utf-8"
        )
    )
    expected[section_name] = []
    expected_path = tmp_path / f"expected_invalid_{section_name}.json"
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
        == 2
    )
    assert (
        f"expected results {section_name} must be a JSON object"
        in capsys.readouterr().err
    )


def test_values_equal_preserves_nested_mapping_key_types():
    assert not _values_equal({1: "value"}, {"1": "value"})


def test_values_equal_does_not_collapse_distinct_nested_keys():
    assert not _values_equal({1: "numeric", "1": "text"}, {"1": "text"})
