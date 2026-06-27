from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _write_payload(path: Path, entry: dict[str, Any]) -> None:
    path.write_text(json.dumps({"benchmarks": [entry]}), encoding="utf-8")


def _run_checker(
    tmp_path: Path, *, actual_entry: dict[str, Any], baseline_entry: dict[str, Any]
) -> subprocess.CompletedProcess[str]:
    actual_path = tmp_path / "actual.json"
    baseline_path = tmp_path / "baseline.json"
    _write_payload(actual_path, actual_entry)
    _write_payload(baseline_path, baseline_entry)
    return subprocess.run(
        [
            sys.executable,
            "scripts/check_benchmark_regression.py",
            str(actual_path),
            str(baseline_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_benchmark_regression_rejects_boolean_elapsed(tmp_path: Path) -> None:
    completed = _run_checker(
        tmp_path,
        actual_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "elapsed_seconds": True,
            "final_estimate": [200.0, 1.0],
        },
        baseline_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "max_elapsed_seconds": 30.0,
            "final_estimate": [200.0, 1.0],
        },
    )

    assert completed.returncode == 1
    assert (
        "::error::linear_kalman.elapsed_seconds must be numeric, not boolean"
        in completed.stdout
    )
    assert "Traceback" not in completed.stderr


def test_benchmark_regression_rejects_text_elapsed_without_traceback(
    tmp_path: Path,
) -> None:
    completed = _run_checker(
        tmp_path,
        actual_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "elapsed_seconds": "0.01",
            "final_estimate": [200.0, 1.0],
        },
        baseline_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "max_elapsed_seconds": 30.0,
            "final_estimate": [200.0, 1.0],
        },
    )

    assert completed.returncode == 1
    assert (
        "::error::linear_kalman.elapsed_seconds must be numeric, got '0.01'"
        in completed.stdout
    )
    assert "Traceback" not in completed.stderr


def test_benchmark_regression_rejects_text_iterations(tmp_path: Path) -> None:
    completed = _run_checker(
        tmp_path,
        actual_entry={
            "name": "linear_kalman",
            "iterations": "200",
            "elapsed_seconds": 0.01,
            "final_estimate": [200.0, 1.0],
        },
        baseline_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "max_elapsed_seconds": 30.0,
            "final_estimate": [200.0, 1.0],
        },
    )

    assert completed.returncode == 1
    assert (
        "::error::linear_kalman.iterations must be an integer, got '200'"
        in completed.stdout
    )
    assert "Traceback" not in completed.stderr


def test_benchmark_regression_reports_invalid_final_estimate_without_traceback(
    tmp_path: Path,
) -> None:
    completed = _run_checker(
        tmp_path,
        actual_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "elapsed_seconds": 0.01,
            "final_estimate": [200.0, True],
        },
        baseline_entry={
            "name": "linear_kalman",
            "iterations": 200,
            "max_elapsed_seconds": 30.0,
            "final_estimate": [200.0, 1.0],
        },
    )

    assert completed.returncode == 1
    assert (
        "::error::linear_kalman.final_estimate[1] must be numeric, not boolean"
        in completed.stdout
    )
    assert "Traceback" not in completed.stderr
