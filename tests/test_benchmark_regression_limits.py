from __future__ import annotations

import importlib.util
import math
from pathlib import Path


def _load_checker_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "check_benchmark_regression.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_benchmark_regression", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejects_unbounded_cli_limits():
    checker = _load_checker_module()

    assert checker._validate_cli_limits(  # pylint: disable=protected-access
        max_slowdown=math.inf,
        abs_tol=0.0,
        rel_tol=0.0,
    ) == ["max-slowdown must be finite"]


def test_accepts_default_cli_limits():
    checker = _load_checker_module()

    assert (
        checker._validate_cli_limits(  # pylint: disable=protected-access
            max_slowdown=1.5,
            abs_tol=1e-8,
            rel_tol=1e-8,
        )
        == []
    )
