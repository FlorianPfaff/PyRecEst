"""Tests for the SO(3)^K sequence-filtering example."""

from __future__ import annotations

import runpy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "basic" / "so3_product_sequence_filter.py"


def test_so3_product_sequence_filter_example_runs():
    namespace = runpy.run_path(str(EXAMPLE_PATH))

    result, summary = namespace["run_example"]()

    assert result.estimates.shape == (12, 3, 4)
    assert summary["mean_effective_sample_size"] > 0.0
    assert summary["resampling_count"] >= 0
