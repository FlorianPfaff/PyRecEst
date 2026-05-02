"""Tests for public protocol extension examples."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from pyrecest.protocols.common import SupportsDim, SupportsInputDim

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPOSITORY_ROOT / "examples" / "basic"


def load_example_namespace(example_name: str) -> dict[str, Any]:
    """Load an example module without executing its ``main`` function."""
    return runpy.run_path(str(EXAMPLES_DIR / f"{example_name}.py"))


def test_custom_distribution_protocol_example_runs():
    namespace = load_example_namespace("custom_distribution_protocol")
    distribution, samples, density_at_mean = namespace["run_example"]()

    assert isinstance(distribution, SupportsDim)
    assert isinstance(distribution, SupportsInputDim)
    assert distribution.dim == 1
    assert distribution.input_dim == 1
    assert len(samples) == 3
    assert density_at_mean > 0.0


def test_custom_filter_protocol_example_runs():
    namespace = load_example_namespace("custom_filter_protocol")
    custom_filter, estimates = namespace["run_example"]()

    assert isinstance(custom_filter, SupportsDim)
    assert custom_filter.dim == 1
    assert len(estimates) == 3
    assert len(custom_filter.history) == 4
    assert custom_filter.get_point_estimate() == estimates[-1]
