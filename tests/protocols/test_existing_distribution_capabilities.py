"""Capability smoke tests for existing public distribution classes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, ones
from pyrecest.distributions import (
    CircularUniformDistribution,
    GaussianDistribution,
    LinearDiracDistribution,
    VonMisesDistribution,
)
from pyrecest.protocols.common import SupportsDim, SupportsInputDim


@dataclass(frozen=True)
class DistributionCapabilityRow:
    """Expected capability snapshot for one concrete distribution instance."""

    name: str
    factory: Callable[[], Any]
    pdf_probe: Callable[[], Any]
    expected_capabilities: frozenset[str]


DISTRIBUTION_CAPABILITIES = frozenset(
    {
        "dim",
        "input_dim",
        "pdf",
        "ln_pdf",
        "sample",
        "mean",
        "covariance",
        "convert_to",
        "approximate_as",
        "from_distribution",
    }
)


DISTRIBUTION_ROWS = (
    DistributionCapabilityRow(
        name="GaussianDistribution",
        factory=lambda: GaussianDistribution(
            array([0.0, 1.0]), eye(2), check_validity=False
        ),
        pdf_probe=lambda: array([[0.0, 1.0]]),
        expected_capabilities=frozenset(
            {
                "dim",
                "input_dim",
                "pdf",
                "ln_pdf",
                "sample",
                "mean",
                "covariance",
                "convert_to",
                "approximate_as",
                "from_distribution",
            }
        ),
    ),
    DistributionCapabilityRow(
        name="LinearDiracDistribution",
        factory=lambda: LinearDiracDistribution(
            array([[0.0, 0.0], [1.0, 1.0]]), ones(2) / 2
        ),
        pdf_probe=lambda: array([[0.0, 0.0]]),
        expected_capabilities=frozenset(
            {
                "dim",
                "input_dim",
                "sample",
                "mean",
                "covariance",
                "convert_to",
                "approximate_as",
                "from_distribution",
            }
        ),
    ),
    DistributionCapabilityRow(
        name="VonMisesDistribution",
        factory=lambda: VonMisesDistribution(0.0, 1.0),
        pdf_probe=lambda: array([0.0]),
        expected_capabilities=frozenset(
            {
                "dim",
                "input_dim",
                "pdf",
                "ln_pdf",
                "sample",
                "mean",
                "convert_to",
                "approximate_as",
            }
        ),
    ),
    DistributionCapabilityRow(
        name="CircularUniformDistribution",
        factory=CircularUniformDistribution,
        pdf_probe=lambda: array([0.0]),
        expected_capabilities=frozenset(
            {
                "dim",
                "input_dim",
                "pdf",
                "ln_pdf",
                "sample",
                "convert_to",
                "approximate_as",
            }
        ),
    ),
)


def _call_succeeds(func: Callable[..., Any], *args: Any) -> bool:
    """Return whether a capability is usable for a smoke-test probe."""
    try:
        func(*args)
    except (NotImplementedError, ValueError):
        return False
    return True


def _callable_attribute(instance: Any, name: str) -> bool:
    return callable(getattr(instance, name, None))


def _class_callable(instance: Any, name: str) -> bool:
    return callable(getattr(type(instance), name, None))


def _observed_capabilities(row: DistributionCapabilityRow) -> frozenset[str]:
    distribution = row.factory()
    pdf_probe = row.pdf_probe()

    checks: dict[str, Callable[[], bool]] = {
        "dim": lambda: isinstance(distribution, SupportsDim)
        and getattr(distribution, "dim", None) is not None,
        "input_dim": lambda: isinstance(distribution, SupportsInputDim)
        and getattr(distribution, "input_dim", None) is not None,
        "pdf": lambda: _callable_attribute(distribution, "pdf")
        and _call_succeeds(distribution.pdf, pdf_probe),
        "ln_pdf": lambda: _callable_attribute(distribution, "ln_pdf")
        and _call_succeeds(distribution.ln_pdf, pdf_probe),
        "sample": lambda: _callable_attribute(distribution, "sample"),
        "mean": lambda: _callable_attribute(distribution, "mean")
        and _call_succeeds(distribution.mean),
        "covariance": lambda: _callable_attribute(distribution, "covariance")
        and _call_succeeds(distribution.covariance),
        "convert_to": lambda: _callable_attribute(distribution, "convert_to"),
        "approximate_as": lambda: _callable_attribute(distribution, "approximate_as"),
        "from_distribution": lambda: _class_callable(distribution, "from_distribution"),
    }

    return frozenset(capability for capability, check in checks.items() if check())


@pytest.mark.parametrize("row", DISTRIBUTION_ROWS, ids=lambda row: row.name)
def test_public_distribution_capability_matrix(row: DistributionCapabilityRow):
    observed = _observed_capabilities(row)

    assert observed == row.expected_capabilities
    assert observed <= DISTRIBUTION_CAPABILITIES
