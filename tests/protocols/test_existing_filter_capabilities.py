"""Capability smoke tests for existing public filter classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.circular_particle_filter import CircularParticleFilter
from pyrecest.filters.kalman_filter import KalmanFilter
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter
from pyrecest.protocols.common import SupportsDim


@dataclass(frozen=True)
class ClassCapability:
    """A class-level capability identified by a public attribute name."""

    name: str
    attribute: str
    must_be_callable: bool = True


@dataclass(frozen=True)
class FilterCapabilityRow:
    """Expected capability snapshot for one public filter class."""

    cls: type
    expected_capabilities: frozenset[str]

    @property
    def name(self) -> str:
        return self.cls.__name__


FILTER_CAPABILITIES = (
    ClassCapability("dim", "dim", must_be_callable=False),
    ClassCapability("filter_state", "filter_state", must_be_callable=False),
    ClassCapability("get_point_estimate", "get_point_estimate"),
    ClassCapability("predict_linear", "predict_linear"),
    ClassCapability("update_linear", "update_linear"),
    ClassCapability("predict_nonlinear", "predict_nonlinear"),
    ClassCapability("update_nonlinear", "update_nonlinear"),
    ClassCapability("predict_model", "predict_model"),
    ClassCapability("update_model", "update_model"),
    ClassCapability(
        "update_nonlinear_using_likelihood", "update_nonlinear_using_likelihood"
    ),
    ClassCapability("record_filter_state", "record_filter_state"),
    ClassCapability("record_point_estimate", "record_point_estimate"),
)


FILTER_ROWS = (
    FilterCapabilityRow(
        cls=KalmanFilter,
        expected_capabilities=frozenset(
            {
                "dim",
                "filter_state",
                "get_point_estimate",
                "predict_linear",
                "update_linear",
                "predict_model",
                "update_model",
                "record_filter_state",
                "record_point_estimate",
            }
        ),
    ),
    FilterCapabilityRow(
        cls=UnscentedKalmanFilter,
        expected_capabilities=frozenset(
            {
                "dim",
                "filter_state",
                "get_point_estimate",
                "predict_linear",
                "update_linear",
                "predict_nonlinear",
                "update_nonlinear",
                "predict_model",
                "update_model",
                "record_filter_state",
                "record_point_estimate",
            }
        ),
    ),
    FilterCapabilityRow(
        cls=CircularParticleFilter,
        expected_capabilities=frozenset(
            {
                "dim",
                "filter_state",
                "get_point_estimate",
                "predict_nonlinear",
                "predict_model",
                "update_model",
                "update_nonlinear_using_likelihood",
                "record_filter_state",
                "record_point_estimate",
            }
        ),
    ),
)


def _class_attribute(cls: type, name: str) -> Any:
    """Return a class attribute without instantiating the class."""
    return getattr(cls, name, None)


def _has_class_capability(cls: type, capability: ClassCapability) -> bool:
    attribute = _class_attribute(cls, capability.attribute)
    if attribute is None:
        return False
    if capability.must_be_callable:
        return callable(attribute)
    return True


def _observed_capabilities(row: FilterCapabilityRow) -> frozenset[str]:
    return frozenset(
        capability.name
        for capability in FILTER_CAPABILITIES
        if _has_class_capability(row.cls, capability)
    )


@pytest.mark.parametrize("row", FILTER_ROWS, ids=lambda row: row.name)
def test_public_filter_capability_matrix(row: FilterCapabilityRow):
    observed = _observed_capabilities(row)

    assert observed == row.expected_capabilities


def test_kalman_filter_satisfies_common_dimension_protocol_at_runtime():
    filter_ = KalmanFilter(
        GaussianDistribution(array([0.0]), array([[1.0]]), check_validity=False)
    )

    assert isinstance(filter_, SupportsDim)
    assert filter_.dim == 1
    assert filter_.filter_state.dim == 1
    assert filter_.get_point_estimate() is not None
