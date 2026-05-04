"""Tests for reusable protocol-compliance helpers."""

from __future__ import annotations

import pytest
from pyrecest.protocols.common import SupportsDim
from pyrecest.protocols.testing import (
    ProtocolAssertionError,
    assert_callable_attribute,
    assert_filter_basic_contract,
    assert_has_attribute,
    assert_method_returns_non_none,
    assert_protocol_instance,
    assert_shape,
    assert_shape_prefix,
    assert_supports_covariance,
    assert_supports_dim,
    assert_supports_input_dim,
    assert_supports_likelihood,
    assert_supports_ln_pdf,
    assert_supports_log_likelihood,
    assert_supports_mean,
    assert_supports_pdf,
    assert_supports_sampling,
    assert_supports_transition_density,
    assert_supports_transition_sampling,
    assert_trailing_dimension,
    assert_value_is_not_none,
)


class ArrayLikeObject:
    def __init__(self, shape):
        self.shape = shape


class ObjectWithDimensions:
    dim = 2
    input_dim = 3


class ObjectWithMethod:
    value = 1

    def method(self):
        return self.value

    def none_method(self):
        return None


class DemoDistribution:
    dim = 2
    input_dim = 2

    def pdf(self, xs):
        return ArrayLikeObject((len(xs),))

    def ln_pdf(self, xs):
        return ArrayLikeObject((len(xs),))

    def sample(self, n):
        return ArrayLikeObject((n, self.input_dim))

    def mean(self):
        return ArrayLikeObject((self.input_dim,))

    def covariance(self):
        return ArrayLikeObject((self.input_dim, self.input_dim))


class DemoFilter:
    dim = 2
    filter_state = object()

    def get_point_estimate(self):
        return ArrayLikeObject((self.dim,))


class DemoLikelihoodModel:
    def likelihood(self, measurement, state):
        return (measurement, state)

    def log_likelihood(self, measurement, state):
        return (measurement, state, "log")


class DemoTransitionModel:
    def sample_next(self, state, n=1):
        return ArrayLikeObject((n, len(state)))

    def transition_density(self, state_next, state_previous):
        return (state_next, state_previous)


def test_assert_protocol_instance_accepts_runtime_checkable_protocol():
    assert_protocol_instance(ObjectWithDimensions(), SupportsDim)


def test_assert_protocol_instance_reports_missing_protocol_attribute():
    with pytest.raises(ProtocolAssertionError, match="SupportsDim"):
        assert_protocol_instance(object(), SupportsDim)


def test_attribute_helpers_return_values():
    obj = ObjectWithMethod()

    assert assert_has_attribute(obj, "value") == 1
    assert assert_callable_attribute(obj, "method")() == 1
    assert assert_method_returns_non_none(obj, "method") == 1


def test_attribute_helpers_report_missing_or_non_callable_attributes():
    obj = ObjectWithMethod()

    with pytest.raises(ProtocolAssertionError):
        assert_has_attribute(obj, "missing")
    with pytest.raises(ProtocolAssertionError):
        assert_callable_attribute(obj, "value")
    with pytest.raises(ProtocolAssertionError):
        assert_method_returns_non_none(obj, "none_method")


def test_value_helper_rejects_none():
    assert assert_value_is_not_none(1) == 1

    with pytest.raises(ProtocolAssertionError):
        assert_value_is_not_none(None)


def test_shape_helpers_return_actual_shape():
    value = ArrayLikeObject((2, 3))

    assert assert_shape(value, (2, 3)) == (2, 3)
    assert assert_shape_prefix(value, (2,)) == (2, 3)
    assert assert_trailing_dimension(value, 3) == (2, 3)


def test_shape_helpers_report_mismatches():
    value = ArrayLikeObject((2, 3))

    with pytest.raises(ProtocolAssertionError):
        assert_shape(value, (3, 2))
    with pytest.raises(ProtocolAssertionError):
        assert_shape_prefix(value, (3,))
    with pytest.raises(ProtocolAssertionError):
        assert_trailing_dimension(value, 2)
    with pytest.raises(ProtocolAssertionError):
        assert_shape(object(), ())


def test_common_dimension_helpers_return_dimensions():
    obj = ObjectWithDimensions()

    assert assert_supports_dim(obj) == 2
    assert assert_supports_input_dim(obj) == 3


def test_common_dimension_helpers_reject_invalid_dimensions():
    class BadDim:
        dim = -1

    with pytest.raises(ProtocolAssertionError):
        assert_supports_dim(BadDim())


def test_distribution_helpers_return_method_results():
    distribution = DemoDistribution()
    xs = [(0.0, 0.0), (1.0, 1.0)]

    assert_shape(assert_supports_pdf(distribution, xs), (2,))
    assert_shape(assert_supports_ln_pdf(distribution, xs), (2,))
    assert_shape(assert_supports_sampling(distribution, 4), (4, 2))
    assert_shape(assert_supports_mean(distribution), (2,))
    assert_shape(assert_supports_covariance(distribution), (2, 2))


def test_filter_basic_contract_returns_point_estimate():
    assert_shape(assert_filter_basic_contract(DemoFilter()), (2,))


def test_model_helpers_return_method_results():
    measurement = (1.0,)
    state = (2.0, 3.0)
    next_state = (2.5, 3.5)

    likelihood_model = DemoLikelihoodModel()
    transition_model = DemoTransitionModel()

    assert assert_supports_likelihood(likelihood_model, measurement, state) == (
        measurement,
        state,
    )
    assert assert_supports_log_likelihood(likelihood_model, measurement, state) == (
        measurement,
        state,
        "log",
    )
    assert_shape(
        assert_supports_transition_sampling(transition_model, state, 3), (3, 2)
    )
    assert assert_supports_transition_density(transition_model, next_state, state) == (
        next_state,
        state,
    )
