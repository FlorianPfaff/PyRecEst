"""Smoke tests for public filter capability protocols."""

from __future__ import annotations

from pyrecest.protocols.filters import (
    LinearFilterLike,
    ModelBasedFilterLike,
    NonlinearFilterLike,
    RecursiveFilterLike,
    SupportsFilterState,
    SupportsFilterStatePlotting,
    SupportsFilterStateRecording,
    SupportsHistoryClearing,
    SupportsHistoryRecording,
    SupportsIdentityPredict,
    SupportsIdentityUpdate,
    SupportsLinearPredict,
    SupportsLinearUpdate,
    SupportsModelPredict,
    SupportsModelUpdate,
    SupportsNonlinearPredict,
    SupportsNonlinearUpdate,
    SupportsPointEstimate,
    SupportsPointEstimateRecording,
)


class DemoDistribution:
    """Small distribution-like state object for protocol smoke tests."""

    dim = 2

    def mean(self):
        return (0.0, 0.0)

    def plot(self):
        return None


class DemoRecursiveFilter:
    """Small recursive-filter-like object matching AbstractFilter conventions."""

    def __init__(self):
        self._filter_state = DemoDistribution()
        self.history = {}

    @property
    def dim(self):
        return self.filter_state.dim

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, value):
        self._filter_state = value

    def get_point_estimate(self):
        return self.filter_state.mean()

    def record_history(self, name, value, pad_with_nan=False, copy_value=True):
        self.history[name] = (value, pad_with_nan, copy_value)
        return self.history[name]

    def clear_history(self, name=None):
        if name is None:
            self.history.clear()
        else:
            self.history.pop(name, None)

    def record_filter_state(self, history_name="filter_state"):
        return self.record_history(history_name, self.filter_state)

    def record_point_estimate(self, history_name="point_estimate"):
        return self.record_history(history_name, self.get_point_estimate())

    def plot_filter_state(self):
        return self.filter_state.plot()


class DemoLinearFilter(DemoRecursiveFilter):
    """Small linear-filter-like object for structural protocol checks."""

    def predict_identity(self, sys_noise_cov, sys_input=None):
        return (sys_noise_cov, sys_input)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        return (system_matrix, sys_noise_cov, sys_input)

    def update_identity(self, meas_noise, measurement):
        return (meas_noise, measurement)

    def update_linear(self, measurement, measurement_matrix, meas_noise):
        return (measurement, measurement_matrix, meas_noise)


class DemoNonlinearFilter(DemoRecursiveFilter):
    """Small nonlinear-filter-like object for structural protocol checks."""

    def predict_nonlinear(self, *args, **kwargs):
        return (args, kwargs)

    def update_nonlinear(self, *args, **kwargs):
        return (args, kwargs)


class DemoModelBasedFilter(DemoRecursiveFilter):
    """Small model-based-filter-like object for structural protocol checks."""

    def predict_model(self, transition_model):
        return transition_model

    def update_model(self, measurement_model, measurement):
        return (measurement_model, measurement)


def test_recursive_filter_protocols_are_runtime_checkable():
    filter_ = DemoRecursiveFilter()

    assert isinstance(filter_, SupportsFilterState)
    assert isinstance(filter_, SupportsPointEstimate)
    assert isinstance(filter_, SupportsHistoryRecording)
    assert isinstance(filter_, SupportsHistoryClearing)
    assert isinstance(filter_, SupportsFilterStateRecording)
    assert isinstance(filter_, SupportsPointEstimateRecording)
    assert isinstance(filter_, SupportsFilterStatePlotting)
    assert isinstance(filter_, RecursiveFilterLike)


def test_linear_filter_protocols_are_runtime_checkable():
    filter_ = DemoLinearFilter()

    assert isinstance(filter_, SupportsIdentityPredict)
    assert isinstance(filter_, SupportsLinearPredict)
    assert isinstance(filter_, SupportsIdentityUpdate)
    assert isinstance(filter_, SupportsLinearUpdate)
    assert isinstance(filter_, LinearFilterLike)


def test_nonlinear_filter_protocols_are_runtime_checkable():
    filter_ = DemoNonlinearFilter()

    assert isinstance(filter_, SupportsNonlinearPredict)
    assert isinstance(filter_, SupportsNonlinearUpdate)
    assert isinstance(filter_, NonlinearFilterLike)


def test_model_based_filter_protocols_are_runtime_checkable():
    filter_ = DemoModelBasedFilter()

    assert isinstance(filter_, SupportsModelPredict)
    assert isinstance(filter_, SupportsModelUpdate)
    assert isinstance(filter_, ModelBasedFilterLike)


def test_object_without_filter_methods_does_not_match_filter_protocols():
    obj = object()

    assert not isinstance(obj, SupportsFilterState)
    assert not isinstance(obj, SupportsPointEstimate)
    assert not isinstance(obj, SupportsLinearPredict)
    assert not isinstance(obj, RecursiveFilterLike)
