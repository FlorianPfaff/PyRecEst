"""Public filter capability protocols for PyRecEst components."""

# pylint: disable=unused-argument

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .common import ArrayLike, BackendArray, SupportsDim


@runtime_checkable
class SupportsFilterState(Protocol):
    """Object exposing a mutable posterior/filter-state representation."""

    @property
    def filter_state(self) -> Any:
        """Current posterior or internal filter-state representation."""
        raise NotImplementedError

    @filter_state.setter
    def filter_state(self, value: Any) -> None:
        """Replace the current posterior or internal filter-state representation."""
        raise NotImplementedError


@runtime_checkable
class SupportsPointEstimate(Protocol):
    """Object exposing a point estimate of its current filter state."""

    def get_point_estimate(self) -> BackendArray:
        """Return a point estimate for the current filter state."""
        raise NotImplementedError


@runtime_checkable
class SupportsHistoryRecording(Protocol):
    """Object that can append values to named filter histories."""

    def record_history(
        self,
        name: str,
        value: Any,
        pad_with_nan: bool = False,
        copy_value: bool = True,
    ) -> Any:
        """Append a value to a named history and return the updated history."""
        raise NotImplementedError


@runtime_checkable
class SupportsHistoryClearing(Protocol):
    """Object that can clear recorded filter histories."""

    def clear_history(self, name: str | None = None) -> None:
        """Clear a named history or all histories."""
        raise NotImplementedError


@runtime_checkable
class SupportsFilterStateRecording(Protocol):
    """Object that can record its current filter state."""

    def record_filter_state(self, history_name: str = "filter_state") -> Any:
        """Record the current filter state under ``history_name``."""
        raise NotImplementedError


@runtime_checkable
class SupportsPointEstimateRecording(Protocol):
    """Object that can record its current point estimate."""

    def record_point_estimate(self, history_name: str = "point_estimate") -> Any:
        """Record the current point estimate under ``history_name``."""
        raise NotImplementedError


@runtime_checkable
class SupportsFilterStatePlotting(Protocol):
    """Object that can plot its current filter state."""

    def plot_filter_state(self) -> Any:
        """Plot the current filter-state representation."""
        raise NotImplementedError


@runtime_checkable
class SupportsIdentityPredict(Protocol):
    """Object supporting an identity-transition prediction step."""

    def predict_identity(
        self,
        sys_noise_cov: ArrayLike,
        second_arg: ArrayLike | None = None,
        /,
    ) -> Any:
        """Predict with identity dynamics and an implementation-specific second argument.

        Existing PyRecEst filters use this method name with slightly different
        optional second-argument semantics. The protocol therefore requires only
        the shared positional shape of the capability.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsLinearPredict(Protocol):
    """Object supporting a linear prediction step."""

    def predict_linear(
        self,
        system_matrix: ArrayLike,
        sys_noise_cov: ArrayLike,
        sys_input: ArrayLike | None = None,
        /,
    ) -> Any:
        """Predict with a linear transition matrix and additive process noise."""
        raise NotImplementedError


@runtime_checkable
class SupportsLinearUpdate(Protocol):
    """Object supporting a linear measurement-update step."""

    def update_linear(
        self,
        measurement: ArrayLike,
        measurement_matrix: ArrayLike,
        measurement_noise: ArrayLike,
        /,
    ) -> Any:
        """Update with a linear measurement matrix and additive measurement noise."""
        raise NotImplementedError


@runtime_checkable
class SupportsIdentityUpdate(Protocol):
    """Object supporting an identity-measurement update step."""

    def update_identity(
        self,
        first_arg: ArrayLike,
        second_arg: ArrayLike,
        /,
    ) -> Any:
        """Update with an identity measurement map.

        Existing PyRecEst filters use this method name with different positional
        argument order. The protocol therefore exposes only the shared structural
        capability.
        """
        raise NotImplementedError


@runtime_checkable
class SupportsNonlinearPredict(Protocol):
    """Object supporting a nonlinear prediction step."""

    def predict_nonlinear(self, *args: Any, **kwargs: Any) -> Any:
        """Predict through implementation-specific nonlinear dynamics."""
        raise NotImplementedError


@runtime_checkable
class SupportsNonlinearUpdate(Protocol):
    """Object supporting a nonlinear measurement-update step."""

    def update_nonlinear(self, *args: Any, **kwargs: Any) -> Any:
        """Update through an implementation-specific nonlinear measurement model."""
        raise NotImplementedError


@runtime_checkable
class SupportsModelPredict(Protocol):
    """Object supporting prediction with a reusable transition-model object."""

    def predict_model(self, transition_model: Any, /) -> Any:
        """Predict using a transition model object."""
        raise NotImplementedError


@runtime_checkable
class SupportsModelUpdate(Protocol):
    """Object supporting update with a reusable measurement-model object."""

    def update_model(
        self,
        measurement_model: Any,
        measurement: ArrayLike,
        /,
    ) -> Any:
        """Update using a measurement model object and a measurement."""
        raise NotImplementedError


@runtime_checkable
class RecursiveFilterLike(
    SupportsDim,
    SupportsFilterState,
    SupportsPointEstimate,
    Protocol,
):
    """Minimal recursive-filter contract shared by most PyRecEst filters."""


@runtime_checkable
class LinearFilterLike(
    RecursiveFilterLike,
    SupportsLinearPredict,
    SupportsLinearUpdate,
    Protocol,
):
    """Recursive filter supporting linear prediction and update steps."""


@runtime_checkable
class NonlinearFilterLike(
    RecursiveFilterLike,
    SupportsNonlinearPredict,
    SupportsNonlinearUpdate,
    Protocol,
):
    """Recursive filter supporting nonlinear prediction and update steps."""


@runtime_checkable
class ModelBasedFilterLike(
    RecursiveFilterLike,
    SupportsModelPredict,
    SupportsModelUpdate,
    Protocol,
):
    """Recursive filter supporting model-object prediction and update steps."""


__all__ = [
    "LinearFilterLike",
    "ModelBasedFilterLike",
    "NonlinearFilterLike",
    "RecursiveFilterLike",
    "SupportsFilterState",
    "SupportsFilterStatePlotting",
    "SupportsFilterStateRecording",
    "SupportsHistoryClearing",
    "SupportsHistoryRecording",
    "SupportsIdentityPredict",
    "SupportsIdentityUpdate",
    "SupportsLinearPredict",
    "SupportsLinearUpdate",
    "SupportsModelPredict",
    "SupportsModelUpdate",
    "SupportsNonlinearPredict",
    "SupportsNonlinearUpdate",
    "SupportsPointEstimate",
    "SupportsPointEstimateRecording",
]
