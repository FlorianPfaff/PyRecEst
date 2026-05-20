"""Public estimator protocols used by examples, tools, and evaluators."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EstimatorProtocol(Protocol):
    """Structural protocol for recursive estimators.

    Implementations are expected to mutate their internal state during
    ``predict`` and ``update`` and expose a point estimate for evaluation and
    plotting utilities.
    """

    def predict(self, *args: Any, **kwargs: Any) -> None:
        """Advance the estimator state."""

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Assimilate a measurement or measurement set."""

    def get_point_estimate(self, *args: Any, **kwargs: Any) -> Any:
        """Return the current point estimate."""


@runtime_checkable
class DistributionBackedEstimatorProtocol(EstimatorProtocol, Protocol):
    """Estimator protocol for filters that expose a distributional state."""

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        """Return the current distributional state."""


__all__ = ["DistributionBackedEstimatorProtocol", "EstimatorProtocol"]
