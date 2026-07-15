"""Reusable model objects for grid-based filters.

The classes in this module are deliberately small wrappers around the two
capabilities grid filters need most often:

* evaluating a measurement likelihood on all grid points; and
* providing a precomputed transition density for grid-based prediction.

They are intentionally not tied to one concrete grid-filter class. Concrete
filters may still impose additional requirements on the transition-density
object, for example a manifold-specific conditional grid distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

GridLikelihoodFunction = Callable[[Any, Any], Any]
GridTransitionDensityFactory = Callable[[Any], Any]


def _ensure_callable(value: Any, name: str) -> None:
    if not callable(value):
        raise TypeError(f"{name} must be callable")


def _require_transition_density(value: Any, name: str) -> Any:
    if value is None:
        raise ValueError(f"{name} must not be None")
    return value


@dataclass(frozen=True)
class GridLikelihoodMeasurementModel:
    """Measurement model that evaluates likelihoods on a filter grid.

    Parameters
    ----------
    likelihood : callable
        Function with signature ``likelihood(measurement, grid)`` returning one
        non-negative likelihood value per grid point. The returned shape should
        be compatible with the grid distribution's ``grid_values``.

    Notes
    -----
    This model is useful for grid filters whose update step multiplies the
    current grid weights by likelihood values evaluated at the support grid.
    """

    likelihood: GridLikelihoodFunction

    def __post_init__(self) -> None:
        _ensure_callable(self.likelihood, "likelihood")

    def likelihood_values(self, measurement: Any, grid: Any) -> Any:
        """Evaluate the measurement likelihood on ``grid``."""
        return self.likelihood(measurement, grid)


@dataclass(frozen=True)
class GridTransitionDensityModel:
    """Transition model backed by a precomputed conditional grid density.

    Parameters
    ----------
    transition_density : object
        Conditional grid distribution or transition-density object understood by
        the concrete grid filter. For example, ``HyperhemisphericalGridFilter``
        expects an ``SdHalfCondSdHalfGridDistribution``.
    """

    transition_density: Any

    def __post_init__(self) -> None:
        _require_transition_density(self.transition_density, "transition_density")

    def transition_density_for_filter(self, _filter: Any) -> Any:
        """Return the precomputed transition density for ``_filter``."""
        return self.transition_density


@dataclass(frozen=True)
class GridTransitionDensityFactoryModel:
    """Transition model that lazily creates a conditional grid density.

    Parameters
    ----------
    transition_density_factory : callable
        Function with signature ``transition_density_factory(filter)`` returning
        a transition-density object compatible with the given filter.

    Notes
    -----
    This is useful when the density depends on the active filter grid or the
    number of grid points. The factory is called by the filter adapter at
    prediction time.
    """

    transition_density_factory: GridTransitionDensityFactory

    def __post_init__(self) -> None:
        _ensure_callable(self.transition_density_factory, "transition_density_factory")

    def transition_density_for_filter(self, filter_instance: Any) -> Any:
        """Create a transition density compatible with ``filter_instance``."""
        transition_density = self.transition_density_factory(filter_instance)
        return _require_transition_density(
            transition_density,
            "transition_density_factory result",
        )
