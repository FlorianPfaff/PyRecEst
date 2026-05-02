"""Custom filter example for the public protocol seed.

The filter in this example does not inherit from a PyRecEst filter base class.
It exposes the currently public dimension protocol and follows common PyRecEst
filter naming conventions such as ``filter_state`` and ``get_point_estimate``.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrecest.protocols.common import SupportsDim


@dataclass(frozen=True)
class ScalarGaussianState:
    """Gaussian-like scalar state summary."""

    estimate: float
    variance: float


class ScalarRecursiveFilter:
    """Small scalar recursive filter for protocol examples."""

    def __init__(self, initial_estimate: float, initial_variance: float) -> None:
        if initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")

        self.filter_state = ScalarGaussianState(
            estimate=float(initial_estimate),
            variance=float(initial_variance),
        )
        self._history = [self.filter_state]

    @property
    def dim(self) -> int:
        """Intrinsic state-space dimension."""
        return 1

    @property
    def history(self) -> tuple[ScalarGaussianState, ...]:
        """Immutable view of filter states after construction and updates."""
        return tuple(self._history)

    def predict(self, process_variance: float) -> None:
        """Propagate the scalar state with additive process variance."""
        if process_variance < 0.0:
            raise ValueError("process_variance must be non-negative")

        self.filter_state = ScalarGaussianState(
            estimate=self.filter_state.estimate,
            variance=self.filter_state.variance + float(process_variance),
        )

    def update(self, measurement: float, measurement_variance: float) -> None:
        """Condition the scalar state on a scalar measurement."""
        if measurement_variance <= 0.0:
            raise ValueError("measurement_variance must be positive")

        prior_estimate = self.filter_state.estimate
        prior_variance = self.filter_state.variance
        kalman_gain = prior_variance / (prior_variance + measurement_variance)
        posterior_estimate = prior_estimate + kalman_gain * (
            float(measurement) - prior_estimate
        )
        posterior_variance = (1.0 - kalman_gain) * prior_variance

        self.filter_state = ScalarGaussianState(
            estimate=posterior_estimate,
            variance=posterior_variance,
        )
        self._history.append(self.filter_state)

    def get_point_estimate(self) -> float:
        """Return the current scalar point estimate."""
        return self.filter_state.estimate


def describe_filter_dimension(filter_obj: SupportsDim) -> str:
    """Return a human-readable intrinsic-dimension description."""
    return f"filter dimension: {filter_obj.dim}"


def run_example() -> tuple[ScalarRecursiveFilter, list[float]]:
    """Run a short predict/update sequence with the custom filter."""
    custom_filter = ScalarRecursiveFilter(
        initial_estimate=0.0,
        initial_variance=1.0,
    )

    if not isinstance(custom_filter, SupportsDim):
        raise TypeError("custom_filter must expose an intrinsic dimension")

    estimates = []
    for measurement in [0.8, 1.1, 0.9]:
        custom_filter.predict(process_variance=0.05)
        custom_filter.update(measurement, measurement_variance=0.25)
        estimates.append(custom_filter.get_point_estimate())

    return custom_filter, estimates


def main() -> None:
    """Print a compact demonstration of the custom filter contract."""
    custom_filter, estimates = run_example()

    print("Custom filter protocol example")
    print(describe_filter_dimension(custom_filter))
    print("step  estimate  variance")
    for step, state in enumerate(custom_filter.history[1:], start=1):
        print(f"{step:>4}  {state.estimate:>8.3f}  {state.variance:>8.3f}")

    print("point estimates:")
    for estimate in estimates:
        print(f"  {estimate:.3f}")


if __name__ == "__main__":
    main()
