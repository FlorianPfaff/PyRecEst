"""Custom distribution example for the public protocol seed.

The class in this example does not inherit from a PyRecEst distribution base
class. It exposes the currently public dimension protocols and uses method names
that follow common PyRecEst distribution conventions.
"""

from __future__ import annotations

from math import exp, pi, sqrt
from random import Random
from typing import Sequence

from pyrecest.protocols.common import SupportsDim, SupportsInputDim


class ScalarGaussianLikeDistribution:
    """Small scalar Gaussian-like distribution for protocol examples."""

    def __init__(self, mean: float, variance: float) -> None:
        if variance <= 0.0:
            raise ValueError("variance must be positive")

        self._mean = float(mean)
        self._variance = float(variance)

    @property
    def dim(self) -> int:
        """Intrinsic state-space dimension."""
        return 1

    @property
    def input_dim(self) -> int:
        """Ambient/input coordinate dimension accepted by ``pdf``."""
        return 1

    def pdf(self, xs: Sequence[float]) -> list[float]:
        """Evaluate the scalar density for a sequence of scalar inputs."""
        return [self._pdf_scalar(float(x)) for x in xs]

    def sample(self, n: int, *, rng: Random | None = None) -> list[float]:
        """Draw ``n`` scalar samples."""
        if n < 0:
            raise ValueError("n must be non-negative")

        generator = Random(0) if rng is None else rng  # nosec B311
        standard_deviation = sqrt(self._variance)
        return [generator.gauss(self._mean, standard_deviation) for _ in range(n)]

    def mean(self) -> float:
        """Return the scalar mean."""
        return self._mean

    def covariance(self) -> float:
        """Return the scalar variance as the one-dimensional covariance."""
        return self._variance

    def _pdf_scalar(self, x: float) -> float:
        normalization = 1.0 / sqrt(2.0 * pi * self._variance)
        exponent = -0.5 * ((x - self._mean) ** 2) / self._variance
        return normalization * exp(exponent)


def describe_intrinsic_dimension(obj: SupportsDim) -> str:
    """Return a human-readable intrinsic-dimension description."""
    return f"intrinsic dimension: {obj.dim}"


def describe_input_dimension(obj: SupportsInputDim) -> str:
    """Return a human-readable input-dimension description."""
    return f"input dimension: {obj.input_dim}"


def run_example() -> tuple[ScalarGaussianLikeDistribution, list[float], float]:
    """Create the custom distribution and evaluate its basic capabilities."""
    distribution = ScalarGaussianLikeDistribution(mean=1.0, variance=0.25)

    if not isinstance(distribution, SupportsDim):
        raise TypeError("distribution must expose an intrinsic dimension")
    if not isinstance(distribution, SupportsInputDim):
        raise TypeError("distribution must expose an input dimension")

    samples = distribution.sample(3, rng=Random(1))  # nosec B311
    density_at_mean = distribution.pdf([distribution.mean()])[0]
    return distribution, samples, density_at_mean


def main() -> None:
    """Print a compact demonstration of the custom distribution contract."""
    distribution, samples, density_at_mean = run_example()

    print("Custom distribution protocol example")
    print(describe_intrinsic_dimension(distribution))
    print(describe_input_dimension(distribution))
    print(f"mean: {distribution.mean():.3f}")
    print(f"covariance: {distribution.covariance():.3f}")
    print(f"density at mean: {density_at_mean:.3f}")
    print("samples:")
    for sample in samples:
        print(f"  {sample:.3f}")


if __name__ == "__main__":
    main()
