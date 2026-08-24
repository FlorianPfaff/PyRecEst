from pyrecest.backend import ones, reshape

from ..hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CircularDiracDistribution(
    HypertoroidalDiracDistribution, AbstractCircularDistribution
):
    def __init__(self, d, w=None):
        """
        Initializes a CircularDiracDistribution instance.

        Args:
            d (): The Dirac locations.
            w (Optional[]): The weights for each Dirac location.
        """
        super().__init__(
            d, w, dim=1
        )  # Necessary so it is clear that the dimension is 1.
        self.d = reshape(self.d, (-1,))
        if self.d.shape != self.w.shape:
            raise ValueError("The shapes of d and w should match.")

    @classmethod
    def from_distribution(
        cls, distribution: AbstractCircularDistribution, n_particles: int | None = None
    ):
        """Create a circular Dirac approximation from a circular distribution."""
        if not isinstance(distribution, AbstractCircularDistribution):
            raise ValueError(
                "from_distribution: invalidObject: First argument has to be "
                "a circular distribution."
            )

        get_grid = getattr(distribution, "get_grid", None)
        if hasattr(distribution, "grid_values") and callable(get_grid):
            weights = reshape(distribution.grid_values, (-1,))
            # Reuse the hardened Dirac-weight normalizer instead of dividing by
            # the backend maximum directly. On JAX/XLA, division by a near-maximum
            # finite value can be lowered to multiplication by an underflowed
            # reciprocal, turning valid grid weights into zeros/NaNs.
            weights = cls._normalized_weights(weights)
            return cls(get_grid(), weights)

        if n_particles is None:
            raise ValueError("n_particles is required for sampling-based conversion.")
        n_particles = cls._validate_particle_count(n_particles)
        return cls(distribution.sample(n_particles), ones(n_particles) / n_particles)

    def plot_interpolated(self, _=None):
        """Raise because interpolation is unavailable for Dirac distributions."""
        raise NotImplementedError(
            "Interpolation is not available for CircularDiracDistribution."
        )
