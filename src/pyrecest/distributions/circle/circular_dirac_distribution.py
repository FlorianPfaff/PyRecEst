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
            # CircularDiracDistribution already performs the hardened Dirac-weight
            # validation and normalization in its constructor. Avoid pre-normalizing
            # here: direct division by a near-maximum backend value can be lowered
            # to multiplication by an underflowed reciprocal on JAX/XLA, turning
            # valid finite grid weights into zeros/NaNs before the robust normalizer
            # gets a chance to handle them.
            return cls(get_grid(), weights)

        if n_particles is None:
            raise ValueError("n_particles is required for sampling-based conversion.")
        n_particles = cls._validate_particle_count(n_particles)
        return cls(
            distribution.sample(n_particles), ones(n_particles) / n_particles
        )

    def plot_interpolated(self, _=None):
        """Raise because interpolation is unavailable for Dirac distributions."""
        raise NotImplementedError(
            "Interpolation is not available for CircularDiracDistribution."
        )
