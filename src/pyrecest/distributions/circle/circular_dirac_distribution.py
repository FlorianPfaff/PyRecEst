from pyrecest.backend import reshape

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
        assert self.d.shape == self.w.shape, "The shapes of d and w should match."

    def plot_interpolated(self, _):
        """
        Raises an exception since interpolation is not available for WDDistribution.
        """
        raise NotImplementedError("No interpolation available for WDDistribution.")
