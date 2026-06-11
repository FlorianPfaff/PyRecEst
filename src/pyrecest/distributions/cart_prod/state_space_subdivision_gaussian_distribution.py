import copy
import warnings

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
)
from pyrecest.backend import any as backend_any
from pyrecest.backend import (
    argmax,
    array,
    asarray,
    concatenate,
    stack,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    zeros,
)

from ..nonperiodic.gaussian_distribution import GaussianDistribution
from ..nonperiodic.gaussian_mixture import GaussianMixture
from .state_space_subdivision_distribution import StateSpaceSubdivisionDistribution


class StateSpaceSubdivisionGaussianDistribution(StateSpaceSubdivisionDistribution):
    """
    Joint distribution over a Cartesian product of a grid-based
    (periodic/bounded) space and a linear space where every conditional
    linear distribution is a Gaussian.

    The periodic part is a grid distribution (e.g. HypertoroidalGridDistribution
    or HyperhemisphericalGridDistribution).  The linear part is a list of
    GaussianDistribution objects, one per grid point.
    """

    def __init__(self, gd, gaussians):
        """
        Parameters
        ----------
        gd : AbstractGridDistribution
            Grid-based distribution for the periodic/bounded part.
        gaussians : list of GaussianDistribution
            One Gaussian per grid point of *gd*.
        """
        if not all(isinstance(g, GaussianDistribution) for g in gaussians):
            raise TypeError(
                "All elements of gaussians must be GaussianDistribution instances."
            )
        super().__init__(gd, gaussians)

    # ------------------------------------------------------------------
    # Marginalisation
    # ------------------------------------------------------------------

    def marginalize_linear(self):
        """Return the grid distribution (marginalised over the linear part)."""
        return copy.deepcopy(self.gd)

    def marginalize_periodic(self):
        """
        Marginalise over the periodic/bounded dimensions.

        Returns a GaussianMixture whose components are the conditional
        Gaussians and whose weights are the (normalised) grid values.
        """
        weights = self.gd.grid_values / backend_sum(self.gd.grid_values)
        return GaussianMixture(list(self.linear_distributions), weights)

    # ------------------------------------------------------------------
    # Linear moments
    # ------------------------------------------------------------------

    def linear_mean(self):
        """
        Compute the mean of the marginal linear distribution by treating
        the state as a Gaussian mixture.

        Returns
        -------
        mu : array, shape (lin_dim,)
        """
        means = array([ld.mu for ld in self.linear_distributions])  # (n, lin_dim)
        covs = stack(
            [ld.C for ld in self.linear_distributions], axis=2
        )  # (lin_dim, lin_dim, n)
        weights = self.gd.grid_values / backend_sum(self.gd.grid_values)
        mu, _ = GaussianMixture.mixture_parameters_to_gaussian_parameters(
            means, covs, weights
        )
        return mu

    def linear_covariance(self):
        """
        Compute the covariance of the marginal linear distribution by treating
        the state as a Gaussian mixture.

        Returns
        -------
        C : array, shape (lin_dim, lin_dim)
        """
        means = array([ld.mu for ld in self.linear_distributions])  # (n, lin_dim)
        covs = stack(
            [ld.C for ld in self.linear_distributions], axis=2
        )  # (lin_dim, lin_dim, n)
        weights = self.gd.grid_values / backend_sum(self.gd.grid_values)
        _, C = GaussianMixture.mixture_parameters_to_gaussian_parameters(
            means, covs, weights
        )
        return C

    # ------------------------------------------------------------------
    # Multiplication
    # ------------------------------------------------------------------

    def multiply(self, other):
        """
        Multiply two StateSpaceSubdivisionGaussianDistributions.

        Both operands must be defined on the same grid.  For each grid point
        the conditional Gaussians are multiplied (Bayesian update).  The grid
        weights are updated by the likelihood factors that arise from the
        overlap of the two conditional Gaussians.

        Parameters
        ----------
        other : StateSpaceSubdivisionGaussianDistribution

        Returns
        -------
        StateSpaceSubdivisionGaussianDistribution
        """
        if not isinstance(other, StateSpaceSubdivisionGaussianDistribution):
            raise TypeError(
                "other must be a StateSpaceSubdivisionGaussianDistribution."
            )
        if self.gd.n_grid_points != other.gd.n_grid_points:
            raise ValueError(
                "Can only multiply distributions defined on grids with the same "
                "number of grid points."
            )
        self_grid = asarray(self.gd.get_grid())
        other_grid = asarray(other.gd.get_grid())
        if not allclose(self_grid, other_grid):
            raise ValueError("Can only multiply for equal grids.")

        n = len(self.linear_distributions)
        new_linear_distributions = []
        pdf_values = []

        for i in range(n):
            ld_self = self.linear_distributions[i]
            ld_other = other.linear_distributions[i]

            # The likelihood factor for grid point i is the pdf of
            # N(mu_self_i, C_self_i + C_other_i) evaluated at mu_other_i.
            # This is equivalent to N(0, C_self_i + C_other_i) at 0.
            combined_cov = ld_self.C + ld_other.C
            temp_g = GaussianDistribution(
                ld_other.mu, combined_cov, check_validity=False
            )
            pdf_values.append(temp_g.pdf(ld_self.mu))

            new_linear_distributions.append(ld_self.multiply(ld_other))

        # Build a 1-D factors array.  pdf() may return shape () or (1,) depending
        # on backend and Gaussian dimension; reshape each value to (1,) before
        # concatenating so the result is always shape (n,).
        factors_linear = concatenate([asarray(v).reshape((1,)) for v in pdf_values])

        # Build result
        result = copy.deepcopy(self)
        result.linear_distributions = new_linear_distributions
        result.gd = copy.deepcopy(self.gd)
        result.gd.grid_values = (
            self.gd.grid_values * other.gd.grid_values * array(factors_linear)
        )
        result.gd.normalize_in_place(warn_unnorm=False)
        return result

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def mode(self):
        """
        Compute the (approximate) joint mode.

        The mode is found by maximising the product of the conditional
        Gaussian peak value and the grid weight at each grid point.  Only
        the discrete grid is searched (no interpolation).

        Returns
        -------
        m : array, shape (bound_dim + lin_dim,)
            Concatenation of the periodic mode (grid point) and the linear
            mode (mean of the conditional Gaussian at that grid point).

        Warns
        -----
        UserWarning
            If the density appears multimodal (i.e. another grid point has a
            joint value within a factor of 1.001 of the maximum).
        """
        lin_dim = self.linear_distributions[0].dim
        zeros_d = zeros(lin_dim)

        # Peak value of N(mu_i, C_i) depends only on C_i; it equals
        # N(0 | 0, C_i).  We evaluate each conditional Gaussian at its own
        # mean to obtain the maximum pdf value.
        peak_vals = array(
            [
                float(
                    GaussianDistribution(zeros_d, ld.C, check_validity=False).pdf(
                        zeros_d
                    )
                )
                for ld in self.linear_distributions
            ]
        )

        fun_vals_joint = peak_vals * asarray(self.gd.grid_values)
        index = int(argmax(fun_vals_joint))
        max_val = float(fun_vals_joint[index])

        # Remove the maximum entry to check for multimodality
        remaining = concatenate(
            [fun_vals_joint[:index], fun_vals_joint[index + 1 :]]  # noqa: E203
        )
        if len(remaining) > 0 and (
            backend_any((max_val - remaining) < 1e-15)
            or backend_any((max_val / remaining) < 1.001)
        ):
            warnings.warn(
                "Density may not be unimodal. However, this can also be caused "
                "by a high grid resolution and thus very similar function values "
                "at the grid points.",
                UserWarning,
                stacklevel=2,
            )

        periodic_mode = self.gd.get_grid_point(index)  # shape (bound_dim,)
        linear_mode = self.linear_distributions[index].mu  # shape (lin_dim,)
        return concatenate([periodic_mode.reshape(-1), linear_mode.reshape(-1)])

    # ------------------------------------------------------------------
    # Unsupported operations
    # ------------------------------------------------------------------

    def convolve(self, _other):
        raise NotImplementedError(
            "convolve is not supported for "
            "StateSpaceSubdivisionGaussianDistribution."
        )
