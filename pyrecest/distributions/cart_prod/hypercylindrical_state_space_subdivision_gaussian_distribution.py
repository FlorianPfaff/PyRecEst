from pyrecest.distributions.cart_prod.state_space_subdivision_gaussian_distribution import StateSpaceSubdivisionGaussianDistribution
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_distribution import HypercylindricalStateSpaceSubdivisionDistribution
from pyrecest.distributions import GaussianDistribution
import numpy as np


class HypercylindricalStateSpaceSubdivisionGaussianDistribution(StateSpaceSubdivisionGaussianDistribution, HypercylindricalStateSpaceSubdivisionDistribution):

    def __init__(self, gd_, gaussians):
        HypercylindricalStateSpaceSubdivisionDistribution.__init__(self, gd_, gaussians)
        StateSpaceSubdivisionGaussianDistribution.__init__(self, gd_, gaussians)

    def mode(self):
        return StateSpaceSubdivisionGaussianDistribution.mode(self)

    def linear_mean(self):
        return StateSpaceSubdivisionGaussianDistribution.linear_mean(self)

    def linear_covariance(self):
        return StateSpaceSubdivisionGaussianDistribution.linear_covariance(self)

    def hybrid_mean(self):
        return StateSpaceSubdivisionGaussianDistribution.hybrid_mean(self)

    def hybrid_moment(self):
        trig_mom_complex = self.gd.trigonometric_moment(1)
        trig_mom_real = [trig_mom_complex.real, trig_mom_complex.imag]
        return trig_mom_real + [self.linear_mean()]

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type='CartesianProd'):
        # Even for WN, the conditional is a HypercylindricalWN divided
        # by a WN. No analytical formula is known to me
        hcrbd_non_gauss = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            distribution.pdf, no_of_grid_points, distribution.linD, distribution.boundD, grid_type)
        # Superclass generates CustomLinearDistributions. Convert.
        lin_dists = [GaussianDistribution.from_distribution(dist) for dist in hcrbd_non_gauss.linear_distributions]
        return HypercylindricalStateSpaceSubdivisionGaussianDistribution(hcrbd_non_gauss.gd, lin_dists)

    @staticmethod
    def from_function(fun, no_of_grid_points, dim_lin, dim_bound=1, grid_type='CartesianProd', int_range=(-np.inf , np.inf)):
        hcrbd_non_gauss = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            fun, no_of_grid_points, dim_lin, dim_bound, grid_type, int_range)
        return HypercylindricalStateSpaceSubdivisionGaussianDistribution(hcrbd_non_gauss.gd, lin_dists)


"""
class HypercylindricalStateSpaceSubdivisionGaussianDistribution:
    def __init__(self, gd_, gaussians):
        # In this class, the grid must be on the hypertorus.
        self.gd_ = HypertoroidalGridDistribution(gd_)
        self.gaussians = gaussians

    def mode(self):
        pass

    def linear_mean(self):
        pass

    def linear_covariance(self):
        pass

    def hybrid_mean(self):
        pass

    def hybrid_moment(self):
        pass

    @classmethod
    def from_distribution(cls, distribution, no_of_grid_points, grid_type='CartesianProd'):
        pass

    @classmethod
    def from_function(cls, fun, no_of_grid_points, dim_lin, dim_bound=1, grid_type='CartesianProd', int_range=(-np.inf, np.inf)):
        pass
        """