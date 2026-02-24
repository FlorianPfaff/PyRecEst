import copy
from .abstract_lin_periodic_cart_prod_distribution import AbstractLinPeriodicCartProdDistribution
from ..abstract_grid_distribution import AbstractGridDistribution
from ..hypertorus.hypertoroidal_grid_distribution import HypertoroidalGridDistribution
from ..nonperiodic.linear_mixture import LinearMixture
from ..hypertorus.abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from ..hypersphere_subset.abstract_hypersphere_subset_distribution import AbstractHypersphereSubsetDistribution
from ..abstract_distribution_type import AbstractDistributionType

import numpy as np

class StateSpaceSubdivisionDistribution(AbstractLinPeriodicCartProdDistribution, AbstractDistributionType):
    def __init__(self, gd: AbstractGridDistribution, linear_distributions: list):
        assert len(linear_distributions) == len(gd.grid_values), "The number of linear distributions and grid points must be equal."
        assert np.ndim(linear_distributions) <= 1
        AbstractLinPeriodicCartProdDistribution.__init__(self, gd.dim, linear_distributions[0].dim)
        self.gd = gd
        self.linear_distributions = linear_distributions
        self.normalize()
        
    @property
    def input_dim(self):        
        if isinstance(self.gd, AbstractHypertoroidalDistribution):
            return self.bound_dim + self.lin_dim
        
        if isinstance(self.gd, AbstractHypersphereSubsetDistribution):
            return self.bound_dim + self.lin_dim + 1
        
        raise ValueError("Class unsupported")

    def sample(self, n):
        samples_bounded = self.gd.sample(n)
        _, indices = self.gd.get_closest_point(samples_bounded)
        relevant_grid_ids = np.unique(indices)
        samples_linear = np.full((self.lin_dim, n), np.nan)
        for curr_grid_ind in relevant_grid_ids:
            curr_xa_indices = indices == curr_grid_ind
            samples_linear[:, curr_xa_indices] = self.linear_distributions[curr_grid_ind].sample(np.sum(curr_xa_indices))
        return np.vstack((samples_bounded, samples_linear))

    def marginalize_linear(self):
        return self.gd

    def marginalize_periodic(self):
        linear_mixture = LinearMixture([ld for ld in self.linear_distributions], self.gd.grid_values / np.sum(self.gd.grid_values))
        return linear_mixture

    def pdf(self, xs, lin_dist_interpolation_method='', bound_dist_interpolationmethod='grid_default'):
        assert xs.shape[0] == self.dim, "Dimension of xa does not match the density's dimension."
        if not lin_dist_interpolation_method:
            if isinstance(self.gd, HypertoroidalGridDistribution) and self.bound_dim == 1:
                lin_dist_interpolation_method = 'mixture'
            else:
                lin_dist_interpolation_method = 'nearest_neighbor'

        xs_bound = xs[:self.bound_dim, :]
        xs_lin = xs[self.bound_dim:, :]
        # TODO

    def normalize(self):
        dist = copy.deepcopy(self)
        dist.gd = self.gd.normalize_in_place()
        return dist
    
    def normale_in_place(self):
        self.gd.normalize_in_place()

    def multiply(self, _other):
        raise NotImplementedError("Not supported for arbitrary densities. Use StateSpaceSubdivisionGaussianDistribution.")


