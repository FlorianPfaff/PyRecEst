from .state_space_subdivision_distribution import StateSpaceSubdivisionDistribution
from ..nonperiodic.gaussian_mixture import GaussianMixture
from scipy.stats import multivariate_normal
import numpy as np
import warnings
import copy

class StateSpaceSubdivisionGaussianDistribution(StateSpaceSubdivisionDistribution):
    def __init__(self, gd_, gaussians):
        super().__init__(gd_, gaussians)

    def marginalize_periodic(self):
        d = GaussianMixture(list(self.linear_distributions), self.gd.grid_values / sum(self.gd.grid_values))
        return d

    def multiply(self, other):
        assert np.array_equal(self.gd.get_grid(), other.gd.get_grid()), 'Can only multiply for equal grids.'
        dist_new = copy.deepcopy(self)
        factors_linear = multivariate_normal.pdf(self.linear_distributions.mu.T, other.linear_distributions.mu.T, np.concatenate(self.linear_distributions.C, other.linear_distributions.C, axis=2))
        dist_new.gd.grid_values = self.gd.grid_values * other.gd.grid_values * factors_linear
        dist_new.gd = dist_new.gd.normalize(warn_unnorm=False)
        for i in range(len(self.linear_distributions)):
            dist_new.linear_distributions[i] = self.linear_distributions[i].multiply(other.linear_distributions[i])
        return dist_new

    def linear_mean(self):
        mu_lin = GaussianMixture.mixture_parameters_to_gaussian_parameters(
            self.linear_distributions.mu, np.concatenate(self.linear_distributions.C, axis=2), self.gd.grid_values.T / sum(self.gd.grid_values))
        return mu_lin

    def linear_covariance(self):
        _, C = GaussianMixture.mixture_parameters_to_gaussian_parameters(
            self.linear_distributions.mu, np.concatenate(self.linear_distributions.C, axis=2), self.gd.grid_values.T / sum(self.gd.grid_values))
        return C

    def convolve(self, _):
        raise NotImplementedError('Not supported')

    def mode(self):
        max_fun_vals_cond = np.array([multivariate_normal.pdf(np.zeros_like(dist.C[0]), np.zeros_like(dist.C[0]), dist.C) for dist in self.linear_distributions])
        fun_vals_joint = max_fun_vals_cond * self.gd.grid_values
        max_val, index = np.max(fun_vals_joint), np.argmax(fun_vals_joint)
        fun_vals_joint = np.delete(fun_vals_joint, index)
        if any((max_val - fun_vals_joint) < 1e-15) or any((max_val / fun_vals_joint) < 1.001):
            warnings.warn('Density may not be unimodal. However, this can also be caused by a high grid resolution and thus very similar function values at the grid points.')
        m_periodic = self.gd.get_grid_point(index)
        m = np.concatenate((m_periodic, self.linear_distributions[index].mu), axis=0)
        return m