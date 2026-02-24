import numpy as np
import warnings
from .abstract_grid_filter import AbstractGridFilter
from .abstract_hyperhemispherical_filter import AbstractHyperhemisphericalFilter
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution
from pyrecest.distributions.conditional.sd_half_cond_sd_half_grid_distribution import SdHalfCondSdHalfGridDistribution
from pyrecest.distributions import BinghamDistribution
from pyrecest.distributions import WatsonDistribution
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions import HypersphericalMixture
from pyrecest.distributions import HyperhemisphericalUniformDistribution
from pyrecest.distributions import AbstractHyperhemisphericalDistribution
from pyrecest.distributions import HyperhemisphericalWatsonDistribution

class HyperhemisphericalGridFilter(AbstractGridFilter, AbstractHyperhemisphericalFilter):
    def __init__(self, no_of_coefficients, dim, grid_type='eq_point_set_symm'):
        self.gd = HyperhemisphericalGridDistribution.from_distribution(
            HyperhemisphericalUniformDistribution(dim), no_of_coefficients, grid_type)

    def set_state(self, new_state):
        assert self.dim == new_state.dim
        assert isinstance(new_state, AbstractHyperhemisphericalDistribution)
        self.gd = new_state

    def predict_identity(self, d_sys):
        assert isinstance(d_sys, AbstractHyperhemisphericalDistribution)
        sd_half_cond_sd_half = HyperhemisphericalGridFilter.sys_noise_to_transition_density(
            d_sys, self.gd.grid_values.shape[0])
        self.predict_nonlinear_via_transition_density(sd_half_cond_sd_half)

    def update_identity(self, meas_noise, z=None):
        assert isinstance(meas_noise, AbstractHyperhemisphericalDistribution)
        if not z==None:
            measNoise = measNoise.setMode(z)
        curr_grid = self.gd.get_grid()
        self.gd = self.gd.multiply(HyperhemisphericalGridDistribution(curr_grid, 2 * meas_noise.pdf(curr_grid).T))

    def update_nonlinear(self, likelihood, z):
        self.gd.grid_values = self.gd.grid_values * likelihood(z, self.gd.get_grid()).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.gd = self.gd.normalize()

    def predict_nonlinear_via_transition_density(self, f_trans):
        assert np.array_equal(self.gd.get_grid(), f_trans.get_grid()), \
            "fTrans is using an incompatible grid."
        self.gd = self.gd.normalize()
        grid_values_new = self.gd.get_manifold_size() / self.gd.grid_values.shape[0] * f_trans.grid_values.dot(
            self.gd.grid_values)
        self.gd = HyperhemisphericalGridDistribution(self.gd.get_grid(), grid_values_new)

    def get_point_estimate(self):
        gd_full_sphere = self.gd.to_full_sphere()
        p = BinghamDistribution.fit(gd_full_sphere.get_grid(), gd_full_sphere.grid_values.T / np.sum(
            gd_full_sphere.grid_values)).mode()
        if p[-1] < 0:
            p = -p
        return p

    @staticmethod
    def sys_noise_to_transition_density(d_sys, no_grid_points):
        if isinstance(d_sys, AbstractDistribution):
            if isinstance(d_sys, (HyperhemisphericalWatsonDistribution, WatsonDistribution)):
                def trans(xkk, xk):
                    return np.array([2 * WatsonDistribution(xk[:, i], d_sys.kappa).pdf(xkk) for i in range(xk.shape[1])]).T

            elif (isinstance(d_sys, HypersphericalMixture) and len(d_sys.dists) == 2 and
                    np.all(d_sys.w == 0.5) and np.array_equal(d_sys.dists[0].mu, -d_sys.dists[1].mu) and
                    d_sys.dists[0].kappa == d_sys.dists[1].kappa):
                def trans(xkk, xk):
                    return np.array([(VonMisesFisherDistribution(xk[:, i], d_sys.dists[0].kappa).pdf(xkk) +
                                    VonMisesFisherDistribution(xk[:, i], d_sys.dists[0].kappa).pdf(-xkk))
                                    for i in range(xk.shape[1])]).T
            else:
                raise ValueError("Distribution not supported for predict identity. Must be zonal (rotationally symmetric around last dimension)")

            print("PredictIdentity:Inefficient - Using inefficient prediction. Consider precalculating the SdHalfCondSdHalfGridDistribution and using predictNonlinearViaTransitionDensity.")
            sd_half_cond_sd_half = SdHalfCondSdHalfGridDistribution.from_function(trans, no_grid_points, True, 'eq_point_set_symm', 2 * d_sys.dim)
            return sd_half_cond_sd_half

        else:
            raise TypeError("d_sys must be an instance of AbstractDistribution")

