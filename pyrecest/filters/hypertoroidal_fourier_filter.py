import numpy as np
import warnings

from .abstract_hypertoroidal_filter import AbstractHypertoroidalFilter
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from pyrecest.distributions import HypertoroidalUniformDistribution

class HypertoroidalFourierFilter(AbstractHypertoroidalFilter):
    def __init__(self, noOfCoefficients, transformation='sqrt'):
        self.hfd = HypertoroidalFourierDistribution.from_distribution(
            HypertoroidalUniformDistribution(np.size(noOfCoefficients)), noOfCoefficients, transformation)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if np.ndim(self.hfd.C) != np.ndim(new_state.C):
            warnings.warn('The new state has a different dimensionality.')
        elif self.hfd.C.shape != new_state.C.shape:
            warnings.warn('The new state has a different number of coefficients.')
        self.hfd = new_state

    def predict_identity(self, d_sys):
        size_hfd_c = self.hfd.C.shape
        if not isinstance(d_sys, HypertoroidalFourierDistribution):
            warnings.warn("PredictIdentity:automaticConversion: dSys is not a HypertoroidalFourierDistribution. "
                          "Transforming with a number of coefficients that is equal to that of the filter. "
                          "For non-varying noises, transforming once is much more efficient and should be preferred.")
            d_sys = HypertoroidalFourierDistribution.from_distribution(
                d_sys, size_hfd_c[size_hfd_c > 1], self.hfd.transformation)
        self.hfd = self.hfd.convolve(d_sys, size_hfd_c[size_hfd_c > 1])

    def predictNonlinearViaTransitionDensity(self, fTrans, truncateJointSqrt=True):
        dimC = np.shape(self.hfd.C)
        warnStruct = warnings.catch_warnings()
        warnings.simplefilter('ignore')
        # rest of the method body...
        if self.hfd.transformation == 'identity' or not truncateJointSqrt:
            warnings.resetwarnings()
            self.hfd = HypertoroidalFourierDistribution(CPredictedId,'identity')
        else:
            self.hfd = HypertoroidalFourierDistribution(CPredictedId,'identity')
            warnings.resetwarnings()

        if fTrans.transformation == 'sqrt':
            self.hfd = self.hfd.transformViaFFT('sqrt',dimC[dimC>1])

    def updateNonlinear(self, likelihood, z: np.ndarray | None = None):
        """
        Performs an update for an arbitrary likelihood function and a measurement. If the measurement z is not
        given, assume that likelihood (for varying x) is given as a hfd. Otherwise, transform it.

        Parameters:
        likelihood f(z|x):
            Either given as HypertoroidalFourierDistribution or as a function. If given as a function, we assume
            that it takes matrices (same convention as .pdf) as input for both measurement and state.
        measurement z:
            Used as input for likelihood. Is repmatted if likelihood is to be evaluated at multiple points.
        """

        # Check if z is given
        if z is None:  
            assert isinstance(likelihood, HypertoroidalFourierDistribution)
        else:
            # If z is given, assume likelihood is a function
            def func(*args):
                reshaped_likelihood = likelihood(np.repeat(z, len(args[0]), axis=1), np.concatenate([i.flatten() for i in args], axis=0))
                return reshaped_likelihood.reshape(args[0].shape)

            likelihood = HypertoroidalFourierDistribution.from_function(func, self.hfd.C.shape, self.hfd.transformation)
        
        self.hfd = self.hfd.multiply(likelihood, self.hfd.C.shape)
