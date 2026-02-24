from beartype import beartype
from collections.abc import Callable
import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import fftconvolve
from .abstract_circular_filter import AbstractCircularFilter
from pyrecest.distributions import CircularFourierDistribution, AbstractCircularDistribution, AbstractHypertoroidalDistribution
from pyrecest.distributions.circle.circular_uniform_distribution import CircularUniformDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from pyrecest.distributions.hypertorus.toroidal_fourier_distribution import ToroidalFourierDistribution
from .hypertoroidal_fourier_filter import HypertoroidalFourierFilter
import copy
import warnings
import scipy

class CircularFourierFilter(AbstractCircularFilter, HypertoroidalFourierFilter):
    def __init__(self, no_of_coefficients, transformation='sqrt'):
        assert transformation == 'sqrt' or transformation == 'identity'
        AbstractCircularFilter.__init__(self, CircularFourierDistribution.from_distribution(
            CircularUniformDistribution(), no_of_coefficients, transformation))

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert isinstance(new_state, AbstractCircularDistribution)
        if not isinstance(new_state, CircularFourierDistribution):
            state_to_set = CircularFourierDistribution.from_distribution(
                new_state, 2 * np.size(self.filter_state.a) - 1, self.filter_state.transformation)
        else:
            state_to_set = copy.deepcopy(new_state)
            if self._filter_state.transformation != state_to_set.transformation:
                warnings.warn("Warning: New density is transformed differently.")
            if np.size(new_state.a) != np.size(self.filter_state.a):
                warnings.warn("Warning: New density has a different number of coefficients.")
        self._filter_state = state_to_set

    def get_estimate(self):
        return self.filter_state

    def get_point_estimate(self):
        return self.filter_state.mean_direction()

    def predict_identity(self, d_sys):
        if isinstance(d_sys, AbstractCircularDistribution):
            if not isinstance(d_sys, CircularFourierDistribution):
                warnings.warn("Warning: d_sys is not a FourierDistribution. Transforming with a number of coefficients that is equal to that of the filter. For non-varying noises, transforming once is much more efficient and should be preferred.")
                d_sys = CircularFourierDistribution.from_distribution(d_sys, 2 * len(self.filter_state.a) - 1, self.filter_state.transformation)
            self.filter_state = self.filter_state.convolve(d_sys)

        elif isinstance(d_sys, np.ndarray):
            assert self.filter_state.transformation == 'sqrt', "Only sqrt transformation currently supported"
            assert d_sys.size == self.n, "Assume that as many grid points are used as there are coefficients."
            fdvals = np.fft.ifftshift(self.filter_state.c) ** 2
            f_pred_vals = fftconvolve(fdvals, d_sys, mode='same') * self.n * 2 * np.pi
            self.filter_state = CircularFourierDistribution(transformation = 'sqrt', c=np.fft.fftshift(np.fft.fft(np.sqrt(f_pred_vals))) / self.n)

        else:
            raise ValueError("Input format of d_sys is not supported")

    def update_identity(self, d_meas, z):
        assert isinstance(d_meas, AbstractCircularDistribution)
        if not isinstance(d_meas, CircularFourierDistribution):
            print("Warning: d_meas is not a FourierDistribution. Transforming with a number of coefficients that is equal to that of the filter. For non-varying noises, transforming once is much more efficient and should be preferred.")
            d_meas = CircularFourierDistribution.from_distribution(d_meas, 2 * len(self.filter_state.a) - 1, self.filter_state.transformation)
        d_meas_shifted = d_meas.shift(z)
        self.filter_state = self.filter_state.multiply(d_meas_shifted, 2 * len(self.filter_state.a) - 1)

    def predict_nonlinear(self, f, noise_distribution, truncate_joint_sqrt=True):
        assert isinstance(noise_distribution, AbstractCircularDistribution)
        assert callable(f)
        f_trans = lambda xkk, xk: np.reshape(noise_distribution.pdf(xkk.T - f(xk.T)), xk.shape)
        self.predict_nonlinear_via_transition_density(f_trans, truncate_joint_sqrt)

    def updateIdentity(self, dMeas, z):
        """
        Updates assuming identity measurement model, i.e.,
        z(k) = x(k) + v(k) mod 2pi,
        where v(k) is additive noise given by dMeas.
        The modulo operation is carried out componentwise.

        Parameters:
        dMeas (AbstractHypertoroidalDistribution):
            distribution of additive noise
        z (dim x 1 vector):
            measurement in [0, 2pi)^dim
        """

        assert isinstance(dMeas, AbstractHypertoroidalDistribution)

        if not isinstance(dMeas, HypertoroidalFourierDistribution):
            print("Update:automaticConversion: dMeas is not a HypertoroidalFourierDistribution. \
            Transforming with an amount of coefficients that is equal to that of the filter. \
            For non-varying noises, transforming once is much more efficient and should be preferred.")
            sizeHfdC = np.shape(self.hfd.C)  # Needed for workaround for 1D case
            dMeas = HypertoroidalFourierDistribution.from_distribution(dMeas, sizeHfdC[sizeHfdC > 1], self.hfd.transformation)

        assert np.shape(z) == (self.hfd.dim, 1)

        dMeasShifted = dMeas.shift(z)
        self.hfd = self.hfd.multiply(dMeasShifted, np.shape(self.hfd.C))

    def predict_nonlinear_via_transition_density(self, f_trans, truncate_joint_sqrt=True):

        if callable(f_trans):
            f_trans = ToroidalFourierDistribution.from_function(f_trans, self.filter_state.n * np.array([1, 1]), dim=2, desired_transformation=self.filter_state.transformation)
        else:
            assert self.filter_state.transformation == f_trans.transformation

        if self.filter_state.transformation == 'identity':
            c_predicted_id = (2 * np.pi) ** 2 * scipy.signal.convolve2d(f_trans.C, np.atleast_2d(self.filter_state.c), mode='valid')
        elif self.filter_state.transformation == 'sqrt':
            if not truncate_joint_sqrt:
                c_joint_sqrt = scipy.signal.convolve2d(np.sqrt(2 * np.pi) * f_trans.C, self.filter_state.c, mode='full')
            else:
                c_joint_sqrt = scipy.signal.convolve2d(np.sqrt(2 * np.pi) * f_trans.C, self.filter_state.c, mode='same')

            additional_columns = 2 * len(self.filter_state.b)
            c_predicted_id = 2 * np.pi * scipy.signal.convolve2d(
                np.pad(c_joint_sqrt, ((additional_columns, additional_columns), (0, 0))),
                c_joint_sqrt,
                mode='valid'
            )

        if self.filter_state.transformation == 'identity' or not truncate_joint_sqrt:
            self.filter_state = CircularFourierDistribution(transformation='identity', c=c_predicted_id)
        else:
            self.filter_state = CircularFourierDistribution(transformation='identity', c=c_predicted_id)

        if f_trans.transformation == 'sqrt':
            self.filter_state = self.filter_state.transform_via_fft('sqrt', self.filter_state.n)
    
    def predict_identity(self, d_sys):
        if isinstance(d_sys, AbstractCircularDistribution):
            if not isinstance(d_sys, CircularFourierDistribution):
                print("Warning: d_sys is not a FourierDistribution. Transforming with a number of coefficients that is equal to that of the filter. For non-varying noises, transforming once is much more efficient and should be preferred.")
                d_sys = CircularFourierDistribution.from_distribution(d_sys, 2 * len(self.filter_state.a) - 1, self.filter_state.transformation)
            self.filter_state = self.filter_state.convolve(d_sys)

        elif isinstance(d_sys, np.ndarray):
            no_coeffs = 2 * len(self.filter_state.a) - 1
            assert self.filter_state.transformation == 'sqrt', "Only sqrt transformation currently supported"
            assert d_sys.size == no_coeffs, "Assume that as many grid points are used as there are coefficients."
            fdvals = np.fft.ifftshift(self.filter_state.c) ** 2
            f_pred_vals = fftconvolve(fdvals, d_sys, mode='same') * no_coeffs * 2 * np.pi
            self.filter_state = CircularFourierDistribution(transformation = 'sqrt', c=np.fft.fftshift(np.fft.fft(np.sqrt(f_pred_vals))) / no_coeffs)

        else:
            raise ValueError("Input format of d_sys is not supported")

    @beartype
    def update_identity(self, d_meas: AbstractCircularDistribution, z):
        if not isinstance(d_meas, CircularFourierDistribution):
            print("Warning: d_meas is not a FourierDistribution. Transforming with a number of coefficients that is equal to that of the filter. For non-varying noises, transforming once is much more efficient and should be preferred.")
            d_meas = CircularFourierDistribution.from_distribution(d_meas, 2 * len(self.filter_state.a) - 1, self.filter_state.transformation)
        d_meas_shifted = d_meas.shift(z)
        self.filter_state = self.filter_state.multiply(d_meas_shifted, 2 * len(self.filter_state.a) - 1)

    @beartype
    def predict_nonlinear(self, f: Callable, noise_distribution: AbstractCircularDistribution, truncate_joint_sqrt: bool=True):
        f_trans = lambda xkk, xk: np.reshape(noise_distribution.pdf(xkk.T - f(xk.T)), xk.shape)
        self.predict_nonlinear_via_transition_density(f_trans, truncate_joint_sqrt)


    def update_nonlinear(self, likelihood, z):
        fd_meas = CircularFourierDistribution.from_function(
            lambda x: likelihood(z, x.ravel()).reshape(x.shape),
            2 * len(self.filter_state.a) - 1,
            self.filter_state.transformation
        )
        self.update_identity(fd_meas, np.zeros_like(z))

    def update_nonlinear_via_ifft(self, likelihood, z):
        c_curr = self.filter_state.c
        prior_vals = ifft(ifftshift(c_curr), overwrite_x=True, workers=-1) * len(c_curr)
        x_vals = np.linspace(0, 2 * np.pi, len(c_curr) + 1)
        
        if self.filter_state.transformation == 'identity':
            posterior_vals = prior_vals * likelihood(z, x_vals[:-1])
        elif self.filter_state.transformation == 'sqrt':
            posterior_vals = prior_vals * np.sqrt(likelihood(z, x_vals[:-1]))
        else:
            raise ValueError('Transformation currently not supported')

        self.filter_state = CircularFourierDistribution.from_complex(fftshift(fft(posterior_vals, overwrite_x=True, workers=-1)), self.filter_state.transformation)

    def association_likelihood(self, likelihood):
        assert len(self.get_estimate.a) == len(likelihood.a)
        assert len(self.get_estimate.transformation) == len(likelihood.transformation)

        if self.get_estimate.transformation == 'identity':
            likelihood_val = 2 * np.pi * np.real(np.dot(self.get_estimate.c, likelihood.c.T))
        elif self.get_estimate.transformation == 'sqrt':
            likelihood_val = 2 * np.pi * np.linalg.norm(np.convolve(self.get_estimate.c, likelihood.c))**2
        else:
            raise ValueError('Transformation not supported')

        return likelihood_val