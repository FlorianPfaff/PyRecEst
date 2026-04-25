import copy
import warnings

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    concatenate,
    conj,
    fft,
    is_array,
    linspace,
    pi,
    reshape,
    signal,
    sqrt,
    zeros,
)
from pyrecest.distributions import AbstractCircularDistribution, CircularFourierDistribution
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)

from .abstract_circular_filter import AbstractCircularFilter
from .hypertoroidal_fourier_filter import HypertoroidalFourierFilter


class CircularFourierFilter(AbstractCircularFilter):
    def __init__(self, no_of_coefficients, transformation="sqrt"):
        if transformation not in ("sqrt", "identity"):
            raise ValueError("transformation must be either 'sqrt' or 'identity'")

        initial_state = CircularFourierDistribution.from_distribution(
            CircularUniformDistribution(), no_of_coefficients, transformation
        )
        super().__init__(initial_state)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert isinstance(new_state, AbstractCircularDistribution)

        if not isinstance(new_state, CircularFourierDistribution):
            state_to_set = self._convert_to_circular_fourier(
                new_state, "setState:nonFourier"
            )
        else:
            state_to_set = copy.deepcopy(new_state)
            if self._filter_state.transformation != state_to_set.transformation:
                warnings.warn(
                    "Warning: New density is transformed differently.",
                    RuntimeWarning,
                )
            if self._filter_state.n != state_to_set.n:
                warnings.warn(
                    "Warning: New density has a different number of coefficients.",
                    RuntimeWarning,
                )

        self._filter_state = state_to_set

    @property
    def no_of_coefficients(self):
        return int(self.filter_state.n)

    def get_estimate(self):
        return self.filter_state

    def get_point_estimate(self):
        return self.filter_state.mean_direction()

    def predict_identity(self, d_sys):
        if isinstance(d_sys, AbstractCircularDistribution):
            d_sys = self._convert_to_circular_fourier(
                d_sys, "predictIdentity:automaticConversion"
            )
            hypertoroidal_filter = self._as_hypertoroidal_filter()
            hypertoroidal_filter.predict_identity(self._to_hfd(d_sys))
            self.filter_state = self._from_hfd(hypertoroidal_filter.filter_state)
            return

        if is_array(d_sys):
            no_coefficients = self.no_of_coefficients
            assert self.filter_state.transformation == "sqrt", (
                "Only sqrt transformation currently supported"
            )
            assert d_sys.size == no_coefficients, (
                "Assume that as many grid points are used as there are coefficients."
            )
            density_values = fft.irfft(
                self.filter_state.get_c(), n=no_coefficients
            ) ** 2
            predicted_values = (
                signal.fftconvolve(density_values, d_sys, mode="same")
                * no_coefficients
                * 2
                * pi
            )
            self.filter_state = CircularFourierDistribution.from_function_values(
                sqrt(predicted_values), self.filter_state.transformation
            )
            return

        raise ValueError("Input format of d_sys is not supported")

    def update_identity(self, d_meas, z):
        d_meas = self._convert_to_circular_fourier(
            d_meas, "updateIdentity:automaticConversion"
        )
        hypertoroidal_filter = self._as_hypertoroidal_filter()
        hypertoroidal_filter.update_identity(self._to_hfd(d_meas), array(z).reshape(-1))
        self.filter_state = self._from_hfd(hypertoroidal_filter.filter_state)

    def predict_nonlinear(
        self,
        f,
        noise_distribution,
        truncate_joint_sqrt=True,
    ):
        assert callable(f)
        hypertoroidal_filter = self._as_hypertoroidal_filter()
        hypertoroidal_filter.predict_nonlinear(
            f, noise_distribution, truncate_joint_sqrt
        )
        self.filter_state = self._from_hfd(hypertoroidal_filter.filter_state)

    def predict_nonlinear_via_transition_density(
        self, f_trans, truncate_joint_sqrt=True
    ):
        hypertoroidal_filter = self._as_hypertoroidal_filter()
        hypertoroidal_filter.predict_nonlinear_via_transition_density(
            f_trans, truncate_joint_sqrt
        )
        self.filter_state = self._from_hfd(hypertoroidal_filter.filter_state)

    def update_nonlinear(self, likelihood, z):
        x_vals = linspace(0.0, 2.0 * pi, self.no_of_coefficients, endpoint=False)
        likelihood_values = reshape(likelihood(z, x_vals), x_vals.shape)
        fd_meas = self._from_density_values(likelihood_values)
        self.update_identity(fd_meas, zeros(1))

    def update_nonlinear_via_ifft(self, likelihood, z):
        no_coefficients = self.no_of_coefficients
        prior_values = fft.irfft(self.filter_state.get_c(), n=no_coefficients)
        x_vals = linspace(0.0, 2.0 * pi, no_coefficients, endpoint=False)
        likelihood_values = likelihood(z, x_vals)

        if self.filter_state.transformation == "identity":
            posterior_values = prior_values * likelihood_values
        elif self.filter_state.transformation == "sqrt":
            posterior_values = prior_values * sqrt(likelihood_values)
        else:
            raise ValueError("Transformation currently not supported")

        self.filter_state = CircularFourierDistribution.from_function_values(
            posterior_values, self.filter_state.transformation
        )

    def updateIdentity(self, dMeas, z):
        self.update_identity(dMeas, z)

    def _convert_to_circular_fourier(
        self, distribution, warning_id
    ):
        if isinstance(distribution, CircularFourierDistribution):
            return copy.deepcopy(distribution)

        warnings.warn(
            f"{warning_id}: distribution is not a CircularFourierDistribution. "
            "Transforming with the same number of coefficients as the filter. "
            "For non-varying noises, transforming once is more efficient.",
            RuntimeWarning,
        )
        return CircularFourierDistribution.from_distribution(
            distribution,
            self.no_of_coefficients,
            self.filter_state.transformation,
        )

    def _as_hypertoroidal_filter(self):
        hypertoroidal_filter = HypertoroidalFourierFilter(
            (self.no_of_coefficients,), self.filter_state.transformation
        )
        hypertoroidal_filter.filter_state = self._to_hfd(self.filter_state)
        return hypertoroidal_filter

    def _from_density_values(self, density_values):
        if self.filter_state.transformation == "identity":
            values = density_values
        elif self.filter_state.transformation == "sqrt":
            values = sqrt(density_values)
        else:
            raise ValueError("Transformation currently not supported")

        return CircularFourierDistribution.from_function_values(
            values, self.filter_state.transformation
        )

    @staticmethod
    def _to_hfd(distribution):
        positive_coefficients = distribution.get_c()
        coeff_mat = concatenate(
            (conj(positive_coefficients[-1:0:-1]), positive_coefficients)
        )
        if distribution.multiplied_by_n:
            coeff_mat = coeff_mat / distribution.n
        return HypertoroidalFourierDistribution(coeff_mat, distribution.transformation)

    @staticmethod
    def _from_hfd(distribution):
        no_coefficients = int(distribution.coeff_mat.shape[0])
        center_index = no_coefficients // 2
        positive_coefficients = distribution.coeff_mat[center_index:] * no_coefficients
        return CircularFourierDistribution(
            c=positive_coefficients,
            n=no_coefficients,
            transformation=distribution.transformation,
            multiplied_by_n=True,
        )
