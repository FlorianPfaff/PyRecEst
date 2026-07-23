"""Positive-kernel identity Fourier filter."""

from __future__ import annotations

import warnings

import pyrecest.backend
from pyrecest.backend import array, column_stack, pi, reshape, stack, tile, zeros
from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.fejer import (
    _validate_integer_control,
    normalize_coefficient_shape,
    normalize_kernel_name,
)
from pyrecest.distributions.hypertorus.fejer_hypertoroidal_fourier_distribution import (
    FejerHypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.filters.abstract_filter import AbstractFilter
from pyrecest.filters.manifold_mixins import HypertoroidalFilterMixin
from scipy import signal


class FejerIdentityFilter(AbstractFilter, HypertoroidalFilterMixin):
    """Hypertoroidal identity Fourier filter with adaptive positive-kernel reduction.

    The filter follows the PyRecEst ``HypertoroidalFourierFilter`` interface for
    the identity transformation. The prior/posterior state is a
    :class:`FejerHypertoroidalFourierDistribution`.

    By default, the filter acts as an IFF with a nonnegativity safeguard: update
    reductions are first tried as sharp identity reductions and the
    Fejer-Korovkin kernel is only activated when the diagnostic FFT grid shows
    negative values. Plain unconditional Fejer reduction remains available via
    ``reduction_kernel="fejer", adaptive_reduction=False`` for baseline tests.
    """

    def __init__(
        self,
        n_coefficients: int | tuple[int, ...],
        *,
        reduction_kernel: str = "korovkin",
        adaptive_reduction: bool = True,
        min_value_tolerance: float = 1e-12,
        oversampling_factor: int = 1,
        exponent_search_steps: int = 24,
    ):
        if pyrecest.backend.__backend_name__ in (
            "jax",
            "pytorch",
        ):  # pylint: disable=no-member
            raise NotImplementedError(
                f"FejerIdentityFilter is not supported on the {pyrecest.backend.__backend_name__} backend."
            )

        if isinstance(n_coefficients, int):
            n_coefficients = (n_coefficients,)
        n_coefficients = normalize_coefficient_shape(n_coefficients)
        dim = len(n_coefficients)

        self.reduction_kernel = normalize_kernel_name(reduction_kernel)
        self.adaptive_reduction = bool(adaptive_reduction)
        self.min_value_tolerance = float(min_value_tolerance)
        self.oversampling_factor = _validate_integer_control(
            oversampling_factor, "oversampling_factor", minimum=1
        )
        self.exponent_search_steps = _validate_integer_control(
            exponent_search_steps, "exponent_search_steps", minimum=0
        )

        coeff_mat = zeros(n_coefficients, dtype=complex)
        center = tuple((n - 1) // 2 for n in n_coefficients)
        coeff_mat[center] = 1.0 / (2.0 * pi) ** dim

        HypertoroidalFilterMixin.__init__(self)
        AbstractFilter.__init__(
            self,
            FejerHypertoroidalFourierDistribution(coeff_mat, **self.reduction_options),
        )

    @property
    def reduction_options(self):
        return {
            "reduction_kernel": self.reduction_kernel,
            "adaptive_reduction": self.adaptive_reduction,
            "min_value_tolerance": self.min_value_tolerance,
            "oversampling_factor": self.oversampling_factor,
            "exponent_search_steps": self.exponent_search_steps,
        }

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(new_state, FejerHypertoroidalFourierDistribution):
            if new_state.coeff_mat.shape != self._filter_state.coeff_mat.shape:
                warnings.warn(
                    "setState:noOfCoeffsDiffer: New density has a different number of coefficients; applying configured positive-kernel reduction to the filter shape.",
                    RuntimeWarning,
                )
                new_state = (
                    FejerHypertoroidalFourierDistribution.from_fourier_distribution(
                        new_state,
                        self._filter_state.coeff_mat.shape,
                        apply_fejer=True,
                        **self.reduction_options,
                    )
                )
            elif new_state.reduction_options != self.reduction_options:
                new_state = (
                    FejerHypertoroidalFourierDistribution.from_fourier_distribution(
                        new_state,
                        self._filter_state.coeff_mat.shape,
                        apply_fejer=False,
                        **self.reduction_options,
                    )
                )
            self._filter_state = new_state
            return

        if isinstance(new_state, HypertoroidalFourierDistribution):
            if new_state.transformation != "identity":
                raise ValueError(
                    "FejerIdentityFilter only accepts identity Fourier distributions."
                )
            warnings.warn(
                "setState:nonFejerFourier: converting identity Fourier distribution to positive-kernel identity form.",
                RuntimeWarning,
            )
            self._filter_state = (
                FejerHypertoroidalFourierDistribution.from_fourier_distribution(
                    new_state,
                    self._filter_state.coeff_mat.shape,
                    apply_fejer=True,
                    **self.reduction_options,
                )
            )
            return

        if isinstance(new_state, AbstractHypertoroidalDistribution):
            warnings.warn(
                "setState:nonFourier: new_state is not a Fourier distribution. Transforming with the same number of coefficients as the filter.",
                RuntimeWarning,
            )
            self._filter_state = (
                FejerHypertoroidalFourierDistribution.from_distribution(
                    new_state,
                    self._filter_state.coeff_mat.shape,
                    apply_fejer=True,
                    **self.reduction_options,
                )
            )
            return

        raise TypeError("new_state must be an AbstractHypertoroidalDistribution.")

    def predict_identity(self, d_sys):
        """Predict with ``x(k+1) = x(k) + w(k) mod 2*pi``."""

        d_sys = self._as_fejer_identity_distribution(
            d_sys,
            "predict_identity:automaticConversion: d_sys is not a positive-kernel identity Fourier distribution. Transforming with the same number of coefficients as the filter.",
        )
        self._filter_state = self._filter_state.convolve(
            d_sys, self._filter_state.coeff_mat.shape
        )

    def get_f_trans_as_hfd(self, f, noise_distribution):
        """Build a positive-kernel identity transition density for nonlinear prediction."""

        if not isinstance(noise_distribution, AbstractHypertoroidalDistribution):
            raise TypeError(
                "noise_distribution must be an AbstractHypertoroidalDistribution."
            )
        if not callable(f):
            raise TypeError("f must be callable.")

        dim = self._filter_state.dim
        n_coefficients_2d = self._filter_state.coeff_mat.shape * 2

        def _f_trans(*args):
            grid_shape = args[0].shape
            f_out = f(*args[dim:])
            if not isinstance(f_out, (tuple, list)):
                f_out = (f_out,)

            if dim == 1:
                ws_flat = args[0].ravel() - f_out[0].ravel()
            else:
                ws_flat = column_stack(
                    [args[i].ravel() - f_out[i].ravel() for i in range(dim)]
                )

            pdf_vals = noise_distribution.pdf(ws_flat)
            return reshape(pdf_vals, grid_shape) / (2.0 * pi) ** dim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Normalization:notNormalized")
            return FejerHypertoroidalFourierDistribution.from_function(
                _f_trans, n_coefficients_2d, apply_fejer=True, **self.reduction_options
            )

    def predict_nonlinear(self, f, noise_distribution):
        """Predict with ``x(k+1) = f(x(k)) + w(k) mod 2*pi``."""

        self.predict_nonlinear_via_transition_density(
            self.get_f_trans_as_hfd(f, noise_distribution)
        )

    def predict_nonlinear_via_transition_density(self, f_trans):
        """Predict using a 2*dim-dimensional transition density."""

        dim = self._filter_state.dim
        n_coefficients = self._filter_state.coeff_mat.shape

        if callable(f_trans) and not isinstance(
            f_trans, HypertoroidalFourierDistribution
        ):
            f_trans = FejerHypertoroidalFourierDistribution.from_function(
                f_trans,
                n_coefficients * 2,
                apply_fejer=True,
                **self.reduction_options,
            )
        else:
            if not isinstance(f_trans, HypertoroidalFourierDistribution):
                raise TypeError(
                    "f_trans must be a HypertoroidalFourierDistribution or a callable."
                )
            if f_trans.transformation != "identity":
                raise ValueError("f_trans must use the identity transformation.")
            if f_trans.dim != 2 * dim:
                raise ValueError(
                    "f_trans must be a 2*dim-dimensional HFD (first dim dims for x_{k+1}, last dim dims for x_k)."
                )
            if not isinstance(f_trans, FejerHypertoroidalFourierDistribution):
                f_trans = (
                    FejerHypertoroidalFourierDistribution.from_fourier_distribution(
                        f_trans,
                        f_trans.coeff_mat.shape,
                        apply_fejer=True,
                        **self.reduction_options,
                    )
                )

        hfd_reshaped = reshape(
            self._filter_state.coeff_mat,
            (1,) * dim + self._filter_state.coeff_mat.shape,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Normalization:notNormalized")
            c_predicted = (2.0 * pi) ** (2 * dim) * signal.fftconvolve(
                f_trans.coeff_mat, hfd_reshaped, mode="valid"
            )
            c_predicted = reshape(c_predicted, n_coefficients)
            self._filter_state = FejerHypertoroidalFourierDistribution(
                c_predicted, **self.reduction_options
            )

    def update_identity(self, d_meas, z):
        """Update with ``z(k) = x(k) + v(k) mod 2*pi``."""

        d_meas = self._as_fejer_identity_distribution(
            d_meas,
            "update_identity:automaticConversion: d_meas is not a positive-kernel identity Fourier distribution. Transforming with the same number of coefficients as the filter.",
        )
        z = array(z)
        if z.shape != (self._filter_state.dim,):
            raise ValueError(
                f"z must have shape ({self._filter_state.dim},), got {z.shape}"
            )
        d_meas_shifted = d_meas.shift(z)
        self._filter_state = self._filter_state.multiply(
            d_meas_shifted, self._filter_state.coeff_mat.shape
        )

    def update_nonlinear(self, likelihood, z=None):
        """Update with a Fourier likelihood or a callable likelihood."""

        n_coefficients = self._filter_state.coeff_mat.shape
        if z is None:
            if not isinstance(likelihood, HypertoroidalFourierDistribution):
                raise TypeError(
                    "When z is not given, likelihood must be a HypertoroidalFourierDistribution."
                )
            likelihood = self._as_fejer_identity_distribution(
                likelihood,
                "update_nonlinear:nonFejerFourier: converting likelihood to positive-kernel identity form.",
            )
        else:
            if not callable(likelihood):
                raise TypeError("likelihood must be callable when z is provided.")
            z = array(z)
            if z.shape != (self._filter_state.dim,):
                raise ValueError(
                    f"z must have shape ({self._filter_state.dim},), got {z.shape}"
                )
            z_col = reshape(z, (-1, 1))

            def _likelihood_fn(*grid_args):
                n_pts = grid_args[0].size
                grid_shape = grid_args[0].shape
                x_flat = stack([g.ravel() for g in grid_args], axis=0)
                z_rep = tile(z_col, (1, n_pts))
                vals = likelihood(z_rep, x_flat)
                return reshape(vals, grid_shape)

            likelihood = FejerHypertoroidalFourierDistribution.from_function(
                _likelihood_fn,
                n_coefficients,
                apply_fejer=True,
                **self.reduction_options,
            )

        self._filter_state = self._filter_state.multiply(likelihood, n_coefficients)

    def _as_fejer_identity_distribution(
        self, distribution, warning_message: str
    ) -> FejerHypertoroidalFourierDistribution:
        if (
            isinstance(distribution, FejerHypertoroidalFourierDistribution)
            and distribution.coeff_mat.shape == self._filter_state.coeff_mat.shape
        ):
            return distribution

        if isinstance(distribution, HypertoroidalFourierDistribution):
            if distribution.transformation != "identity":
                raise ValueError("Only identity Fourier distributions are supported.")
            warnings.warn(warning_message, RuntimeWarning)
            return FejerHypertoroidalFourierDistribution.from_fourier_distribution(
                distribution,
                self._filter_state.coeff_mat.shape,
                apply_fejer=True,
                **self.reduction_options,
            )

        if not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise TypeError(
                "distribution must be an AbstractHypertoroidalDistribution."
            )
        warnings.warn(warning_message, RuntimeWarning)
        return FejerHypertoroidalFourierDistribution.from_distribution(
            distribution,
            self._filter_state.coeff_mat.shape,
            apply_fejer=True,
            **self.reduction_options,
        )
