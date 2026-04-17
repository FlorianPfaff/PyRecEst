import warnings

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    column_stack,
    pi,
    reshape,
    signal,
    sqrt,
    squeeze,
    stack,
    tile,
    zeros,
)
from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)

from .abstract_filter import AbstractFilter
from .manifold_mixins import HypertoroidalFilterMixin


class HypertoroidalFourierFilter(AbstractFilter, HypertoroidalFilterMixin):
    """
    Filter based on Fourier series on the hypertorus.

    References:
    - Florian Pfaff, Gerhard Kurz, Uwe D. Hanebeck,
      Multivariate Angular Filtering Using Fourier Series
      Journal of Advances in Information Fusion, December 2016.
    """

    def __init__(self, n_coefficients, transformation="sqrt"):
        """
        Constructor.

        Parameters
        ----------
        n_coefficients : int or tuple of int
            Number of Fourier coefficients per dimension. The length of the
            tuple determines the dimensionality of the distribution.
        transformation : str, optional
            Transformation to use ('sqrt' or 'identity'). Default is 'sqrt'.
        """
        if pyrecest.backend.__backend_name__ in (
            "jax",
            "pytorch",
        ):  # pylint: disable=no-member
            raise NotImplementedError(
                "HypertoroidalFourierFilter is not supported on the "
                f"{pyrecest.backend.__backend_name__} backend."
            )
        if isinstance(n_coefficients, int):
            n_coefficients = (n_coefficients,)
        n_coefficients = tuple(int(n) for n in n_coefficients)
        dim = len(n_coefficients)

        # Build a uniform HFD directly (only the DC component is non-zero)
        coeff_shape = n_coefficients
        C = zeros(coeff_shape, dtype=complex)
        center = tuple(int((n - 1) // 2) for n in n_coefficients)
        if transformation == "sqrt":
            C[center] = 1.0 / sqrt((2.0 * pi) ** dim)
        elif transformation == "identity":
            C[center] = 1.0 / (2.0 * pi) ** dim
        else:
            raise ValueError(f"Unsupported transformation: '{transformation}'")
        hfd = HypertoroidalFourierDistribution(C, transformation)
        HypertoroidalFilterMixin.__init__(self)
        AbstractFilter.__init__(self, hfd)

    @property
    def filter_state(self):
        """Return the current filter state."""
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        """
        Set the filter state.

        If new_state is not a HypertoroidalFourierDistribution it is
        automatically converted using the current filter's coefficient shape
        and transformation.
        """
        if not isinstance(new_state, HypertoroidalFourierDistribution):
            warnings.warn(
                "setState:nonFourier: new_state is not a "
                "HypertoroidalFourierDistribution. Transforming with the same "
                "number of coefficients as the filter.",
                RuntimeWarning,
            )
            new_state = HypertoroidalFourierDistribution.from_distribution(
                new_state,
                self._filter_state.coeff_mat.shape,
                self._filter_state.transformation,
            )
        else:
            if new_state.transformation != self._filter_state.transformation:
                warnings.warn(
                    "setState:transDiffer: New density is transformed differently.",
                    RuntimeWarning,
                )
            if new_state.dim != self._filter_state.dim:
                warnings.warn(
                    "setState:noOfDimsDiffer: New density has different "
                    "dimensionality.",
                    RuntimeWarning,
                )
            elif new_state.coeff_mat.shape != self._filter_state.coeff_mat.shape:
                warnings.warn(
                    "setState:noOfCoeffsDiffer: New density has different number "
                    "of coefficients.",
                    RuntimeWarning,
                )
        self._filter_state = new_state

    def predict_identity(self, d_sys):
        """
        Predicts assuming identity system model, i.e.,
        x(k+1) = x(k) + w(k)  mod 2*pi,
        where w(k) is additive noise given by d_sys.

        Parameters
        ----------
        d_sys : AbstractHypertoroidalDistribution
            Distribution of additive noise.
        """
        if not isinstance(d_sys, HypertoroidalFourierDistribution):
            warnings.warn(
                "predict_identity:automaticConversion: d_sys is not a "
                "HypertoroidalFourierDistribution. Transforming with the same "
                "number of coefficients as the filter. For non-varying noises, "
                "transforming once is much more efficient.",
                RuntimeWarning,
            )
            d_sys = HypertoroidalFourierDistribution.from_distribution(
                d_sys,
                self._filter_state.coeff_mat.shape,
                self._filter_state.transformation,
            )
        self._filter_state = self._filter_state.convolve(
            d_sys, self._filter_state.coeff_mat.shape
        )

    def update_identity(self, d_meas, z):
        """
        Updates assuming identity measurement model, i.e.,
        z(k) = x(k) + v(k)  mod 2*pi,
        where v(k) is additive noise given by d_meas.

        Parameters
        ----------
        d_meas : AbstractHypertoroidalDistribution
            Distribution of additive noise.
        z : array_like, shape (dim,)
            Measurement in [0, 2*pi)^dim.
        """
        assert isinstance(d_meas, AbstractHypertoroidalDistribution)
        if not isinstance(d_meas, HypertoroidalFourierDistribution):
            warnings.warn(
                "update_identity:automaticConversion: d_meas is not a "
                "HypertoroidalFourierDistribution. Transforming with the same "
                "number of coefficients as the filter. For non-varying noises, "
                "transforming once is much more efficient.",
                RuntimeWarning,
            )
            d_meas = HypertoroidalFourierDistribution.from_distribution(
                d_meas,
                self._filter_state.coeff_mat.shape,
                self._filter_state.transformation,
            )
        z = array(z)
        assert z.shape == (
            self._filter_state.dim,
        ), f"z must have shape ({self._filter_state.dim},), got {z.shape}"
        d_meas_shifted = d_meas.shift(z)
        self._filter_state = self._filter_state.multiply(
            d_meas_shifted, self._filter_state.coeff_mat.shape
        )

    def get_f_trans_as_hfd(self, f, noise_distribution):
        """
        Build a HypertoroidalFourierDistribution representing the transition
        density f(x_{k+1} | x_k) from a deterministic system function f and
        an additive noise distribution.

        Parameters
        ----------
        f : callable
            Deterministic system function.  Takes ``dim`` arrays (x_k values)
            and returns either a tuple of ``dim`` arrays or a single array
            (for dim == 1).  Must support vectorized evaluation on N-D grids.
        noise_distribution : AbstractHypertoroidalDistribution
            Additive noise distribution.

        Returns
        -------
        HypertoroidalFourierDistribution
            2*dim-dimensional transition density.
        """
        assert isinstance(noise_distribution, AbstractHypertoroidalDistribution)
        assert callable(f)
        dim = self._filter_state.dim
        n_coefficients_2d = self._filter_state.coeff_mat.shape * 2

        def _f_trans(*args):
            # args[0:dim]     -> x_{k+1} grid values (shape: grid_shape)
            # args[dim:2*dim] -> x_k grid values      (shape: grid_shape)
            grid_shape = args[0].shape

            # Propagate x_k through the deterministic system function
            f_out = f(*args[dim:])
            if not isinstance(f_out, (tuple, list)):
                f_out = (f_out,)

            # Compute noise components: w_i = x_{k+1,i} - f_i(x_k)
            if dim == 1:
                ws_flat = args[0].ravel() - f_out[0].ravel()  # (n_pts,)
            else:
                ws_flat = column_stack(
                    [args[i].ravel() - f_out[i].ravel() for i in range(dim)]
                )  # (n_pts, dim)

            pdf_vals = noise_distribution.pdf(ws_flat)
            # Normalize: transition density integrates to 1 over x_{k+1},
            # but as a joint distribution over (x_{k+1}, x_k) it integrates
            # to (2*pi)^dim.  Divide here to compensate.
            return reshape(pdf_vals, grid_shape) / (2.0 * pi) ** dim

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Normalization:notNormalized")
            hfd_trans = HypertoroidalFourierDistribution.from_function(
                _f_trans,
                n_coefficients_2d,
                self._filter_state.transformation,
            )
        return hfd_trans

    def predict_nonlinear(self, f, noise_distribution, truncate_joint_sqrt=True):
        """
        Predicts assuming a nonlinear system model, i.e.,
        x(k+1) = f(x(k)) + w(k)  mod 2*pi.

        Parameters
        ----------
        f : callable
            System function.  See ``get_f_trans_as_hfd`` for calling convention.
        noise_distribution : AbstractHypertoroidalDistribution
            Additive process noise distribution.
        truncate_joint_sqrt : bool, optional
            Whether to truncate the intermediate joint sqrt representation
            (only relevant for sqrt transformation).  Default is True.
        """
        hfd_trans = self.get_f_trans_as_hfd(f, noise_distribution)
        self.predict_nonlinear_via_transition_density(hfd_trans, truncate_joint_sqrt)

    def predict_nonlinear_via_transition_density(  # pylint: disable=too-many-locals
        self, f_trans, truncate_joint_sqrt=True
    ):
        """
        Predicts using a probabilistic transition density.

        Parameters
        ----------
        f_trans : HypertoroidalFourierDistribution or callable
            Transition density f(x_{k+1} | x_k).

            * If a ``HypertoroidalFourierDistribution``: the first ``dim``
              dimensions must be for x_{k+1} and the remaining ``dim``
              dimensions for x_k.  Its transformation must match that of
              the current filter state.
            * If a callable: a function of 2*dim arguments (first dim for
              x_{k+1}, last dim for x_k); it is converted internally.

        truncate_joint_sqrt : bool, optional
            Whether to truncate the intermediate joint sqrt coefficient tensor.
            Default is True.
        """
        dim = self._filter_state.dim
        n_coefficients = self._filter_state.coeff_mat.shape

        if callable(f_trans) and not isinstance(
            f_trans, HypertoroidalFourierDistribution
        ):
            n_coefficients_2d = n_coefficients * 2
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Normalization:notNormalized")
                f_trans = HypertoroidalFourierDistribution.from_function(
                    f_trans,
                    n_coefficients_2d,
                    self._filter_state.transformation,
                )
        else:
            assert isinstance(
                f_trans, HypertoroidalFourierDistribution
            ), "f_trans must be a HypertoroidalFourierDistribution or a callable."
            assert (
                f_trans.transformation == self._filter_state.transformation
            ), "f_trans must use the same transformation as the filter state."
            assert f_trans.dim == 2 * dim, (
                "f_trans must be a 2*dim-dimensional HFD (first dim dims for "
                "x_{k+1}, last dim dims for x_k)."
            )

        # Reshape the prior coefficient tensor to have dim leading singleton
        # dimensions so that fftconvolve can marginalize over the x_k dims.
        hfd_reshaped = reshape(
            self._filter_state.coeff_mat,
            (1,) * dim + self._filter_state.coeff_mat.shape,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Normalization:notNormalized")
            if self._filter_state.transformation == "identity":
                # Marginalisation via valid convolution; extra (2*pi)^(2*dim) factor
                # for the two marginalisations (one already included in fTrans
                # normalisation convention, one from the actual marginalisation).
                c_predicted = (2.0 * pi) ** (2 * dim) * signal.fftconvolve(
                    f_trans.coeff_mat, hfd_reshaped, mode="valid"
                )
                # Remove trailing singleton dimensions introduced by 'valid' mode
                c_predicted = reshape(c_predicted, n_coefficients)
                new_hfd = HypertoroidalFourierDistribution(c_predicted, "identity")

            elif self._filter_state.transformation == "sqrt":
                scaled_f_trans = (2.0 * pi) ** (dim / 2.0) * f_trans.coeff_mat
                conv_mode = "same" if truncate_joint_sqrt else "full"
                c_joint_sqrt = signal.fftconvolve(
                    scaled_f_trans, hfd_reshaped, mode=conv_mode
                )

                # Pad c_joint_sqrt for the marginalization convolution.
                # Along the first dim dimensions (x_{k+1}): pad to 3*n_i-2.
                # Along the last dim dimensions (x_k): keep as-is.
                joint_size = c_joint_sqrt.shape  # 2*dim-dimensional
                pad_shape = tuple(int(3 * s - 2) for s in joint_size[:dim]) + tuple(
                    int(s) for s in joint_size[dim:]
                )
                c_joint_padded = zeros(pad_shape, dtype=c_joint_sqrt.dtype)

                # Place c_joint_sqrt in the centre of the padded tensor
                # (along the first dim dimensions).
                idx = tuple(
                    slice(int(s) - 1, 2 * int(s) - 1) for s in joint_size[:dim]
                ) + tuple(slice(None) for _ in range(dim))
                c_joint_padded[idx] = c_joint_sqrt

                c_predicted = (2.0 * pi) ** dim * signal.fftconvolve(
                    c_joint_padded, c_joint_sqrt, mode="valid"
                )
                # Result has trailing singleton dims along the x_k dimensions
                c_predicted = squeeze(c_predicted)
                new_hfd = HypertoroidalFourierDistribution(c_predicted, "identity")

            else:
                raise ValueError(
                    f"Unsupported transformation: "
                    f"'{self._filter_state.transformation}'"
                )

        if f_trans.transformation == "sqrt":
            self._filter_state = new_hfd.transform_via_fft("sqrt", n_coefficients)
        else:
            self._filter_state = new_hfd.truncate(n_coefficients)

    def update_nonlinear(self, likelihood, z=None):
        """
        Updates using an arbitrary likelihood function and measurement.

        Parameters
        ----------
        likelihood : HypertoroidalFourierDistribution or callable
            * If a ``HypertoroidalFourierDistribution``: used directly for the
              Bayes update (multiplication).  ``z`` must be ``None``.
            * If a callable ``likelihood(z, x)``: the likelihood function
              f(z | x), where both ``z`` and ``x`` are ``dim x n_pts``
              arrays.  ``z`` must be provided.
        z : array_like, shape (dim,), optional
            Measurement.  Must be provided when ``likelihood`` is a callable.
        """
        n_coefficients = self._filter_state.coeff_mat.shape

        if z is None:
            assert isinstance(likelihood, HypertoroidalFourierDistribution), (
                "When z is not given, likelihood must be a "
                "HypertoroidalFourierDistribution."
            )
        else:
            z = array(z)
            z_col = reshape(z, (-1, 1))  # (dim, 1)

            def _likelihood_fn(*grid_args):
                n_pts = grid_args[0].size
                grid_shape = grid_args[0].shape
                # State grid: (dim, n_pts)
                x_flat = stack([g.ravel() for g in grid_args], axis=0)  # (dim, n_pts)
                z_rep = tile(z_col, (1, n_pts))  # (dim, n_pts)
                vals = likelihood(z_rep, x_flat)  # (n_pts,)
                return reshape(vals, grid_shape)

            likelihood = HypertoroidalFourierDistribution.from_function(
                _likelihood_fn,
                n_coefficients,
                self._filter_state.transformation,
            )

        self._filter_state = self._filter_state.multiply(likelihood, n_coefficients)
