import copy
import math
import warnings

import pyrecest.backend
from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    all,
    any,
    arange,
    array,
    atleast_1d,
    column_stack,
    conj,
    exp,
    fft,
    imag,
    linalg,
    linspace,
    log,
    meshgrid,
    mod,
    ndim,
    ones,
    pad,
    pi,
    prod,
    real,
    reshape,
    shape,
    signal,
    sqrt,
    sum,
    zeros,
)

from ..abstract_orthogonal_basis_distribution import AbstractOrthogonalBasisDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from .hypertoroidal_uniform_distribution import HypertoroidalUniformDistribution


class HypertoroidalFourierDistribution(
    AbstractOrthogonalBasisDistribution, AbstractHypertoroidalDistribution
):
    """
    Hypertoroidal distribution represented by a Fourier series.

    coeff_mat (self.coeff_mat) contains the complex Fourier coefficients
    on an index grid k ∈ {-k_max_d, ..., 0, ..., +k_max_d}^dim.

    The 'transformation' argument controls how the coefficients relate
    to the actual pdf:

        - 'identity':  pdf(x) = sum_k C_k exp(i k⋅x)
        - 'sqrt':      pdf(x) = | sum_k C_k exp(i k⋅x) |^2
        - 'log':       pdf(x) ∝ exp(sum_k C_k exp(i k⋅x))  (no auto-normalization)
    """

    def __init__(self, coeff_mat, transformation: str = "sqrt"):
        # If user accidentally passes a distribution as coeffs
        if isinstance(coeff_mat, AbstractHypertoroidalDistribution):
            raise ValueError(
                "fourierCoefficients:invalidCoefficientMatrix: "
                "You gave a distribution as the first argument. "
                "To convert distributions to a Fourier-representation, "
                "use HypertoroidalFourierDistribution.from_distribution(...)."
            )

        # Single coefficient -> assume dimension 1
        if ndim(coeff_mat) == 0:
            warnings.warn(
                "fourierCoefficients:singleCoefficient: "
                "Fourier series only has one element, assuming dimension 1.",
                RuntimeWarning,
            )
            coeff_mat = atleast_1d(coeff_mat)

        dim = ndim(coeff_mat)

        # Each axis length must be odd: size = 2*k_max + 1
        if not all((array(coeff_mat.shape) - 1) % 2 == 0):
            raise ValueError(
                "Each dimension of coeff_mat must have odd length (2*k+1) "
                "so that the frequency index ranges from -k…k."
            )

        AbstractHypertoroidalDistribution.__init__(self, dim)
        # Let the orthogonal-basis base class store coeff_mat + transformation
        AbstractOrthogonalBasisDistribution.__init__(
            self, coeff_mat, transformation=transformation
        )

        self.normalize_in_place()

    def value(self, xs):
        """
        Evaluate the underlying (pre-transformation) Fourier series at xs.

        Parameters
        ----------
        xs : array_like, shape (n_eval, dim) or (dim,)
            Points on the hypertorus in radians.

        Returns
        -------
        val : array, shape (n_eval,)
            Complex-valued series evaluation (real up to numerical error).
        """
        xs = array(xs)

        if self.dim > 1 and ndim(xs) == 1:
            # Must be single point in multi-dim case
            assert xs.shape[0] == self.dim

        if self.dim not in (1, xs.shape[-1]):
            raise ValueError(
                f"Expected xs with shape (n_eval, {self.dim}), got {xs.shape}"
            )

        # Sanity: all coefficient axes must be odd length
        if not all((array(self.coeff_mat.shape) - 1) % 2 == 0):
            raise ValueError(
                "Supporting even numbers of coefficients would make the "
                "representation ambiguous. Every axis length must be odd."
            )

        # maxk_d for each dimension
        maxk = ((array(self.coeff_mat.shape)) - 1) // 2
        k_ranges = [arange(int(-m), int(m) + 1) for m in maxk]

        # Build integer index grid k over all dimensions
        index_mats = meshgrid(*k_ranges, indexing="ij")  # each shape (n1,...,nd)
        # Flatten k to shape (dim, K)
        k_flat = array([mat.reshape(-1) for mat in index_mats])  # (dim, K)

        xs_aug = xs[..., None]  # (n_eval, dim, 1)
        # Broadcast-multiply: (dim, K) * (n_eval, dim, 1) -> (n_eval, dim, K)
        mat_curr = k_flat * xs_aug

        # Sum over dimensions to get k⋅x for each eval point and Fourier mode
        if self.dim > 1:
            mat_curr = sum(mat_curr, axis=-2)  # (n_eval, K)
        # For dim == 1, mat_curr already has shape (n_eval, 1, K)

        # Flatten coefficients to (K,)
        coeff_flat = reshape(self.coeff_mat, (-1,))

        # Evaluate series: Σ_k C_k exp(i k⋅x)
        mat_curr = exp(1j * mat_curr) * coeff_flat  # broadcast (n_eval, K)
        p = sum(mat_curr, axis=-1)  # (n_eval,)

        if not all(imag(p) < 0.1):
            warnings.warn(
                "value: imaginary part is not negligible; returning real part.",
                RuntimeWarning,
            )

        return real(p)

    def normalize_in_place(self, tol: float = 1e-4, warn_unnorm: bool = True):
        """
        Normalize the coefficients so that the implied pdf integrates to 1.
        """
        if self.transformation == "sqrt":
            # For sqrt-transform, ∫ pdf = (2π)^dim * ||C||^2
            c00 = linalg.norm(reshape(self.coeff_mat, (-1,))) ** 2
            assert complex(c00).imag <= 0.001, "Center coefficient must be real-valued."
            c00 = real(c00)
            factor_for_id = c00 * (2 * pi) ** self.dim
            normalization_factor = sqrt(real(factor_for_id))
        elif self.transformation == "identity":
            # Center coefficient corresponds to k = 0
            center_indices = tuple(s // 2 for s in self.coeff_mat.shape)
            c00 = self.coeff_mat[center_indices]
            assert complex(c00).imag <= 0.001, "Center coefficient must be real-valued."
            c00 = real(c00)
            factor_for_id = c00 * (2 * pi) ** self.dim
            normalization_factor = factor_for_id
        else:
            warnings.warn(
                "Normalization:cannotTest: Unable to test if normalized "
                f"for transformation '{self.transformation}'.",
                RuntimeWarning,
            )
            return self

        if c00 < 0:
            warnings.warn(
                "Normalization:negative: C00 is negative. "
                "This can either be a user error or caused by a "
                "non-square-rooted representation.",
                RuntimeWarning,
            )
        elif abs(c00) < 1e-200:
            raise ValueError(
                "Normalization:almostZero: C00 is too close to zero; "
                "this usually points to a user error."
            )
        elif abs(factor_for_id - 1) > tol:
            if warn_unnorm:
                warnings.warn(
                    "Normalization:notNormalized: Coefficients apparently do "
                    "not belong to a normalized density. Normalizing...",
                    RuntimeWarning,
                )
        else:
            # Already normalized
            return self

        self.coeff_mat = self.coeff_mat / normalization_factor
        return self

    @beartype
    def truncate(
        self, n_coefficients: tuple[int, ...], force_normalization: bool = False
    ):
        """
        Truncate or pad the coefficient tensor to a desired size, centered.

        n_coefficients : int or sequence of ints
            Desired number of complex coefficients along each dimension.
            Must be odd and >1 except possibly in the 1-D case.

        force_normalization : bool
            If True, ensures the resulting distribution is normalized
            even if the transformation is 'identity'.
        """
        assert all((array(n_coefficients) - 1) % 2 == 0)

        if not all(array(n_coefficients) > 0):
            raise ValueError("truncate: n_coefficients must be positive.")

        current_shape = self.coeff_mat.shape

        # Already correct size
        if all(current_shape == n_coefficients):
            result = copy.deepcopy(self)
            if force_normalization:
                result.normalize_in_place(warn_unnorm=False)
            return result

        if any(array(current_shape) < array(n_coefficients)):
            warnings.warn(
                "Truncate:TooFewCoefficients: At least in one dimension, "
                "truncate has to fill up due to too few coefficients.",
                RuntimeWarning,
            )

        new_shape = tuple(int(x) for x in n_coefficients)
        coeff_new = zeros(new_shape, dtype=self.coeff_mat.dtype)

        # Centre-aligned copy of overlapping region
        slices_old = []
        slices_new = []
        for old_len, new_len in zip(current_shape, n_coefficients):
            overlap = int(min(old_len, new_len))
            start_old = int((old_len - overlap) // 2)
            start_new = int((new_len - overlap) // 2)
            slices_old.append(slice(start_old, start_old + overlap))
            slices_new.append(slice(start_new, start_new + overlap))

        coeff_new[tuple(slices_new)] = self.coeff_mat[tuple(slices_old)]

        result = copy.deepcopy(self)
        result.coeff_mat = coeff_new

        # Truncation can void normalization for non-identity transformations
        if force_normalization or (
            self.transformation != "identity" and any(n_coefficients < current_shape)
        ):
            result.normalize_in_place(warn_unnorm=False)

        return result

    def transform_via_coefficients(
        self, desired_transformation: str, n_coefficients=None
    ):
        """
        Transform the representation (e.g., square the underlying function)
        directly in coefficient space, where supported.

        Currently supports only desired_transformation == 'square'.
        """
        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape

        assert all((array(n_coefficients) - 1) % 2 == 0)

        if desired_transformation == "identity":
            return copy.deepcopy(self)

        if desired_transformation != "square":
            raise ValueError(
                "Desired transformation not supported via coefficients; "
                "only 'square' is implemented."
            )

        if self.transformation == "sqrt":
            new_trans = "identity"
        elif self.transformation == "identity":
            new_trans = "square"
        else:
            new_trans = "multiple"

        current_shape = self.coeff_mat.shape

        # Convolution in coefficient space (multi-dimensional)
        if all(n_coefficients <= current_shape):
            mode = "same"
        else:
            mode = "full"

        if pyrecest.backend.__backend_name__ == "pytorch":  # pylint: disable=no-member
            raise NotImplementedError(
                "transform_via_coefficients (square) is not supported for the pytorch backend."
            )
        conv = signal.fftconvolve(self.coeff_mat, self.coeff_mat, mode=mode)
        result = copy.deepcopy(self)
        result.coeff_mat = conv
        result.transformation = new_trans

        # Enforce normalization
        return result.truncate(n_coefficients, force_normalization=True)

    def multiply(self, f2: "HypertoroidalFourierDistribution", n_coefficients=None):
        """
        Multiply two hypertoroidal Fourier distributions (same transformation).
        """
        if self.transformation != f2.transformation:
            raise ValueError("Transformations must match for multiply().")

        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape

        if self.transformation == "log":
            # Log-space: multiplication becomes addition of log-densities
            f1 = self
            g2 = f2
            if any(array(f1.coeff_mat.shape) != n_coefficients):
                f1 = f1.truncate(n_coefficients)
            if any(array(g2.coeff_mat.shape) != n_coefficients):
                g2 = g2.truncate(n_coefficients)

            result = copy.deepcopy(self)
            result.coeff_mat = f1.coeff_mat + g2.coeff_mat
            warnings.warn(
                "Multiply:NotNormalizing: Not performing normalization when "
                "using log transformation.",
                RuntimeWarning,
            )
            return result

        if self.transformation in ("identity", "sqrt"):
            current_shape = self.coeff_mat.shape
            if all(array(n_coefficients) <= array(current_shape)):
                mode = "same"
            else:
                mode = "full"

            conv = signal.fftconvolve(self.coeff_mat, f2.coeff_mat, mode=mode)
            result = copy.deepcopy(self)
            result.coeff_mat = conv
            # Truncate to desired size and enforce normalization
            return result.truncate(n_coefficients, force_normalization=True)

        raise ValueError(
            "Multiply:unsupportedTransformation: "
            f"Transformation '{self.transformation}' not recognized or unsupported."
        )

    @beartype
    def trigonometric_moment(self, n: int):
        """
        Compute the n-th trigonometric moment for each angular component.

        Returns a vector m of length self.dim where

            m[d] = E[exp(i * n * x_d)],  for d = 0..dim-1
        """
        n = int(n)
        if n == 0:
            return ones(self.dim)

        if n < 0:
            return conj(self.trigonometric_moment(-n))

        # Ensure we have at least 2n+1 coefficients in each dimension
        target_size = tuple(int(x) for x in (2 * n * ones(self.dim, dtype=int) + 1))

        if self.transformation == "sqrt":
            tfd = self.transform_via_coefficients("square", target_size)
        elif self.transformation == "identity":
            tfd = self.truncate(target_size)
        else:
            raise ValueError(
                "Transformation not recognized or unsupported for trigonometric_moment."
            )

        coeff = tfd.coeff_mat
        shape_arr = array(coeff.shape, dtype=int)
        center = (shape_arr - 1) // 2  # index corresponding to k=0

        # For moment n we need coefficients with k_d = -n and all other k_j = 0.
        m = zeros(self.dim, dtype=complex)
        for d in range(self.dim):
            idx = [int(c) for c in center]
            idx[d] = 0  # 0-based index for k_d = -n
            m[d] = (2 * pi) ** self.dim * coeff[tuple(idx)]

        return m

    @staticmethod
    @beartype
    def from_function(
        fun,
        n_coefficients: tuple[int, ...],
        desired_transformation: str = "sqrt",
    ) -> "HypertoroidalFourierDistribution":
        """
        Construct a Fourier distribution by sampling a function on a grid.

        Parameters
        ----------
        fun : callable
            Takes 'dim' arrays as input (each axis of the grid) and returns
            an array of the same shape representing the (possibly
            unnormalized) pdf on the grid.
        n_coefficients : tuple[int]
            Desired number of Fourier coefficients along each dimension.
        desired_transformation : str
            Transformation to apply to the function values before computing
            Fourier coefficients. One of 'sqrt', 'log', 'identity'.
        """
        # Check that number of arguments matches dimensionality, where possible
        axes = [linspace(0.0, 2.0 * pi, int(n), endpoint=False) for n in n_coefficients]
        grid = meshgrid(*axes, indexing="ij")
        fvals = fun(*grid)  # expect vectorized function

        if math.prod(fvals.shape) != math.prod(n_coefficients):
            raise ValueError(
                "Size of output of function is incorrect. "
                "Please ensure that it returns exactly one scalar per grid point."
            )

        fvals = reshape(fvals, tuple(int(n) for n in n_coefficients))

        return HypertoroidalFourierDistribution.from_function_values(
            fvals,
            n_coefficients=n_coefficients,
            desired_transformation=desired_transformation,
        )

    @classmethod
    @beartype
    def from_function_values(
        cls,
        fvals,
        n_coefficients: tuple[int, ...] | None = None,
        desired_transformation: str = "sqrt",
        already_transformed: bool = False,
    ) -> "HypertoroidalFourierDistribution":
        """
        Creates Fourier distribution from function values on a regular grid.

        fvals : n-D array
            Values of the (possibly unnormalized) density on a regular grid.

        n_coefficients : tuple[int, ...], optional
            Desired number of Fourier coefficients along each dimension.
            If None, defaults to size(fvals) plus 1 in even-dimension sizes.
        """
        if n_coefficients is None:
            n_coefficients = fvals.shape

        dim = len(n_coefficients)
        assert dim == ndim(fvals), (
            "from_function_values: n_coefficients length must match "
            "number of dimensions of fvals."
        )

        # Ensure no dimension has only one entry, except [N,1] 1‑D case
        sizes = array(shape(fvals), dtype=int)
        if not ((ndim(fvals) == 2 and sizes[1] == 1) or all(sizes > 1)):
            raise ValueError(
                "from_function_values: Some dimension has only one entry. "
                "Fix the shape of fvals."
            )

        if not already_transformed:
            if desired_transformation == "sqrt":
                fvals = sqrt(fvals)
            elif desired_transformation == "log":
                fvals = log(fvals)
            elif desired_transformation in ("identity", "custom"):
                pass  # keep as is
            else:
                raise ValueError(
                    "from_function_values:unrecognizedTransformation: "
                    f"Transformation '{desired_transformation}' not recognized "
                    "or unsupported for transformation via FFT."
                )

        # Compute Fourier coefficients via FFT
        fourier_coefficients = fft.fftshift(fft.fftn(fvals) / math.prod(fvals.shape))

        # If any axis has even length, pad and symmetrize to get odd sizes
        shape_fc = array(shape(fourier_coefficients), dtype=int)
        if not all(mod(shape_fc, 2) == 1):
            pad_width = []
            for m in mod(shape_fc, 2):
                if int(m) == 0:
                    pad_width.append((0, 1))  # pad one element at the end
                else:
                    pad_width.append((0, 0))
            fourier_coefficients = pad(fourier_coefficients, pad_width, mode="constant")

            rev_slices = tuple(
                slice(None, None, -1) for _ in range(ndim(fourier_coefficients))
            )
            fourier_coefficients = 0.5 * (
                fourier_coefficients + conj(fourier_coefficients[rev_slices])
            )

        hfd = cls(fourier_coefficients, desired_transformation)
        return hfd.truncate(n_coefficients)

    @classmethod
    def from_distribution(
        cls,
        distribution: AbstractHypertoroidalDistribution,
        n_coefficients: tuple[int, ...],
        desired_transformation: str = "sqrt",
    ) -> "HypertoroidalFourierDistribution":
        """
        Approximate a given hypertoroidal distribution with a Fourier series.

        n_coefficients can be a single integer (broadcast to all dimensions)
        or a sequence with length equal to distribution.dim.
        """

        # Special closed-form case: uniform distribution
        if isinstance(distribution, HypertoroidalUniformDistribution):
            coeff_shape = tuple(int(n) for n in n_coefficients)
            C = zeros(coeff_shape, dtype=complex)
            center = tuple(int((n - 1) // 2) for n in n_coefficients)

            if desired_transformation == "sqrt":
                C[center] = 1.0 / sqrt((2 * pi) ** distribution.dim)
            elif desired_transformation == "identity":
                C[center] = 1.0 / (2 * pi) ** distribution.dim
            else:
                raise ValueError(
                    "from_distribution: Transformation not recognized or unsupported."
                )

            return cls(C, desired_transformation)

        if not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise ValueError(
                "from_distribution: invalidObject: First argument has to be "
                "a hypertoroidal distribution."
            )

        # Generic case: sample pdf of the distribution on a grid
        def pdf_on_grid(*axes):
            # each axes[i] has shape (n1,...,nd)
            pts = column_stack([ax.ravel() for ax in axes])
            vals = distribution.pdf(pts)  # expects (n_eval, dim)
            return reshape(vals, axes[0].shape)

        return cls.from_function(
            pdf_on_grid,
            n_coefficients=n_coefficients,
            desired_transformation=desired_transformation,
        )

    def integrate(self, integration_boundaries=None):
        raise NotImplementedError("integrate is not implemented yet in Python.")

    @beartype
    def transform_via_fft(
        self, desired_transformation: str, n_coefficients: tuple[int, ...] | None = None
    ):
        """
        Transform the distribution by:
        1) Going to state space via IFFT (using current 'identity' coeffs),
        2) Applying the desired transformation (sqrt / log / identity / custom),
        3) Recomputing Fourier coeffs via FFT and truncating.
        """
        if self.transformation != "identity":
            raise ValueError(
                "Cannot transform via FFT if transformation is not 'identity'."
            )

        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape

        # 1) Get function values on the implicit grid from the current coeffs
        fvals_complex = fft.ifftn(fft.ifftshift(self.coeff_mat)) * prod(
            shape(self.coeff_mat)
        )
        fvals = real(fvals_complex)

        # 2) Use from_function_values to apply desired_transformation and compute new coeffs
        hfd_tmp = HypertoroidalFourierDistribution.from_function_values(
            fvals,
            n_coefficients=n_coefficients,
            desired_transformation=desired_transformation,
            already_transformed=False,
        )

        # 3) Return a copy of 'self' with new coeffs and transformation
        result = copy.deepcopy(self)
        result.coeff_mat = hfd_tmp.coeff_mat
        result.transformation = desired_transformation
        return result

    def convolve(self, f2: "HypertoroidalFourierDistribution", n_coefficients=None):
        """
        Convolution of two hypertoroidal Fourier distributions.
        """
        if self.transformation != f2.transformation:
            raise ValueError("convolve: transformations must match.")

        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape

        if self.dim != len(n_coefficients):
            raise ValueError("convolve: Incompatible dimensions for n_coefficients.")

        # --- Adjust coefficient matrices if shapes differ (truncate both to max) ---
        shape1 = self.coeff_mat.shape
        shape2 = f2.coeff_mat.shape

        if not all(shape1 == shape2):
            target_shape = tuple(max(a, b) for a, b in zip(shape1, shape2))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Truncate:TooFewCoefficients")
                hfd_tmp = self.truncate(target_shape)
                f2_tmp = f2.truncate(target_shape)
        else:
            hfd_tmp = self
            f2_tmp = f2

        # --- Main cases depending on transformation ---
        if self.transformation == "sqrt":
            conv1 = signal.fftconvolve(
                hfd_tmp.coeff_mat, hfd_tmp.coeff_mat, mode="full"
            )
            conv2 = signal.fftconvolve(f2_tmp.coeff_mat, f2_tmp.coeff_mat, mode="full")
            c_conv = (2 * pi) ** hfd_tmp.dim * conv1 * conv2

            hfd_mid = copy.deepcopy(hfd_tmp)
            hfd_mid.coeff_mat = c_conv
            hfd_mid.transformation = "identity"

            hfd = hfd_mid.transform_via_fft("sqrt", n_coefficients)
            # Normalization is performed in truncation
            hfd = hfd.truncate(n_coefficients)
            return hfd

        if self.transformation == "identity":
            c_conv = ((2 * pi) ** self.dim) * hfd_tmp.coeff_mat * f2_tmp.coeff_mat
            hfd_mid = copy.deepcopy(hfd_tmp)
            hfd_mid.coeff_mat = c_conv
            hfd_mid.transformation = "identity"
            # Enforce normalization just to be safe
            return hfd_mid.truncate(n_coefficients, force_normalization=True)

        raise ValueError(
            f"transformation:unrecognizedTransformation: "
            f"Transformation '{self.transformation}' not recognized or unsupported."
        )

    def shift(self, shift_by):
        """
        Shift distribution by shift_by (on the hypertorus).

        Parameters
        ----------
        shift_by : array_like, shape (dim,)
            Shift in radians for each angular dimension.
        """
        assert ndim(shift_by) <= 1
        if shift_by.shape[0] != self.dim:
            raise ValueError(
                f"shift: expected {self.dim} shift angles, got {shift_by.shape[0]}"
            )

        # All angles are zero, return unchanged copy
        if all(shift_by == 0):
            return copy.deepcopy(self)

        # maxk_d for each dimension, using the first `self.dim` axes
        shape_fc = array(self.coeff_mat.shape[: self.dim])
        maxk = (shape_fc - 1) // 2
        k_ranges = [arange(int(-m), int(m) + 1) for m in maxk]

        # Build integer index grid k over all dimensions
        exp_factor_mats = meshgrid(*k_ranges, indexing="ij")  # list of length dim

        # Compute exponent sum_d k_d * shift_by[d]
        exponent = 0.0
        for k_mat, angle in zip(exp_factor_mats, shift_by):
            exponent = exponent + k_mat * angle

        # Multiply coefficients by exp(-i * k⋅shift_by)
        factor = exp(-1j * exponent)
        new_coeff = self.coeff_mat * factor

        hfd = copy.deepcopy(self)
        hfd.coeff_mat = new_coeff
        return hfd
