import copy
from math import factorial

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    arctan2,
    array,
    cos,
    exp,
    isfinite,
    linalg,
    max,
    mod,
    pi,
    sin,
    sqrt,
)
from scipy.integrate import dblquad
from scipy.special import iv

from ..circle.custom_circular_distribution import CustomCircularDistribution
from ._input_validation import as_shift_vector
from .abstract_toroidal_bivar_vm_distribution import (
    _as_python_bool,
    validate_toroidal_vm_parameters,
)
from .abstract_toroidal_distribution import AbstractToroidalDistribution

_2pi = 2.0 * pi


class ToroidalVMMatrixDistribution(AbstractToroidalDistribution):
    """Bivariate von Mises distribution, matrix version.

    References
    ----------
    Mardia, K. V. (1975). Statistics of Directional Data. Journal of the
    Royal Statistical Society: Series B, 37, 349-393.

    Mardia, K. V., & Jupp, P. E. (1999). Directional Statistics. Wiley.

    Kurz, G., & Hanebeck, U. D. (2015). Toroidal Information Fusion Based on
    the Bivariate von Mises Distribution. Proceedings of the 2015 IEEE
    International Conference on Multisensor Fusion and Information
    Integration.
    """

    def __init__(self, mu, kappa, A):
        AbstractToroidalDistribution.__init__(self)
        mu, kappa = validate_toroidal_vm_parameters(
            mu, kappa, require_positive_kappa=True
        )
        A = array(A)
        if A.shape != (2, 2):
            raise ValueError("A must have shape (2, 2)")
        if not _as_python_bool(all(isfinite(A))):
            raise ValueError("A must contain only finite values")

        self.mu = mod(mu, _2pi)
        self.kappa = kappa
        self.A = A

        use_numerical = kappa[0] > 1.5 or kappa[1] > 1.5 or max(abs(A)) > 1.0

        if use_numerical:
            self.C = 1.0
            Cinv, _ = dblquad(
                lambda y, x: self.pdf(array([x, y])).item(),
                0.0,
                _2pi,
                0.0,
                _2pi,
            )
            self.C = 1.0 / Cinv
        else:
            self.C = self._norm_const_approx()

    def pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 1:
            if xs.shape[0] != self.dim:
                raise ValueError(
                    f"xs must have trailing dimension {self.dim}, got {xs.shape}."
                )
            xs = xs.reshape((1, self.dim))
        elif xs.ndim == 0 or xs.shape[-1] != self.dim:
            raise ValueError(
                f"xs must have trailing dimension {self.dim}, got {xs.shape}."
            )
        x1_mm = xs[..., 0] - self.mu[0]
        x2_mm = xs[..., 1] - self.mu[1]
        exponent = (
            self.kappa[0] * cos(x1_mm)
            + self.kappa[1] * cos(x2_mm)
            + cos(x1_mm) * self.A[0, 0] * cos(x2_mm)
            + cos(x1_mm) * self.A[0, 1] * sin(x2_mm)
            + sin(x1_mm) * self.A[1, 0] * cos(x2_mm)
            + sin(x1_mm) * self.A[1, 1] * sin(x2_mm)
        )
        return self.C * exp(exponent)

    def _norm_const_approx(self, n=8):
        """Approximate normalization constant using Taylor series (up to n=8 summands)."""
        a11 = self.A[0, 0]
        a12 = self.A[0, 1]
        a21 = self.A[1, 0]
        a22 = self.A[1, 1]
        k1 = self.kappa[0]
        k2 = self.kappa[1]
        pi_f = pi

        total = 4 * pi_f**2  # n=0 term
        # n=1 term is zero
        if n >= 2:
            total += (
                (a11**2 + a12**2 + a21**2 + a22**2 + 2 * k1**2 + 2 * k2**2)
                * pi_f**2
                / factorial(2)
            )
        if n >= 3:
            total += 6 * a11 * k1 * k2 * pi_f**2 / factorial(3)
        if n >= 4:
            total += (
                3
                / 16
                * (
                    3 * a11**4
                    + 3 * a12**4
                    + 3 * a21**4
                    + 8 * a11 * a12 * a21 * a22
                    + 6 * a21**2 * a22**2
                    + 3 * a22**4
                    + 8 * a21**2 * k1**2
                    + 8 * a22**2 * k1**2
                    + 8 * k1**4
                    + 8 * (3 * a21**2 + a22**2 + 4 * k1**2) * k2**2
                    + 8 * k2**4
                    + 2
                    * a11**2
                    * (3 * a12**2 + 3 * a21**2 + a22**2 + 12 * (k1**2 + k2**2))
                    + 2 * a12**2 * (a21**2 + 3 * a22**2 + 4 * (3 * k1**2 + k2**2))
                )
                * pi_f**2
                / factorial(4)
            )
        if n >= 5:
            total += (
                15
                / 4
                * pi_f**2
                * k1
                * k2
                * (
                    3 * a11**3
                    + 3 * a11 * a12**2
                    + 3 * a11 * a21**2
                    + a11 * a22**2
                    + 4 * a11 * k1**2
                    + 4 * a11 * k2**2
                    + 2 * a12 * a21 * a22
                )
                / factorial(5)
            )
        if n >= 6:
            total += (
                5
                / 64
                * pi_f**2
                * (
                    5 * a11**6
                    + 15 * a11**4 * a12**2
                    + 15 * a11**4 * a21**2
                    + 3 * a11**4 * a22**2
                    + 90 * a11**4 * k1**2
                    + 90 * a11**4 * k2**2
                    + 24 * a11**3 * a12 * a21 * a22
                    + 15 * a11**2 * a12**4
                    + 18 * a11**2 * a12**2 * a21**2
                    + 18 * a11**2 * a12**2 * a22**2
                    + 180 * a11**2 * a12**2 * k1**2
                    + 108 * a11**2 * a12**2 * k2**2
                    + 15 * a11**2 * a21**4
                    + 18 * a11**2 * a21**2 * a22**2
                    + 108 * a11**2 * a21**2 * k1**2
                    + 180 * a11**2 * a21**2 * k2**2
                    + 3 * a11**2 * a22**4
                    + 36 * a11**2 * a22**2 * k1**2
                    + 36 * a11**2 * a22**2 * k2**2
                    + 120 * a11**2 * k1**4
                    + 648 * a11**2 * k1**2 * k2**2
                    + 120 * a11**2 * k2**4
                    + 24 * a11 * a12**3 * a21 * a22
                    + 24 * a11 * a12 * a21**3 * a22
                    + 24 * a11 * a12 * a21 * a22**3
                    + 144 * a11 * a12 * a21 * a22 * k1**2
                    + 144 * a11 * a12 * a21 * a22 * k2**2
                    + 5 * a12**6
                    + 3 * a12**4 * a21**2
                    + 15 * a12**4 * a22**2
                    + 90 * a12**4 * k1**2
                    + 18 * a12**4 * k2**2
                    + 3 * a12**2 * a21**4
                    + 18 * a12**2 * a21**2 * a22**2
                    + 36 * a12**2 * a21**2 * k1**2
                    + 36 * a12**2 * a21**2 * k2**2
                    + 15 * a12**2 * a22**4
                    + 108 * a12**2 * a22**2 * k1**2
                    + 36 * a12**2 * a22**2 * k2**2
                    + 120 * a12**2 * k1**4
                    + 216 * a12**2 * k1**2 * k2**2
                    + 24 * a12**2 * k2**4
                    + 5 * a21**6
                    + 15 * a21**4 * a22**2
                    + 18 * a21**4 * k1**2
                    + 90 * a21**4 * k2**2
                    + 15 * a21**2 * a22**4
                    + 36 * a21**2 * a22**2 * k1**2
                    + 108 * a21**2 * a22**2 * k2**2
                    + 24 * a21**2 * k1**4
                    + 216 * a21**2 * k1**2 * k2**2
                    + 120 * a21**2 * k2**4
                    + 5 * a22**6
                    + 18 * a22**4 * k1**2
                    + 18 * a22**4 * k2**2
                    + 24 * a22**2 * k1**4
                    + 72 * a22**2 * k1**2 * k2**2
                    + 24 * a22**2 * k2**4
                    + 16 * k1**6
                    + 144 * k1**4 * k2**2
                    + 144 * k1**2 * k2**4
                    + 16 * k2**6
                )
                / factorial(6)
            )
        if n >= 7:
            total += (
                105
                / 32
                * k1
                * k2
                * pi_f**2
                * (
                    5 * a11**5
                    + 10 * a11**3 * a12**2
                    + 10 * a11**3 * a21**2
                    + 2 * a11**3 * a22**2
                    + 20 * a11**3 * k1**2
                    + 20 * a11**3 * k2**2
                    + 12 * a11**2 * a12 * a21 * a22
                    + 5 * a11 * a12**4
                    + 6 * a11 * a12**2 * a21**2
                    + 6 * a11 * a12**2 * a22**2
                    + 20 * a11 * a12**2 * k1**2
                    + 12 * a11 * a12**2 * k2**2
                    + 5 * a11 * a21**4
                    + 6 * a11 * a21**2 * a22**2
                    + 12 * a11 * a21**2 * k1**2
                    + 20 * a11 * a21**2 * k2**2
                    + a11 * a22**4
                    + 4 * a11 * a22**2 * k1**2
                    + 4 * a11 * a22**2 * k2**2
                    + 8 * a11 * k1**4
                    + 24 * a11 * k1**2 * k2**2
                    + 8 * a11 * k2**4
                    + 4 * a12**3 * a21 * a22
                    + 4 * a12 * a21**3 * a22
                    + 4 * a12 * a21 * a22**3
                    + 8 * a12 * a21 * a22 * k1**2
                    + 8 * a12 * a21 * a22 * k2**2
                )
                / factorial(7)
            )
        return 1.0 / total

    def multiply(self, other):
        """Multiply two ToroidalVMMatrixDistributions (exact product)."""
        if not isinstance(other, ToroidalVMMatrixDistribution):
            raise ValueError("other must be a ToroidalVMMatrixDistribution")

        C1 = self.kappa[0] * cos(self.mu[0]) + other.kappa[0] * cos(other.mu[0])
        S1 = self.kappa[0] * sin(self.mu[0]) + other.kappa[0] * sin(other.mu[0])
        C2 = self.kappa[1] * cos(self.mu[1]) + other.kappa[1] * cos(other.mu[1])
        S2 = self.kappa[1] * sin(self.mu[1]) + other.kappa[1] * sin(other.mu[1])

        mu_new = array([arctan2(S1, C1) % _2pi, arctan2(S2, C2) % _2pi])
        kappa_new = array([sqrt(C1**2 + S1**2), sqrt(C2**2 + S2**2)])

        def _M(mu_vec):
            c1 = cos(mu_vec[0])
            s1 = sin(mu_vec[0])
            c2 = cos(mu_vec[1])
            s2 = sin(mu_vec[1])
            return array(
                [
                    [c1 * c2, -s1 * c2, -c1 * s2, s1 * s2],
                    [s1 * c2, c1 * c2, -s1 * s2, -c1 * s2],
                    [c1 * s2, -s1 * s2, c1 * c2, -s1 * c2],
                    [s1 * s2, c1 * s2, s1 * c2, c1 * c2],
                ]
            )

        A1 = array([[self.A[0, 0]], [self.A[1, 0]], [self.A[0, 1]], [self.A[1, 1]]])
        A2 = array([[other.A[0, 0]], [other.A[1, 0]], [other.A[0, 1]], [other.A[1, 1]]])
        b = _M(self.mu) @ A1 + _M(other.mu) @ A2
        a_vec = linalg.solve(_M(mu_new), b).ravel()
        A_new = array([[a_vec[0], a_vec[2]], [a_vec[1], a_vec[3]]])

        return ToroidalVMMatrixDistribution(mu_new, kappa_new, A_new)

    def marginalize_to_1d(self, dimension):
        """Get marginal distribution in the given dimension (0 or 1, 0-indexed).

        Integrates out the *other* dimension analytically using the Bessel
        function identity for the von-Mises-type integral.
        """
        if dimension not in (0, 1):
            raise ValueError("dimension must be 0 or 1")
        other = 1 - dimension

        mu_d = self.mu[dimension]
        k_d = self.kappa[dimension]
        k_o = self.kappa[other]
        a11 = self.A[0, 0]
        a12 = self.A[0, 1]
        a21 = self.A[1, 0]
        a22 = self.A[1, 1]
        C_val = self.C

        if dimension == 0:
            # Integrate over x2; x = x1
            def f(x):
                dx = x - mu_d
                c, s = cos(dx), sin(dx)
                alpha = k_o + c * a11 + s * a21
                beta = c * a12 + s * a22
                return 2.0 * pi * C_val * iv(0, sqrt(alpha**2 + beta**2)) * exp(k_d * c)

        else:
            # Integrate over x1; x = x2
            def f(x):
                dx = x - mu_d
                c, s = cos(dx), sin(dx)
                alpha = k_o + c * a11 + s * a12
                beta = c * a21 + s * a22
                return 2.0 * pi * C_val * iv(0, sqrt(alpha**2 + beta**2)) * exp(k_d * c)

        return CustomCircularDistribution(f)

    def shift(self, shift_by):
        """Return a copy of this distribution shifted by shift_by."""
        shift_by = as_shift_vector(shift_by, self.dim)
        result = copy.copy(self)
        result.mu = mod(self.mu + shift_by, _2pi)
        return result
