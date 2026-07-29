import copy
import math

import mpmath
import numpy as np
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    allclose,
    array,
    concatenate,
    diag,
    exp,
    full,
    gammaln,
    hstack,
    isfinite,
    linalg,
    log,
    ndim,
    ones,
    tile,
    zeros,
)

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution

_INVALID_REAL_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
    np.datetime64,
    np.timedelta64,
)


def _as_python_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _as_finite_scalar(value, name: str) -> float:
    message = f"{name} must be a finite real scalar."
    dtype_kind = getattr(getattr(value, "dtype", None), "kind", None)
    if isinstance(value, _INVALID_REAL_SCALAR_TYPES) or dtype_kind in {
        "b",
        "c",
        "S",
        "U",
        "M",
        "m",
    }:
        raise ValueError(message)

    shape = getattr(value, "shape", None)
    if shape is not None and tuple(shape) != ():
        raise ValueError(message)

    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, _INVALID_REAL_SCALAR_TYPES):
        raise ValueError(message)
    try:
        scalar = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc

    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _as_unit_direction(mu, *, name: str = "mu", tolerance: float = 1e-6):
    mu = array(mu)
    if ndim(mu) != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if mu.shape[0] < 2:
        raise ValueError(f"{name} must be at least two-dimensional")
    if not _as_python_bool(all(isfinite(mu))):
        raise ValueError(f"{name} must contain only finite values")
    if not _as_python_bool(abs(linalg.norm(mu) - 1.0) < tolerance):
        raise ValueError(f"{name} is unnormalized")
    return mu


class WatsonDistribution(AbstractHypersphericalDistribution):
    EPSILON = 1e-6

    def __init__(self, mu, kappa, norm_const: float | None = None):
        """
        Initializes a new instance of the WatsonDistribution class.

        Args:
            mu (): The mean direction of the distribution.
            kappa (float): The concentration parameter of the distribution.
        """
        mu = _as_unit_direction(mu, tolerance=self.EPSILON)
        kappa = _as_finite_scalar(kappa, "kappa")
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)

        self.mu = mu
        self.kappa = kappa
        self._norm_const = norm_const
        self._ln_norm_const = log(norm_const) if norm_const is not None else None

    @property
    def norm_const(self):
        if self._norm_const is None:
            self._norm_const = array(
                float(
                    mpmath.gamma((self.dim + 1) / 2)
                    / (2 * float(pyrecest.backend.pi) ** ((self.dim + 1) / 2))
                    / mpmath.hyper([0.5], [(self.dim + 1) / 2.0], self.kappa)
                )
            )
        return self._norm_const

    @property
    def ln_norm_const(self):
        if self._ln_norm_const is None:
            self._ln_norm_const = array(
                (gammaln(array((self.dim + 1) / 2)))
                - log(2 * pyrecest.backend.pi ** ((self.dim + 1) / 2))
                - float(
                    mpmath.log(mpmath.hyper([0.5], [(self.dim + 1) / 2.0], self.kappa))
                )
            )
        return self._ln_norm_const

    def pdf(self, xs):
        """
        Computes the probability density function at xs.

        Args:
            xs: The values at which to evaluate the pdf.

        Returns:
            np.generic: The value of the pdf at xs.
        """
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
            raise ValueError(
                f"xs must have trailing dimension {self.input_dim}, got {xs.shape}."
            )
        p = self.norm_const * exp(self.kappa * (xs @ self.mu) ** 2)
        return p

    def ln_pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
            raise ValueError(
                f"xs must have trailing dimension {self.input_dim}, got {xs.shape}."
            )
        return self.ln_norm_const + self.kappa * (xs @ self.mu) ** 2

    def to_bingham(self) -> BinghamDistribution:
        if self.kappa < 0:
            raise NotImplementedError(
                "Conversion to Bingham is not implemented for kappa<0"
            )

        M = tile(self.mu.reshape(-1, 1), (1, self.input_dim))
        E = diag(array(concatenate((array([0]), ones(self.input_dim - 1)))))
        M = M + E
        Q, _ = linalg.qr(M)
        M = hstack([Q[:, 1:], Q[:, 0].reshape(-1, 1)])
        Z = hstack((full((self.dim,), -self.kappa), array(0.0)))
        return BinghamDistribution(Z, M)

    def sample(self, n):
        if self.dim != 2:
            return self.to_bingham().sample(n)

        return super().sample(n)

    def mode(self):
        if self.kappa >= 0:
            return self.mu

        return self.mode_numerical()

    def set_mode(self, new_mode):
        new_mode = _as_unit_direction(new_mode, name="new_mode", tolerance=self.EPSILON)
        if new_mode.shape != self.mu.shape:
            raise ValueError("new_mode must have the same shape as mu")
        dist = copy.deepcopy(self)
        dist.mu = copy.deepcopy(new_mode)
        return dist

    def shift(self, shift_by):
        canonical_mu = concatenate((zeros(self.input_dim - 1), array([1.0])))
        if not _as_python_bool(allclose(self.mu, canonical_mu)):
            raise ValueError(
                "There is no true shifting for the hypersphere. This is a function "
                "for compatibility and only works when mu is [0,0,...,1]."
            )
        return self.set_mode(shift_by)
