import copy
import math
from typing import Union

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    arccos,
    array,
    clip,
    concatenate,
    cos,
    exp,
    int32,
    int64,
    isfinite,
    isnan,
    linalg,
    ndim,
    ones,
    pi,
    sin,
    stack,
    zeros,
)
from scipy.special import ive

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


def _as_python_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _as_finite_scalar(value, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc

    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _as_unit_direction(mu, *, name: str = "mu", tolerance: float = 1e-6):
    mu = array(mu)
    if ndim(mu) != 1:
        raise ValueError(f"{name} must be a vector")
    if mu.shape[0] < 2:
        raise ValueError(f"{name} must be at least two-dimensional")
    if not _as_python_bool(all(isfinite(mu))):
        raise ValueError(f"{name} must contain only finite values")
    if not _as_python_bool(abs(linalg.norm(mu) - 1.0) < tolerance):
        raise ValueError(f"{name} must be normalized")
    return mu


def _scaled_log_normalization(input_dim: int, kappa: float) -> float:
    """Return ``log(C_d(kappa)) + kappa`` without Bessel overflow."""
    order = input_dim / 2.0 - 1.0
    scaled_bessel = float(ive(order, kappa))
    if not math.isfinite(scaled_bessel) or scaled_bessel <= 0.0:
        raise ValueError(
            "Could not compute a finite positive scaled Bessel value for kappa."
        )
    return (
        order * math.log(kappa)
        - input_dim / 2.0 * math.log(2.0 * math.pi)
        - math.log(scaled_bessel)
    )


class VonMisesFisherDistribution(AbstractHypersphericalDistribution):
    """
    von Mises-Fisher distribution on the unit hypersphere.

    The distribution is defined on unit vectors in ``R^d``. In PyRecEst,
    ``input_dim`` is ``d`` and ``dim`` is the manifold dimension ``d - 1``.
    Von Mises-Fisher distribution on the hypersphere.

    References
    ----------
    Fisher, R. (1953). Dispersion on a sphere. Proceedings of the Royal
    Society of London. Series A, Mathematical and Physical Sciences,
    217(1130), 295-305.
    """

    _KAPPA_EPS = 1e-12

    def __init__(self, mu, kappa):
        """Create a von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array-like, shape (d,)
            Unit mean direction in the embedding space. For ``kappa == 0``,
            this direction is arbitrary because the distribution is uniform.
        kappa : float
            Nonnegative concentration parameter. Larger values concentrate more
            mass around ``mu``. ``kappa == 0`` is the uniform distribution on the
            hypersphere.
        """
        mu = _as_unit_direction(mu)
        kappa_scalar = _as_finite_scalar(kappa, "kappa")
        if kappa_scalar < 0.0:
            raise ValueError("kappa must be a nonnegative scalar")
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)

        self.mu = mu
        self.kappa = kappa

        if kappa_scalar <= self._KAPPA_EPS:
            self.C = 1.0 / self.compute_unit_hypersphere_surface(self.dim)
            self._log_scaled_normalization = math.log(float(self.C)) + kappa_scalar
        else:
            self._log_scaled_normalization = _scaled_log_normalization(
                self.input_dim, kappa_scalar
            )
            self.C = array(math.exp(self._log_scaled_normalization - kappa_scalar))

    def pdf(self, xs):
        """Evaluate the density at unit vectors.

        Parameters
        ----------
        xs : array-like, shape (d,) or (..., d)
            Unit-vector evaluation point or batch of points in the embedding
            space.
        """
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
            raise ValueError(
                f"xs must have trailing dimension {self.input_dim}, got {xs.shape}."
            )

        return exp(self._log_scaled_normalization + self.kappa * (xs @ self.mu - 1.0))

    def mean_direction(self):
        """Return the unit mean direction with shape ``(d,)``."""
        return self.mu

    def sample(self, n):
        """Generate random unit vectors from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        array-like, shape (n, d)
            Random samples on the unit hypersphere.

        Notes
        -----
        Sampling currently requires the NumPy backend and SciPy's
        ``vonmises_fisher`` implementation.
        """
        if pyrecest.backend.__backend_name__ != "numpy":
            raise NotImplementedError("sample is only supported on the NumPy backend.")

        if self.kappa <= self._KAPPA_EPS:
            from .hyperspherical_uniform_distribution import (
                HypersphericalUniformDistribution,
            )

            return HypersphericalUniformDistribution(self.dim).sample(n)

        from scipy.stats import vonmises_fisher

        # Create a von Mises-Fisher distribution object
        vmf = vonmises_fisher(self.mu, self.kappa)

        # Draw n random samples from the distribution
        samples = vmf.rvs(n)

        return samples

    def sample_deterministic(self):
        """Return deterministic sigma points matched to the mean direction."""
        n_samples = self.dim * 2 + 1
        columns = [array([1.0] + [0.0] * self.dim)]

        mean_res_length = self.a_d(self.input_dim, self.kappa)
        cos_alpha = clip(
            (n_samples * mean_res_length - 1.0) / (n_samples - 1),
            -1.0,
            1.0,
        )
        alpha = arccos(cos_alpha)
        for i in range(self.dim):
            tangent_row = i + 1
            positive = [cos(alpha)] + [0.0] * self.dim
            negative = [cos(alpha)] + [0.0] * self.dim
            positive[tangent_row] = sin(alpha)
            negative[tangent_row] = -sin(alpha)
            columns.extend((array(positive), array(negative)))

        samples = concatenate(tuple(column[:, None] for column in columns), axis=1)
        Q = self.get_rotation_matrix()
        samples = Q @ samples
        return samples.T

    def get_rotation_matrix(self):
        """Return an orthogonal matrix whose first column is ``mu``."""
        M = concatenate((self.mu[:, None], zeros((self.dim + 1, self.dim))), axis=1)
        Q, R = linalg.qr(M)
        if R[0, 0] < 0:
            Q = -Q
        return Q

    def mean_resultant_vector(self):
        """Return the mean resultant vector with shape ``(d,)``."""
        r = self.a_d(self.input_dim, self.kappa) * self.mu
        return r

    @staticmethod
    def from_distribution(d):
        """Fit a von Mises-Fisher distribution to mean-resultant information."""
        if d.input_dim < 2:
            raise ValueError("mu must be at least 2-D for the circular case")

        m = d.mean_resultant_vector()
        return VonMisesFisherDistribution.from_mean_resultant_vector(m)

    @staticmethod
    def _default_mean_direction(input_dim: Union[int, int32, int64]):
        """Return an arbitrary unit direction for uniform vMF objects."""
        input_dim = int(input_dim)
        return array([1.0] + [0.0] * (input_dim - 1))

    @staticmethod
    def from_mean_resultant_vector(m):
        """Create a distribution from a mean resultant vector.

        Parameters
        ----------
        m : array-like, shape (d,)
            Mean resultant vector. Its direction becomes ``mu`` and its norm is
            inverted to estimate ``kappa``. A zero vector represents the uniform
            distribution and therefore receives an arbitrary stored direction.
        """
        m = array(m)
        if ndim(m) != 1:
            raise ValueError("mu must be a vector")
        if len(m) < 2:
            raise ValueError("mu must be at least 2 for the circular case")
        if not _as_python_bool(all(isfinite(m))):
            raise ValueError("mu must contain only finite values")

        mean_res_length = linalg.norm(m)
        if mean_res_length <= VonMisesFisherDistribution._KAPPA_EPS:
            return VonMisesFisherDistribution(
                VonMisesFisherDistribution._default_mean_direction(m.shape[0]), 0.0
            )

        mean_res_vector = m / mean_res_length
        kappa_ = VonMisesFisherDistribution.a_d_inverse(m.shape[0], mean_res_length)

        V = VonMisesFisherDistribution(mean_res_vector, kappa_)
        return V

    def mode(self):
        """Return the modal direction, equal to ``mu``."""
        return self.mu

    def set_mean(self, new_mean):
        """Replace the mean direction and return the distribution."""
        new_mean = _as_unit_direction(new_mean, name="new_mean")
        if new_mean.shape != self.mu.shape:
            raise ValueError("new_mean must have the same shape as mu")
        dist = copy.deepcopy(self)
        dist.mu = copy.deepcopy(new_mean)
        return dist

    def set_mode(self, new_mode):
        """Replace the modal direction and return the distribution."""
        new_mode = _as_unit_direction(new_mode, name="new_mode")
        if new_mode.shape != self.mu.shape:
            raise ValueError("new_mode must have the same shape as mu")
        dist = copy.deepcopy(self)
        dist.mu = copy.deepcopy(new_mode)
        return dist

    def multiply(self, other: "VonMisesFisherDistribution"):
        """Multiply two vMF densities and return the normalized product."""
        if self.mu.shape != other.mu.shape:
            raise ValueError("Dimensions must match")

        mu_ = self.kappa * self.mu + other.kappa * other.mu
        kappa_ = linalg.norm(mu_)
        if kappa_ <= self._KAPPA_EPS:
            return VonMisesFisherDistribution(
                self._default_mean_direction(self.input_dim), 0.0
            )
        mu_ = mu_ / kappa_
        return VonMisesFisherDistribution(mu_, kappa_)

    def convolve(self, other: "VonMisesFisherDistribution"):
        """Convolve with a zonal vMF distribution.

        ``other`` must be zonal around the final coordinate axis unless either
        operand is uniform. Convolution with a uniform density is uniform.
        """
        if self.mu.shape != other.mu.shape:
            raise ValueError("Dimensions must match")
        if self.kappa <= self._KAPPA_EPS or other.kappa <= self._KAPPA_EPS:
            return VonMisesFisherDistribution(
                self._default_mean_direction(self.input_dim), 0.0
            )

        if not _as_python_bool(abs(other.mu[-1] - 1.0) < 1e-8):
            raise ValueError("Other is not zonal")
        d = self.dim + 1

        mu_ = self.mu
        kappa_ = VonMisesFisherDistribution.a_d_inverse(
            d,
            VonMisesFisherDistribution.a_d(d, self.kappa)
            * VonMisesFisherDistribution.a_d(d, other.kappa),
        )
        return VonMisesFisherDistribution(mu_, kappa_)

    @staticmethod
    def a_d(d: Union[int, int32, int64], kappa):
        """Return the ratio of modified Bessel functions used by vMF moments."""
        if kappa <= VonMisesFisherDistribution._KAPPA_EPS:
            return array(0.0)

        bessel1 = array(ive(d / 2, kappa))
        bessel2 = array(ive(d / 2 - 1, kappa))
        if isnan(bessel1) or isnan(bessel2):
            print(f"Bessel functions returned NaN for d={d}, kappa={kappa}")
        return bessel1 / bessel2

    @staticmethod
    def a_d_inverse(d: Union[int, int32, int64], x: float):
        """Numerically invert :meth:`a_d` for dimension ``d`` and value ``x``.

        ``x`` is a mean resultant length and therefore must lie in ``[0, 1)``.
        The boundary value ``x == 1`` is the degenerate point-mass limit and
        corresponds to infinite concentration, which cannot be represented by a
        finite von Mises-Fisher distribution.
        """
        d = int(d)
        x = float(x)
        if not math.isfinite(x):
            raise ValueError("x must be finite.")
        if x < -VonMisesFisherDistribution._KAPPA_EPS:
            raise ValueError("x must be in the interval [0, 1).")
        if x <= VonMisesFisherDistribution._KAPPA_EPS:
            return 0.0
        if x >= 1.0:
            raise ValueError(
                "x must be smaller than 1; x == 1 corresponds to infinite kappa."
            )

        kappa_ = x * (d - x**2) / (1 - x**2)
        if not math.isfinite(kappa_) or kappa_ <= 0:
            raise ValueError(
                "Initial kappa estimate is not finite. "
                "x is likely too close to 1 for stable inversion."
            )

        max_steps = 20
        epsilon = 1e-7

        for _ in range(max_steps):
            kappa_old = kappa_
            ad_value = float(VonMisesFisherDistribution.a_d(d, kappa_old))
            if not math.isfinite(ad_value):
                raise ValueError(
                    f"a_d returned a non-finite value during inversion for d={d}, "
                    f"kappa={kappa_old}, x={x}. x may be too close to 1 for a "
                    "stable finite concentration estimate."
                )

            denominator = 1 - ad_value**2 - (d - 1) / kappa_old * ad_value
            if not math.isfinite(denominator) or denominator == 0:
                raise ValueError(
                    f"Newton denominator became non-finite or zero during inversion "
                    f"for d={d}, kappa={kappa_old}, x={x}."
                )

            kappa_ = kappa_old - (ad_value - x) / denominator

            if not math.isfinite(kappa_) or kappa_ < 0:
                raise ValueError(
                    f"kappa became non-finite or negative during inversion for d={d}, "
                    f"kappa_old={kappa_old}, x={x}."
                )

            if math.fabs(kappa_ - kappa_old) < epsilon:
                break

        return kappa_
