import copy
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
    cos,
    exp,
    int32,
    int64,
    isnan,
    linalg,
    ndim,
    pi,
    sin,
    sinh,
    zeros,
)
from scipy.special import iv, ive

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


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

    def __init__(self, mu, kappa):
        """Create a von Mises-Fisher distribution.

        Parameters
        ----------
        mu : array-like, shape (d,)
            Unit mean direction in the embedding space.
        kappa : float
            Positive concentration parameter. Larger values concentrate more
            mass around ``mu``.
        """
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)
        epsilon = 1e-6
        assert mu.ndim == 1, "mu must be a vector"
        assert (
            mu.shape[0] >= 2
        ), "mu must be at least two-dimensional for the circular case"
        assert kappa > 0, "kappa must be a positive scalar"
        assert abs(linalg.norm(mu) - 1.0) < epsilon, "mu must be a normalized"

        self.mu = mu
        self.kappa = kappa

        if self.dim == 2:
            self.C = kappa / (4 * pi * sinh(kappa))
        else:
            self.C = kappa ** ((self.dim + 1) / 2.0 - 1) / (
                (2.0 * pi) ** ((self.dim + 1) / 2.0) * iv((self.dim + 1) / 2 - 1, kappa)
            )

    def pdf(self, xs):
        """Evaluate the density at unit vectors.

        Parameters
        ----------
        xs : array-like, shape (d,) or (..., d)
            Unit-vector evaluation point or batch of points in the embedding
            space.
        """
        assert xs.shape[-1] == self.input_dim

        return self.C * exp(self.kappa * xs @ self.mu)

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
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported on NumPy backend"
        from scipy.stats import vonmises_fisher

        # Create a von Mises-Fisher distribution object
        vmf = vonmises_fisher(self.mu, self.kappa)

        # Draw n random samples from the distribution
        samples = vmf.rvs(n)

        return samples

    def sample_deterministic(self):
        """Return deterministic sigma points matched to the mean direction."""
        n_samples = self.dim * 2 + 1
        samples = zeros((self.input_dim, n_samples))
        samples[0, 0] = 1.0

        mean_res_length = self.a_d(self.input_dim, self.kappa)
        cos_alpha = clip(
            (n_samples * mean_res_length - 1.0) / (n_samples - 1),
            -1.0,
            1.0,
        )
        alpha = arccos(cos_alpha)
        for i in range(self.dim):
            positive_col = 2 * i + 1
            negative_col = positive_col + 1
            tangent_row = i + 1
            samples[0, positive_col] = cos(alpha)
            samples[0, negative_col] = cos(alpha)
            samples[tangent_row, positive_col] = sin(alpha)
            samples[tangent_row, negative_col] = -sin(alpha)

        Q = self.get_rotation_matrix()
        samples = Q @ samples
        return samples

    def get_rotation_matrix(self):
        """Return an orthogonal matrix whose first column is ``mu``."""
        M = zeros((self.dim + 1, self.dim + 1))
        M[:, 0] = self.mu
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
        assert d.input_dim >= 2, "mu must be at least 2-D for the circular case"

        m = d.mean_resultant_vector()
        return VonMisesFisherDistribution.from_mean_resultant_vector(m)

    @staticmethod
    def from_mean_resultant_vector(m):
        """Create a distribution from a mean resultant vector.

        Parameters
        ----------
        m : array-like, shape (d,)
            Mean resultant vector. Its direction becomes ``mu`` and its norm is
            inverted to estimate ``kappa``.
        """
        assert ndim(m) == 1, "mu must be a vector"
        assert len(m) >= 2, "mu must be at least 2 for the circular case"

        mean_res_vector = m / linalg.norm(m)
        mean_res_length = linalg.norm(m)
        kappa_ = VonMisesFisherDistribution.a_d_inverse(m.shape[0], mean_res_length)

        V = VonMisesFisherDistribution(mean_res_vector, kappa_)
        return V

    def mode(self):
        """Return the modal direction, equal to ``mu``."""
        return self.mu

    def set_mean(self, new_mean):
        """Replace the mean direction and return the distribution."""
        assert new_mean.shape == self.mu.shape
        dist = self
        dist.mu = copy.deepcopy(new_mean)
        return dist

    def set_mode(self, new_mode):
        """Replace the modal direction and return the distribution."""
        assert new_mode.shape == self.mu.shape
        dist = self
        dist.mu = copy.deepcopy(new_mode)
        return dist

    def multiply(self, other: "VonMisesFisherDistribution"):
        """Multiply two vMF densities and return the normalized product."""
        assert self.mu.shape == other.mu.shape

        mu_ = self.kappa * self.mu + other.kappa * other.mu
        kappa_ = linalg.norm(mu_)
        mu_ = mu_ / kappa_
        return VonMisesFisherDistribution(mu_, kappa_)

    def convolve(self, other: "VonMisesFisherDistribution"):
        """Convolve with a zonal vMF distribution.

        ``other`` must be zonal around the final coordinate axis.
        """
        assert other.mu[-1] == 1, "Other is not zonal"
        assert all(self.mu.shape == other.mu.shape)
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
        bessel1 = array(ive(d / 2, kappa))
        bessel2 = array(ive(d / 2 - 1, kappa))
        if isnan(bessel1) or isnan(bessel2):
            print(f"Bessel functions returned NaN for d={d}, kappa={kappa}")
        return bessel1 / bessel2

    @staticmethod
    def a_d_inverse(d: Union[int, int32, int64], x: float):
        """Numerically invert :meth:`a_d` for dimension ``d`` and value ``x``."""
        kappa_ = x * (d - x**2) / (1 - x**2)
        if isnan(kappa_):
            print(f"Initial kappa_ is NaN for d={d}, x={x}")

        max_steps = 20
        epsilon = 1e-7

        for _ in range(max_steps):
            kappa_old = kappa_
            ad_value = VonMisesFisherDistribution.a_d(d, kappa_old)
            if isnan(ad_value):
                print(
                    f"a_d returned NaN during iteration for d={d}, kappa_old={kappa_old}"
                )

            kappa_ = kappa_old - (ad_value - x) / (
                1 - ad_value**2 - (d - 1) / kappa_old * ad_value
            )

            if isnan(kappa_):
                print(
                    f"kappa_ became NaN during iteration for d={d}, kappa_old={kappa_old}, x={x}"
                )

            if abs(kappa_ - kappa_old) < epsilon:
                break

        return kappa_
