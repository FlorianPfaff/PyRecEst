import numpy as _np

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    array,
    cos,
    exp,
    sin,
)

from ..abstract_se2_distribution import AbstractSE2Distribution
from .partially_wrapped_normal_distribution import PartiallyWrappedNormalDistribution


def _contains_masked_values(value):
    if _np.ma.is_masked(value):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(_np.ma.is_masked(item) for item in values)


def _as_se2_samples(samples):
    """Return finite real SE(2) samples with shape ``(n, 3)``."""
    if _contains_masked_values(samples):
        raise ValueError("samples must contain finite real numeric values")
    try:
        sample_array = _np.asarray(samples)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must have shape (n, 3)") from exc
    if sample_array.ndim != 2 or sample_array.shape[1] != 3:
        raise ValueError("samples must have shape (n, 3)")
    if sample_array.shape[0] < 2:
        raise ValueError("samples must contain at least two observations")
    if sample_array.dtype.kind in "bcSUMm":
        raise ValueError("samples must contain finite real numeric values")
    if sample_array.dtype == object:
        rejected_types = (
            bool,
            _np.bool_,
            str,
            bytes,
            bytearray,
            _np.str_,
            _np.bytes_,
            complex,
            _np.complexfloating,
            _np.datetime64,
            _np.timedelta64,
            type(None),
        )
        if any(isinstance(value, rejected_types) for value in sample_array.ravel()):
            raise ValueError("samples must contain finite real numeric values")
    try:
        sample_array = sample_array.astype(float, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("samples must contain finite real numeric values") from exc
    if not _np.isfinite(sample_array).all():
        raise ValueError("samples must contain finite real numeric values")
    return sample_array


class SE2PWNDistribution(PartiallyWrappedNormalDistribution, AbstractSE2Distribution):
    """Partially wrapped normal distribution for SE(2).

    The first component is the rotation angle (periodic), the second and
    third components are the 2-D translation (linear).

    References
    ----------
    Kurz, G., Gilitschenski, I., & Hanebeck, U. D. (2014). The Partially
    Wrapped Normal Distribution for SE(2) Estimation. Proceedings of the
    2014 IEEE International Conference on Multisensor Fusion and Information
    Integration.
    """

    def __init__(self, mu, C):
        PartiallyWrappedNormalDistribution.__init__(self, mu, C, 1)
        AbstractSE2Distribution.__init__(self)

    def mean4D(self):
        """Return the 4-D moment E[cos(x1), sin(x1), x2, x3].

        Returns
        -------
        array of shape (4,)
        """
        return self.hybrid_moment()

    def covariance4D(self):  # pylint: disable=too-many-locals
        """Return the analytical 4-D covariance of [cos(x1), sin(x1), x2, x3].

        Returns
        -------
        array of shape (4, 4)
        """
        mu0 = self.mu[0]
        c11 = self.C[0, 0]
        c12 = self.C[0, 1]
        c13 = self.C[0, 2]

        a = 1 - exp(-c11)
        half_a = 0.5 * a
        e1 = exp(-c11)
        e_half = exp(-0.5 * c11)

        si11 = half_a * (1 - e1 * cos(2 * mu0))
        si22 = half_a * (1 + e1 * cos(2 * mu0))
        si33 = self.C[1, 1]
        si44 = self.C[2, 2]

        si12 = -half_a * e1 * sin(2 * mu0)
        si13 = -e_half * c12 * sin(mu0)
        si23 = e_half * c12 * cos(mu0)
        si14 = -e_half * c13 * sin(mu0)
        si24 = e_half * c13 * cos(mu0)
        si34 = self.C[1, 2]

        return array(
            [
                [si11, si12, si13, si14],
                [si12, si22, si23, si24],
                [si13, si23, si33, si34],
                [si14, si24, si34, si44],
            ]
        )

    def covariance4D_numerical(self, n_samples=10000):
        """Estimate the 4-D covariance of [cos(x1), sin(x1), x2, x3] from samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw for the estimate.

        Returns
        -------
        array of shape (4, 4)
        """
        s = _np.asarray(self.sample(n_samples))
        big_s = _np.column_stack([_np.cos(s[:, 0]), _np.sin(s[:, 0]), s[:, 1], s[:, 2]])
        return array(_np.cov(big_s.T))

    @staticmethod
    def from_samples(samples):
        """Fit an SE2PWNDistribution from samples via moment matching.

        Parameters
        ----------
        samples : array of shape (n, 3)
            Samples on [0, 2*pi) x R^2.

        Returns
        -------
        SE2PWNDistribution
        """
        samples = _as_se2_samples(samples)
        big_s = _np.column_stack(
            [
                _np.cos(samples[:, 0]),
                _np.sin(samples[:, 0]),
                samples[:, 1],
                samples[:, 2],
            ]
        )
        mu4 = _np.mean(big_s, axis=0)

        mu = _np.zeros(3)
        mu[0] = _np.mod(_np.arctan2(mu4[1], mu4[0]), 2.0 * _np.pi)
        m1abs = _np.hypot(mu4[0], mu4[1])
        if m1abs <= _np.finfo(float).eps:
            raise ValueError(
                "sample angles must have a non-zero first trigonometric moment"
            )
        m1abs = min(float(m1abs), 1.0)
        mu[1:] = mu4[2:]

        # Moment matching uses empirical expectations normalized by n. NumPy's
        # default n-1 correction would inflate every fitted covariance block.
        c4 = _np.cov(big_s.T, bias=True)

        c = _np.zeros((3, 3))
        c[0, 0] = -2.0 * _np.log(m1abs)
        factor = _np.exp(0.5 * c[0, 0])
        c[0, 1] = (-c4[0, 2] * _np.sin(mu[0]) + c4[1, 2] * _np.cos(mu[0])) * factor
        c[0, 2] = (-c4[0, 3] * _np.sin(mu[0]) + c4[1, 3] * _np.cos(mu[0])) * factor
        c[1, 0] = c[0, 1]
        c[2, 0] = c[0, 2]
        c[1:3, 1:3] = c4[2:4, 2:4]

        return SE2PWNDistribution(array(mu), array(c))
