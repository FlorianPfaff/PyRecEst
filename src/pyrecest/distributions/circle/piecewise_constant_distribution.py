# pylint: disable=no-name-in-module,no-member,redefined-builtin
from numbers import Integral

import numpy as np
import pyrecest.backend
from pyrecest.backend import (
    arange,
    array,
    exp,
    floor,
    isfinite,
    log,
    mean,
    mod,
    pi,
    random,
    sum,
    zeros,
)

from .abstract_circular_distribution import AbstractCircularDistribution

_INVALID_SAMPLE_COUNT_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    np.datetime64,
    np.timedelta64,
)
_TEMPORAL_DTYPE_KINDS = {"M", "m"}


def _validate_positive_sample_count(n) -> int:
    try:
        count_array = np.asarray(n)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be a scalar integer") from exc
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")
    if count_array.dtype.kind in _TEMPORAL_DTYPE_KINDS:
        raise ValueError("n must be an integer")

    count = count_array.item()
    if isinstance(count, _INVALID_SAMPLE_COUNT_TYPES):
        raise ValueError("n must be an integer")

    try:
        count_int = int(count)
        count_float = float(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    if not np.isfinite(count_float) or not count_float.is_integer():
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


def _validate_interval_index(m, n) -> tuple[int, int]:
    if isinstance(m, bool) or isinstance(n, bool):
        raise ValueError("m and n must be integers")
    if not isinstance(m, Integral) or not isinstance(n, Integral):
        raise ValueError("m and n must be integers")

    m = int(m)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 1 <= m <= n:
        raise ValueError("m must satisfy 1 <= m <= n")
    return m, n


def _reject_complex_input(value, name: str) -> None:
    """Reject complex-valued inputs before a backend float cast can truncate them."""

    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        try:
            if np.dtype(dtype).kind == "c":
                raise ValueError(f"{name} must contain real values")
        except (TypeError, ValueError):
            if "complex" in str(dtype).lower():
                raise ValueError(f"{name} must contain real values")

    try:
        raw = np.asarray(value)
    except (OverflowError, TypeError, ValueError, RuntimeError):
        return
    if raw.dtype.kind == "c":
        raise ValueError(f"{name} must contain real values")


def _validate_moment_order(n) -> int:
    """Return an integer trigonometric-moment order without coercing other scalars."""

    try:
        order = np.asarray(n)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be a scalar integer") from exc
    if order.ndim != 0 or order.dtype.kind not in "iu":
        raise ValueError("n must be a scalar integer")
    return int(order.item())


class PiecewiseConstantDistribution(AbstractCircularDistribution):
    """Piecewise constant (i.e. discrete) circular distribution, similar to a histogram.

    The circle [0, 2*pi) is divided into n equal intervals, each with a constant
    probability density weight.

    Gerhard Kurz, Florian Pfaff, Uwe D. Hanebeck,
    Discrete Recursive Bayesian Filtering on Intervals and the Unit Circle
    Proceedings of the 2016 IEEE International Conference on Multisensor Fusion
    and Integration for Intelligent Systems (MFI 2016),
    Baden-Baden, Germany, September 2016.
    """

    def __init__(self, w):
        """Initialize with a weight vector that is automatically normalized.

        Parameters
        ----------
        w : array_like, shape (n,)
            Weight for each interval (will be normalized to form a valid pdf).
        """
        AbstractCircularDistribution.__init__(self)
        _reject_complex_input(w, "Weights")
        w = array(w, dtype=float)
        if w.ndim == 0:
            w = w.reshape((1,))
        elif w.ndim != 1:
            raise ValueError("Weights must be a one-dimensional array")
        if w.shape[0] == 0:
            raise ValueError("Weights must not be empty")
        if any(not bool(isfinite(weight)) for weight in w):
            raise ValueError("Weights must be finite")
        if any(bool(weight < 0.0) for weight in w):
            raise ValueError("Weights must be nonnegative")

        mean_weight = mean(w)
        if not bool(mean_weight > 0.0):
            raise ValueError("Weights must have positive total mass")

        self.w = w / (mean_weight * 2.0 * pi)

    def pdf(self, xs):
        """Evaluate the pdf at each point in xs.

        Parameters
        ----------
        xs : array_like, shape (n,)
            Points at which to evaluate the pdf.

        Returns
        -------
        p : ndarray, shape (n,)
            Pdf values at each point.
        """
        _reject_complex_input(xs, "xs")
        xs = array(xs, dtype=float)
        if xs.ndim == 0:
            xs = xs.reshape((1,))
        if xs.ndim != 1:
            raise ValueError("xs must be a scalar or one-dimensional array")
        n_intervals = len(self.w)
        xs_mod = mod(xs, 2.0 * pi)
        idx = array(
            [
                min(int(floor(x / (2.0 * pi) * n_intervals)), n_intervals - 1)
                for x in xs_mod
            ]
        )
        return self.w[idx]

    def trigonometric_moment(self, n):
        """Calculate the n-th trigonometric moment analytically.

        Parameters
        ----------
        n : int
            Moment order.

        Returns
        -------
        m : complex
            n-th trigonometric moment.
        """
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError(
                "trigonometric_moment is not supported on the JAX backend."
            )
        n = _validate_moment_order(n)
        if n == 0:
            return 1.0 + 0j
        num = len(self.w)
        interv = zeros(num, dtype=complex)
        for j in range(1, num + 1):
            left = PiecewiseConstantDistribution.left_border(j, num)
            r = PiecewiseConstantDistribution.right_border(j, num)
            c = PiecewiseConstantDistribution.interval_center(j, num)
            w_j = float(self.pdf(array([c]))[0])
            interv[j - 1] = w_j * (exp(1j * n * r) - exp(1j * n * left))
        return complex(-1j / n * sum(interv))

    def entropy(self):
        """Calculate the entropy analytically.

        Returns
        -------
        e : float
            Entropy of the distribution.
        """
        n = len(self.w)
        positive_weights = self.w > 0.0
        safe_weights = pyrecest.backend.where(positive_weights, self.w, 1.0)
        entropy_terms = pyrecest.backend.where(
            positive_weights, self.w * log(safe_weights), 0.0
        )
        return float(-2.0 * pi / n * sum(entropy_terms))

    def sample(self, n):
        """Draw n random samples from the distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        samples : ndarray, shape (n,)
            Samples in [0, 2*pi).
        """
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError("sample is not supported on the JAX backend.")
        n = _validate_positive_sample_count(n)
        num_intervals = len(self.w)
        interval_width = 2.0 * pi / num_intervals
        # Each interval has probability w[j] * interval_width, which sums to 1 by
        # construction. Divide by sum anyway to guard against floating-point drift.
        interval_probs = self.w * interval_width
        interval_probs /= interval_probs.sum()
        interval_indices = random.choice(
            arange(num_intervals), size=(n,), p=interval_probs
        )
        return (
            interval_indices * interval_width
            + random.uniform(size=(n,)) * interval_width
        )

    @staticmethod
    def left_border(m, n):
        """Left border of the m-th interval (1-indexed) for n total intervals.

        Parameters
        ----------
        m : int
            Interval index (1-indexed).
        n : int
            Total number of intervals.

        Returns
        -------
        float
            Left border of the m-th interval.
        """
        m, n = _validate_interval_index(m, n)
        return 2.0 * pi / n * (m - 1)

    @staticmethod
    def right_border(m, n):
        """Right border of the m-th interval (1-indexed) for n total intervals.

        Parameters
        ----------
        m : int
            Interval index (1-indexed).
        n : int
            Total number of intervals.

        Returns
        -------
        float
            Right border of the m-th interval.
        """
        m, n = _validate_interval_index(m, n)
        return 2.0 * pi / n * m

    @staticmethod
    def interval_center(m, n):
        """Center of the m-th interval (1-indexed) for n total intervals.

        Parameters
        ----------
        m : int
            Interval index (1-indexed).
        n : int
            Total number of intervals.

        Returns
        -------
        float
            Center of the m-th interval.
        """
        m, n = _validate_interval_index(m, n)
        return 2.0 * pi / n * (m - 0.5)

    @staticmethod
    def calculate_parameters_numerically(pdf_func, n):
        """Calculate weights by numerically integrating a given pdf over each interval.

        Parameters
        ----------
        pdf_func : callable
            Pdf of a circular density; accepts a 1-D array and returns a 1-D array.
        n : int
            Number of discretization intervals.

        Returns
        -------
        w : ndarray, shape (n,)
            Weights of the corresponding PiecewiseConstantDistribution.
        """
        from scipy.integrate import quad  # pylint: disable=import-outside-toplevel

        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError(
                "calculate_parameters_numerically is not supported on the JAX backend."
            )

        def _evaluate_pdf(x):
            pdf_values = array(pdf_func(array([x])), dtype=float)
            if pdf_values.shape == ():
                scalar_value = pdf_values
            else:
                flat_values = pdf_values.reshape(-1)
                if flat_values.shape[0] != 1:
                    raise ValueError(
                        "pdf_func must return a scalar or single value per integration point"
                    )
                scalar_value = flat_values[0]
            scalar_float = float(scalar_value)
            if not np.isfinite(scalar_float):
                raise ValueError("pdf_func must return finite values")
            return scalar_float

        n = _validate_positive_sample_count(n)
        w = zeros(n)
        for j in range(1, n + 1):
            left = PiecewiseConstantDistribution.left_border(j, n)
            r = PiecewiseConstantDistribution.right_border(j, n)
            w[j - 1] = quad(_evaluate_pdf, left, r)[0]
        return w
