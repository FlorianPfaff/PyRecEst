# pylint: disable=no-name-in-module,no-member,redefined-builtin
import pyrecest.backend
from pyrecest.backend import (
    arange,
    array,
    exp,
    floor,
    log,
    mean,
    mod,
    pi,
    random,
    sum,
    zeros,
)

from .abstract_circular_distribution import AbstractCircularDistribution


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
        w = array(w, dtype=float).ravel()
        assert w.ndim == 1 and w.shape[0] > 0
        self.w = w / (mean(w) * 2.0 * pi)

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
        assert xs.ndim == 1
        n_intervals = len(self.w)
        xs_mod = array(mod(xs, 2.0 * pi), dtype=float)
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
        return float(-2.0 * pi / n * sum(self.w * log(self.w)))

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
        assert 1 <= m <= n
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
        assert 1 <= m <= n
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
        assert 1 <= m <= n
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

        assert n >= 1
        w = zeros(n)
        for j in range(1, n + 1):
            left = PiecewiseConstantDistribution.left_border(j, n)
            r = PiecewiseConstantDistribution.right_border(j, n)
            w[j - 1] = quad(lambda x: float(pdf_func(array([x]))), left, r)[0]
        return w
