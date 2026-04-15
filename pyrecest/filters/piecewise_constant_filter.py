import warnings

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, dot, floor, pi, zeros
from pyrecest.distributions import AbstractCircularDistribution
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin


class PiecewiseConstantFilter(AbstractFilter, CircularFilterMixin):
    """
    A filter based on a piecewise constant distribution on the circle.

    The state is represented as a PiecewiseConstantDistribution over L equal
    intervals of [0, 2*pi).

    References:
    - Gerhard Kurz, Florian Pfaff, Uwe D. Hanebeck,
      Discrete Recursive Bayesian Filtering on Intervals and the Unit Circle
      Proceedings of the 2016 IEEE International Conference on Multisensor Fusion
      and Integration for Intelligent Systems (MFI 2016),
      Baden-Baden, Germany, September 2016.
    """

    def __init__(self, n):
        """
        Initialize the filter with a uniform distribution over n intervals.

        Parameters
        ----------
        n : int
            Number of discretization intervals.
        """
        assert isinstance(n, int) and n > 0
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, PiecewiseConstantDistribution(zeros(n) + 1.0))

    @property
    def filter_state(self):
        """Expose the parent property so we can attach a setter to it."""
        return super().filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        """
        Set the filter state.  If new_state is not a PiecewiseConstantDistribution,
        it is converted by numerically integrating the pdf over each interval.
        """
        if isinstance(new_state, PiecewiseConstantDistribution):
            self._filter_state = new_state
        elif isinstance(new_state, AbstractCircularDistribution):
            warnings.warn(
                "new_state is not a PiecewiseConstantDistribution. "
                "Converting using numerical integration.",
                RuntimeWarning,
            )
            n = len(self.filter_state.w)
            w = PiecewiseConstantDistribution.calculate_parameters_numerically(
                new_state.pdf, n
            )
            self._filter_state = PiecewiseConstantDistribution(w)
        else:
            raise TypeError(
                "new_state must be an instance of AbstractCircularDistribution."
            )

    def predict(self, sys_matrix):
        """
        Perform prediction step based on a transition matrix.

        Parameters
        ----------
        sys_matrix : array_like, shape (L, L)
            System/transition matrix.  Entry (j, i) gives the probability of
            transitioning from interval i to interval j.
        """
        w_new = dot(array(sys_matrix), self.filter_state.w)
        self._filter_state = PiecewiseConstantDistribution(w_new)

    def update(self, meas_matrix, z):
        """
        Perform measurement update based on a measurement matrix.

        Parameters
        ----------
        meas_matrix : array_like, shape (Lw, L)
            Measurement matrix.  Row z_row gives the likelihoods for each state
            interval when the measurement falls in measurement interval z_row.
        z : scalar
            Measurement in [0, 2*pi).
        """
        meas_matrix = array(meas_matrix)
        assert meas_matrix.shape[1] == len(self.filter_state.w)
        lw = meas_matrix.shape[0]
        row = int(floor(z / (2.0 * pi) * lw)) % lw
        w_new = meas_matrix[row, :] * self.filter_state.w
        self._filter_state = PiecewiseConstantDistribution(w_new)

    def update_likelihood(self, likelihood, z):
        """
        Perform measurement update using a likelihood function.

        Parameters
        ----------
        likelihood : callable
            Function ``likelihood(z, x)`` returning f(z | x), where z is the
            measurement and x is the state value.  Maps Z x [0, 2*pi) ->
            [0, infinity).
        z : arbitrary
            Measurement.
        """
        if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
            raise NotImplementedError(
                "update_likelihood is not supported on the JAX backend."
            )

        from scipy.integrate import quad  # pylint: disable=import-outside-toplevel

        L = len(self.filter_state.w)
        tmp = zeros(L)
        for i in range(L):
            left = PiecewiseConstantDistribution.left_border(i + 1, L)
            right = PiecewiseConstantDistribution.right_border(i + 1, L)
            tmp[i] = quad(lambda x, _z=z: likelihood(_z, x), left, right)[0]
        self._filter_state = PiecewiseConstantDistribution(tmp * self.filter_state.w)

    def get_point_estimate(self):
        """Return the mean direction of the filter state."""
        return self.filter_state.mean_direction()

    @staticmethod
    def calculate_system_matrix_numerically(L, a, noise_distribution):
        """
        Obtain the system matrix by 2-D numerical integration from a system function.

        Parameters
        ----------
        L : int
            Number of discretization intervals.
        a : callable
            System function ``x_{k+1} = a(x_k, w_k)``.  Must accept scalar
            arguments and return a scalar.
        noise_distribution : AbstractCircularDistribution
            Distribution of the process noise, defined on [0, 2*pi).

        Returns
        -------
        A : ndarray, shape (L, L)
            System transition matrix.  Entry (j, i) is the probability of
            transitioning from state interval i to state interval j.
        """
        from scipy.integrate import nquad  # pylint: disable=import-outside-toplevel

        assert isinstance(L, int) and L > 0
        assert isinstance(noise_distribution, AbstractCircularDistribution)
        assert callable(a)

        A = zeros((L, L))
        for i in range(L):
            l1 = PiecewiseConstantDistribution.left_border(i + 1, L)
            r1 = PiecewiseConstantDistribution.right_border(i + 1, L)
            for j in range(L):
                l2 = PiecewiseConstantDistribution.left_border(j + 1, L)
                r2 = PiecewiseConstantDistribution.right_border(j + 1, L)

                def integrand(x, w, _l2=l2, _r2=r2):
                    ax = a(x, w)
                    in_interval = 1.0 if _l2 <= ax < _r2 else 0.0
                    return float(noise_distribution.pdf(array([w]))) * in_interval

                A[j, i] = (
                    nquad(integrand, [[l1, r1], [0.0, 2.0 * pi]])[0] * L / (2.0 * pi)
                )

        return A

    @staticmethod
    def calculate_measurement_matrix_numerically(L, l_meas, h, noise_distribution):
        """
        Obtain the measurement matrix by 2-D numerical integration from a
        measurement function.

        Parameters
        ----------
        L : int
            Number of discretization intervals for the state.
        l_meas : int
            Number of discretization intervals for the measurement.
        h : callable
            Measurement function ``z_k = h(x_k, v_k)``.  Must accept scalar
            arguments and return a scalar.
        noise_distribution : AbstractCircularDistribution
            Distribution of the measurement noise, defined on [0, 2*pi).

        Returns
        -------
        H : ndarray, shape (l_meas, L)
            Measurement matrix.  Entry (i, j) is the probability that the
            measurement falls in measurement interval i given that the state is
            in state interval j.
        """
        from scipy.integrate import nquad  # pylint: disable=import-outside-toplevel

        assert isinstance(L, int) and L > 0
        assert isinstance(l_meas, int) and l_meas > 0
        assert isinstance(noise_distribution, AbstractCircularDistribution)
        assert callable(h)

        H = zeros((l_meas, L))
        for i in range(l_meas):
            l1 = PiecewiseConstantDistribution.left_border(i + 1, l_meas)
            r1 = PiecewiseConstantDistribution.right_border(i + 1, l_meas)
            for j in range(L):
                l2 = PiecewiseConstantDistribution.left_border(j + 1, L)
                r2 = PiecewiseConstantDistribution.right_border(j + 1, L)

                def integrand(x, v, _l1=l1, _r1=r1):
                    hx = h(x, v)
                    in_interval = 1.0 if _l1 <= hx < _r1 else 0.0
                    return float(noise_distribution.pdf(array([v]))) * in_interval

                H[i, j] = (
                    nquad(integrand, [[l2, r2], [0.0, 2.0 * pi]])[0] * L / (2.0 * pi)
                )

        return H
