import math
from collections.abc import Callable

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, log, max, maximum, min, minimum, mod, pi
from pyrecest.distributions import CircularDiracDistribution, WrappedNormalDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin

_PROGRESSIVE_TAU_MESSAGE = "tau must be a positive finite scalar"


def _normalize_progressive_tau(tau, default: float) -> float:
    """Normalize the progressive-update threshold without truthiness coercion."""
    if tau is None:
        return default
    if isinstance(tau, (bool, str, bytes, bytearray)):
        raise ValueError(_PROGRESSIVE_TAU_MESSAGE)
    try:
        tau_array = array(tau)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_PROGRESSIVE_TAU_MESSAGE) from exc
    if getattr(tau_array, "shape", ()) != ():
        raise ValueError(_PROGRESSIVE_TAU_MESSAGE)
    try:
        scalar = tau_array.item()
    except AttributeError:
        scalar = tau_array
    if isinstance(scalar, (bool, str, bytes, bytearray)):
        raise ValueError(_PROGRESSIVE_TAU_MESSAGE)
    try:
        tau_value = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_PROGRESSIVE_TAU_MESSAGE) from exc
    if not math.isfinite(tau_value) or tau_value <= 0.0:
        raise ValueError(_PROGRESSIVE_TAU_MESSAGE)
    return tau_value


class WrappedNormalFilter(AbstractFilter, CircularFilterMixin):
    """Filter based on the wrapped normal distribution.

    References
    ----------
    Kurz, G., Gilitschenski, I., & Hanebeck, U. D. (2013). Recursive
    Nonlinear Filtering for Angular Data Based on Circular Distributions.
    Proceedings of the 2013 American Control Conference.

    Kurz, G., Gilitschenski, I., & Hanebeck, U. D. (2015). Recursive
    Bayesian Filtering in Circular State Spaces. arXiv preprint.
    """

    def __init__(self, wn=None):
        """Initialize the filter."""
        if wn is None:
            wn = WrappedNormalDistribution(array(0.0), array(1.0))
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, wn)

    def predict_identity(self, wn_sys):
        """Predicts using an identity system model."""
        self.filter_state = self.filter_state.convolve(wn_sys)

    def update_identity(self, wn_meas, z):
        mu_w_new = mod(z - wn_meas.mu, 2.0 * pi)
        wn_meas_shifted = WrappedNormalDistribution(mu_w_new, wn_meas.sigma)
        self.filter_state = self.filter_state.multiply_vm_approximation(wn_meas_shifted)

    @staticmethod
    def _evaluate_likelihood_values(likelihood: Callable, z, points, *, power=1.0):
        """Evaluate a likelihood callback for each Dirac support point.

        Historically, wrapped-normal updates accepted callbacks with the scalar
        signature ``likelihood(z, x)``.  Some callbacks also support vectorized
        evaluation over all support points; keep that fast path when it returns
        one value per point, and otherwise fall back to pointwise evaluation.
        """
        try:
            values = array(likelihood(z, points) ** power)
            values_flat = values.reshape((-1,))
            try:
                point_count = points.shape[0]
            except (AttributeError, IndexError):
                point_count = None
            if point_count is None:
                return values_flat[0]
            if values_flat.shape == (point_count,):
                return values_flat
        except (RuntimeError, TypeError, ValueError):
            pass

        return array([likelihood(z, point) ** power for point in points])

    def update_nonlinear_particle(self, likelihood, z):
        n = 100
        samples = self.filter_state.sample(n)
        wd = CircularDiracDistribution(samples)
        wd_new = wd.reweigh(
            lambda points: self._evaluate_likelihood_values(likelihood, z, points)
        )
        self.filter_state = wd_new.to_wn()

    def update_nonlinear_progressive(
        self, likelihood: Callable, z: float, tau: float | None = None
    ):
        # pylint: disable=too-many-locals
        DEFAULT_TAU = 0.02
        MINIMUM_LAMBDA: float = 0.001
        tau = _normalize_progressive_tau(tau, DEFAULT_TAU)
        lambda_ = 1.0
        steps = 0

        while lambda_ > 0:
            wd = self.filter_state.to_dirac5()
            likelihood_vals = self._evaluate_likelihood_values(likelihood, z, wd.d)
            likelihood_vals_min = min(likelihood_vals)
            likelihood_vals_max = max(likelihood_vals)

            if likelihood_vals_max == 0:
                raise ValueError(
                    "Progressive update failed because likelihood is 0 everywhere"
                )
            if likelihood_vals_min == likelihood_vals_max and likelihood_vals_max > 0:
                return

            if likelihood_vals_min == likelihood_vals_max:
                current_lambda = lambda_
            else:
                w_min = min(wd.w)
                w_max = max(wd.w)

                if likelihood_vals_min == 0 or w_min == 0:
                    raise ZeroDivisionError("Cannot perform division by zero")

                current_lambda = minimum(
                    log(tau * w_max / w_min)
                    / log(likelihood_vals_min / likelihood_vals_max),
                    lambda_,
                )

                if current_lambda <= 0:
                    raise ValueError(
                        "Progressive update with given threshold impossible"
                    )

                current_lambda = maximum(current_lambda, MINIMUM_LAMBDA)
                current_lambda = minimum(current_lambda, lambda_)
            wd_new = wd.reweigh(
                lambda points, power=current_lambda: self._evaluate_likelihood_values(
                    likelihood, z, points, power=power
                )
            )
            self.filter_state = wd_new.to_wn()
            lambda_ = lambda_ - current_lambda
            steps += 1
