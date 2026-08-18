import copy
import math
import warnings

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, isfinite, mod, pi
from pyrecest.distributions import VonMisesDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin


def _to_python_bool(value):
    """Convert scalar backend booleans to Python bools for validation."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _to_python_float(value, name):
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be scalar.") from exc


def _has_invalid_real_numeric_value(value):
    """Return whether ``value`` is not an unambiguous real numeric input."""
    if isinstance(value, (bool, str, bytes, bytearray, complex)):
        return True

    dtype = getattr(value, "dtype", None)
    dtype_kind = getattr(dtype, "kind", None)
    if dtype_kind in {"b", "c", "S", "U"}:
        return True

    dtype_name = str(dtype).lower() if dtype is not None else ""
    if "bool" in dtype_name or "complex" in dtype_name:
        return True

    if isinstance(value, (list, tuple)):
        return any(_has_invalid_real_numeric_value(item) for item in value)
    return False


def _validate_von_mises_distribution(distribution, role):
    if not isinstance(distribution, VonMisesDistribution):
        raise ValueError(f"{role} must be a VonMisesDistribution.")

    mu = _to_python_float(distribution.mu, f"{role} mean")
    kappa = _to_python_float(distribution.kappa, f"{role} concentration")
    if not math.isfinite(mu):
        raise ValueError(f"{role} mean must be finite.")
    if not math.isfinite(kappa):
        raise ValueError(f"{role} concentration must be finite.")
    if kappa < 0.0:
        raise ValueError(f"{role} concentration must be nonnegative.")


def _validate_circular_measurement(z):
    if _has_invalid_real_numeric_value(z):
        raise ValueError("measurement z must be a real numeric scalar.")
    try:
        measurement = array(z)
    except (TypeError, ValueError) as exc:
        raise ValueError("measurement z must be a real numeric scalar.") from exc
    if measurement.shape not in ((), (1,)):
        raise ValueError("measurement z must be scalar.")
    measurement = measurement[0] if measurement.shape == (1,) else measurement
    if not _to_python_bool(isfinite(measurement)):
        raise ValueError("measurement z must be finite.")
    return measurement


class VonMisesFilter(AbstractFilter, CircularFilterMixin):
    """
    A filter based on the Von Mises distribution.

    References:
    - M. Azmani, S. Reboul, J.-B. Choquel, and M. Benjelloun, "A recursive
      fusion filter for angular data" in 2009 IEEE International Conference
      on Robotics and Biomimetics (ROBIO), Dec. 2009, pp. 882-887.
    - Gerhard Kurz, Igor Gilitschenski, Uwe D. Hanebeck,
      Recursive Bayesian Filtering in Circular State Spaces
      arXiv preprint: Systems and Control (cs.SY), January 2015.
    """

    def __init__(self):
        """
        Constructor
        """
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, VonMisesDistribution(0, 1))

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        _validate_von_mises_distribution(new_state, "filter_state")
        self._filter_state = copy.deepcopy(new_state)

    def predict_identity(self, vmSys: VonMisesDistribution):
        """
        Predicts assuming identity system model, i.e.,
        x(k+1) = x(k) + w(k) mod 2*pi,
        where w(k) is additive noise given by vmSys.

        Parameters:
        vmSys (VMDistribution) : distribution of additive noise
        """
        _validate_von_mises_distribution(vmSys, "system noise")
        self.filter_state = self.filter_state.convolve(vmSys)

    def update_identity(self, vmMeas: VonMisesDistribution, z=0.0):
        """
        Updates assuming identity measurement model, i.e.,
        z(k) = x(k) + v(k) mod 2*pi,
        where v(k) is additive noise given by vmMeas.

        Parameters:
        vmMeas (VMDistribution) : distribution of additive noise
        z : measurement in [0, 2pi)
        """
        _validate_von_mises_distribution(vmMeas, "measurement noise")
        z = _validate_circular_measurement(z)
        measurement_noise_mean = _to_python_float(vmMeas.mu, "measurement noise mean")
        if measurement_noise_mean != 0.0:
            warning_message = (
                "The measurement noise is not centered at 0.0. "
                "This may lead to unexpected results because the "
                "update is based on the difference between the "
                "measurement and the current state."
            )
            warnings.warn(warning_message)

        muWnew = mod(z - vmMeas.mu, 2.0 * pi)
        vmMeasShifted = VonMisesDistribution(muWnew, vmMeas.kappa)
        self.filter_state = self.filter_state.multiply(vmMeasShifted)
