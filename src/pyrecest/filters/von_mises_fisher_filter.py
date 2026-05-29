# pylint: disable=no-name-in-module,no-member
from math import isfinite as python_isfinite

from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    allclose,
    array,
    asarray,
    isfinite,
    linalg,
    ndim,
)
from pyrecest.distributions import VonMisesFisherDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import HypersphericalFilterMixin


def _to_python_bool(value):
    """Convert scalar backend booleans to Python bools for validation."""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _to_python_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _validate_vmf_distribution(distribution, role):
    if not isinstance(distribution, VonMisesFisherDistribution):
        raise ValueError(f"{role} must be a VonMisesFisherDistribution.")

    mu = asarray(distribution.mu)
    if ndim(mu) != 1:
        raise ValueError(f"{role} mean direction must be a vector.")
    if mu.shape[0] < 2:
        raise ValueError(f"{role} mean direction must be at least two-dimensional.")
    if not _to_python_bool(backend_all(isfinite(mu))):
        raise ValueError(f"{role} mean direction must be finite.")
    if not _to_python_bool(allclose(linalg.norm(mu), 1.0)):
        raise ValueError(f"{role} mean direction must be normalized.")

    kappa = _to_python_float(distribution.kappa)
    if not python_isfinite(kappa):
        raise ValueError(f"{role} concentration must be finite.")
    if kappa < 0.0:
        raise ValueError(f"{role} concentration must be nonnegative.")


def _validate_compatible_vmf(distribution, reference, role):
    _validate_vmf_distribution(distribution, role)
    if distribution.input_dim != reference.input_dim:
        raise ValueError(f"{role} dimension must match the filter state dimension.")


def _validate_zonal_vmf(distribution, role):
    if not _to_python_bool(allclose(distribution.mu[-1], 1.0)):
        raise ValueError(
            f"{role} mean direction must be zonal with final coordinate 1."
        )


def _validate_vmf_measurement(z, input_dim):
    measurement = asarray(z).ravel()
    if measurement.shape != (input_dim,):
        raise ValueError(f"measurement z must have shape ({input_dim},).")
    if not _to_python_bool(backend_all(isfinite(measurement))):
        raise ValueError("measurement z must be finite.")
    if not _to_python_bool(allclose(linalg.norm(measurement), 1.0)):
        raise ValueError("measurement z must be a unit vector.")
    return measurement


class VonMisesFisherFilter(AbstractFilter, HypersphericalFilterMixin):
    """Filter based on the von Mises-Fisher distribution.

    References
    ----------
    Kurz, G., Gilitschenski, I., & Hanebeck, U. D. (2016). Unscented von
    Mises-Fisher Filtering. IEEE Signal Processing Letters.
    """

    def __init__(self):
        HypersphericalFilterMixin.__init__(self)
        AbstractFilter.__init__(
            self, VonMisesFisherDistribution(array([1.0, 0.0]), 1.0)
        )

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, filter_state):
        _validate_vmf_distribution(filter_state, "filter_state")
        self._filter_state = filter_state

    def set_state(self, state):
        """Set the filter state."""
        self.filter_state = state

    def get_estimate_mean(self):
        """Return the mean direction of the current filter state."""
        return self.filter_state.mean_direction()

    def predict_identity(self, sys_noise):
        """
        State prediction via mulitiplication. Provide zonal density for update
        Could add support for a rotation Q
        """
        _validate_compatible_vmf(sys_noise, self.filter_state, "system noise")
        _validate_zonal_vmf(sys_noise, "system noise")
        self.filter_state = self.filter_state.convolve(sys_noise)

    def update_identity(self, meas_noise, z):
        """
        State update via mulitiplication. Provide zonal density for update
        Could add support for a rotation Q
        """
        _validate_compatible_vmf(meas_noise, self.filter_state, "measurement noise")
        _validate_zonal_vmf(meas_noise, "measurement noise")
        z = _validate_vmf_measurement(z, self.filter_state.input_dim)
        shifted_meas_noise = VonMisesFisherDistribution(z, meas_noise.kappa)
        self.filter_state = self.filter_state.multiply(shifted_meas_noise)
