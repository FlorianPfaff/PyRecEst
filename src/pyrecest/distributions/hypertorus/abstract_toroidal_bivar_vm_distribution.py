# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import all, array, cos, exp, isfinite, mod, pi

from .abstract_toroidal_distribution import AbstractToroidalDistribution


def _as_python_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def validate_toroidal_vm_parameters(mu, kappa, *, require_positive_kappa=False):
    """Validate shared bivariate von Mises location and concentration parameters."""
    mu = array(mu)
    kappa = array(kappa)
    if mu.shape != (2,):
        raise ValueError("mu must have shape (2,)")
    if kappa.shape != (2,):
        raise ValueError("kappa must have shape (2,)")
    if not _as_python_bool(all(isfinite(mu))):
        raise ValueError("mu must contain only finite values")
    if not _as_python_bool(all(isfinite(kappa))):
        raise ValueError("kappa must contain only finite values")
    if require_positive_kappa:
        if not _as_python_bool(all(kappa > 0.0)):
            raise ValueError("kappa entries must be positive")
    elif not _as_python_bool(all(kappa >= 0.0)):
        raise ValueError("kappa entries must be nonnegative")
    return mu, kappa


def validate_scalar_parameter(value, name):
    """Validate a finite scalar distribution parameter."""
    value = array(value)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    if not _as_python_bool(isfinite(value)):
        raise ValueError(f"{name} must be finite")
    return value


class AbstractToroidalBivarVMDistribution(AbstractToroidalDistribution):
    """Abstract base for bivariate von Mises distributions on the torus.

    Subclasses share the same ``pdf`` structure::

        C * exp(kappa1*cos(x1 - mu1) + kappa2*cos(x2 - mu2) + coupling_term)

    Subclasses with a stable log normalization constant may assign
    ``self._log_norm_const``. The base density then evaluates directly in the
    log domain, avoiding the ``0 * inf`` cancellation at high concentration.
    """

    def __init__(self, mu, kappa):
        AbstractToroidalDistribution.__init__(self)
        mu, kappa = validate_toroidal_vm_parameters(mu, kappa)
        self.mu = mod(mu, 2.0 * pi)
        self.kappa = kappa
        self._log_norm_const = None

    def _coupling_term(self, xs):
        """Return the distribution-specific coupling term for ``pdf``."""
        raise NotImplementedError

    def pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.dim:
            raise ValueError(
                f"xs must have trailing dimension {self.dim}, got {xs.shape}."
            )
        log_unnormalized = (
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self._coupling_term(xs)
        )
        if self._log_norm_const is not None:
            return exp(log_unnormalized - self._log_norm_const)
        return self.C * exp(log_unnormalized)
