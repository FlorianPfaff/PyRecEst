import copy
import math
import warnings
from collections.abc import Callable
from operator import index as _operator_index
from typing import Union

import numpy as np
from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    all,
    apply_along_axis,
    arange,
    argmax,
    asarray,
)
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import (
    int32,
    int64,
    isclose,
    isfinite,
    log,
    ones,
    random,
    reshape,
    stack,
    sum,
    to_numpy,
    where,
)

from .abstract_distribution_type import AbstractDistributionType


def _validate_positive_sample_count(n) -> int:
    """Return ``n`` as a positive Python int after scalar-count validation."""
    message = "n must be a positive integer."
    if isinstance(n, bool):
        raise ValueError(message)

    ndim = getattr(n, "ndim", None)
    if ndim not in (None, 0):
        raise ValueError(message)

    dtype = getattr(n, "dtype", None)
    if getattr(dtype, "kind", None) == "b" or str(dtype) == "torch.bool":
        raise ValueError(message)

    try:
        count = _operator_index(n)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc

    if count <= 0:
        raise ValueError(message)
    return int(count)


class AbstractDiracDistribution(AbstractDistributionType):
    """
    This class represents an abstract base for Dirac distributions.
    """

    def __init__(self, d, w=None):
        """
        Initialize a Dirac distribution with given Dirac locations and weights.

        :param d: Dirac locations as a numpy array.
        :param w: Weights of Dirac locations as a numpy array. If not provided, defaults to uniform weights.
        """
        d = asarray(d)
        self.d = backend_copy(d)
        if w is None:
            self.w = ones(d.shape[0]) / d.shape[0]
        else:
            w = reshape(asarray(w), (-1,))
            if d.shape[0] != w.shape[0]:
                raise ValueError("Number of Diracs and weights must match.")
            self.w = backend_copy(w)
        self.normalize_in_place()

    @staticmethod
    def _validate_weights(w):
        """Validate Dirac weights and return stable normalization divisors."""
        if w.shape[0] == 0:
            raise ValueError("Dirac weights must have positive finite total mass.")

        if not bool(all(isfinite(w))):
            raise ValueError("Dirac weights must be finite.")

        if not bool(all(w >= 0)):
            raise ValueError("Dirac weights must be nonnegative.")

        try:
            total_weight = sum(w)
        except FloatingPointError:
            total_weight = None
        if (
            total_weight is not None
            and bool(isfinite(total_weight))
            and bool(total_weight > 0)
        ):
            return 1.0, total_weight

        # Validation already synchronizes backend scalars through ``bool`` above.
        # A host fallback also avoids backend reductions that flush subnormals or
        # lower division by the largest finite float through a zero reciprocal.
        host_weights = np.asarray(to_numpy(w))
        weight_scale = float(np.max(host_weights))
        if not math.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("Dirac weights must have positive finite total mass.")

        scaled_total_weight = float(np.sum(host_weights / weight_scale))
        if not math.isfinite(scaled_total_weight) or scaled_total_weight <= 0:
            raise ValueError("Dirac weights must have positive finite total mass.")

        normalization_root = math.sqrt(weight_scale) * math.sqrt(scaled_total_weight)
        if not math.isfinite(normalization_root) or normalization_root <= 0:
            raise ValueError("Dirac weights must have positive finite total mass.")

        return normalization_root, normalization_root

    @staticmethod
    def _normalized_weights(w):
        """Return validated weights normalized across all supported backends."""
        first_divisor, second_divisor = AbstractDiracDistribution._validate_weights(w)
        normalized_weights = (w / first_divisor) / second_divisor

        try:
            normalized_total = sum(normalized_weights)
        except FloatingPointError:
            normalized_total = None
        if (
            normalized_total is not None
            and bool(isfinite(normalized_total))
            and bool(normalized_total > 0)
        ):
            return normalized_weights

        # XLA flushes subnormal operands before arithmetic.  Normalize those rare
        # inputs on the host, then move the ordinary-sized probabilities back to
        # the active backend.
        host_weights = np.asarray(to_numpy(w))
        weight_scale = float(np.max(host_weights))
        scaled_weights = host_weights / weight_scale
        host_normalized_weights = scaled_weights / np.sum(scaled_weights)
        return asarray(host_normalized_weights)

    def normalize_in_place(self):
        """
        Normalize the weights in-place to ensure they sum to 1.
        """
        normalized_weights = self._normalized_weights(self.w)
        if not bool(all(isclose(self.w, normalized_weights, atol=1e-10))):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
        self.w = normalized_weights

    def normalize(self) -> "AbstractDiracDistribution":
        dist = copy.deepcopy(self)
        dist.normalize_in_place()
        return dist

    @beartype
    def apply_function(self, f: Callable, function_is_vectorized: bool = True):
        """
        Apply a function to the Dirac locations and return a new distribution.

        :param f: Function to apply.
        :returns: A new distribution with the function applied to the locations.
        """
        dist = copy.deepcopy(self)
        if function_is_vectorized:
            dist.d = f(dist.d)
        else:
            dist.d = stack([asarray(f(point)) for point in dist.d])
        return dist

    def reweigh(self, f: Callable) -> "AbstractDiracDistribution":
        dist = copy.deepcopy(self)
        w_new = asarray(f(dist.d))

        if w_new.shape != dist.w.shape:
            raise ValueError("Function returned wrong output dimensions.")
        self._validate_weights(w_new)

        dist.w = self._normalized_weights(w_new * dist.w)
        return dist

    def sample(self, n: Union[int, int32, int64]):
        n = _validate_positive_sample_count(n)
        indices = random.choice(arange(self.d.shape[0]), n, p=self.w)
        samples = self.d[indices]
        return samples

    def entropy(self) -> float:
        warnings.warn("Entropy is not defined in a continuous sense")
        safe_weights = where(self.w > 0, self.w, 1.0)
        return -sum(self.w * log(safe_weights))

    def integrate(self, left=None, right=None):
        if left is not None or right is not None:
            raise ValueError("Must overwrite in child class to use integral limits")
        return sum(self.w)

    def log_likelihood(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def pdf(self, _):
        raise NotImplementedError("PDF:UNDEFINED, pdf is not defined")

    def integrate_numerically(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def trigonometric_moment_numerical(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def sample_metropolis_hastings(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def squared_distance_numerical(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def kld_numerical(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def mode(self, rel_tol=0.001):
        ind = int(argmax(self.w))
        highest_val = float(self.w[ind])
        if highest_val * self.w.shape[0] < (1 + rel_tol):
            warnings.warn(
                "The samples may be equally weighted, .mode is likely to return a bad result."
            )
        return self.d[ind]

    def mode_numerical(self, _=None):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def entropy_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    @classmethod
    def is_valid_for_conversion(cls, distribution):
        return any(isinstance(distribution, base) for base in cls.__bases__)

    @classmethod
    def from_distribution(cls, distribution, n_particles):
        if not cls.is_valid_for_conversion(distribution):
            raise ValueError(
                "distribution is not valid for conversion to this Dirac type"
            )
        samples = distribution.sample(n_particles)
        return cls(samples)
