import copy
import warnings
from collections.abc import Callable
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    all,
    apply_along_axis,
    int32,
    int64,
    isclose,
    log,
    ones,
    random,
    sum,
)
from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType


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
        self.d = copy.copy(d)
        if w is None:
            self.w = ones(d.shape[0]) / d.shape[0]
        else:
            assert d.shape[0] == w.shape[0], "Number of Diracs and weights must match."
            self.w = copy.copy(w)
        self.normalize_in_place()

    def normalize_in_place(self):
        """
        Normalize the weights in-place to ensure they sum to 1.
        """
        if not isclose(sum(self.w), 1.0, atol=1e-10):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
            self.w = self.w / sum(self.w)

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
            dist.d = apply_along_axis(f, 1, dist.d)
        return dist

    def reweigh(self, f: Callable) -> "AbstractDiracDistribution":
        dist = copy.deepcopy(self)
        w_new = f(dist.d)

        assert w_new.shape == dist.w.shape, "Function returned wrong output dimensions."
        assert all(w_new >= 0), "All weights should be greater than or equal to 0."
        assert sum(w_new) > 0, "The sum of all weights should be greater than 0."

        dist.w = w_new * dist.w
        dist.w = dist.w / sum(dist.w)

        return dist

    def sample(self, n: Union[int, int32, int64]):
        samples = random.choice(self.d, n, p=self.w)
        return samples

    def entropy(self) -> float:
        warnings.warn("Entropy is not defined in a continuous sense")
        return -sum(self.w * log(self.w))

    def integrate(self, left=None, right=None):
        assert (
            left is None and right is None
        ), "Must overwrite in child class to use integral limits"
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
        highest_val, ind = max(self.w)
        if (highest_val / self.w.size) < (1 + rel_tol):
            warnings.warn(
                "The samples may be equally weighted, .mode is likely to return a bad result."
            )
        return self.d[ind, :]

    def mode_numerical(self, _=None):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def entropy_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    @classmethod
    def is_valid_for_conversion(cls, distribution):
        return any(isinstance(distribution, base) for base in cls.__bases__)

    @classmethod
    def from_distribution(cls, distribution, n_particles):
        assert cls.is_valid_for_conversion(distribution)
        samples = distribution.sample(n_particles)
        return cls(samples)
