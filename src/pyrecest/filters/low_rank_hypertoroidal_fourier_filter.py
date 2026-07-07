"""Low-rank hypertoroidal Fourier filter."""

from __future__ import annotations

import warnings

import numpy as np

import pyrecest.backend
from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)
from pyrecest.distributions.hypertorus.low_rank_hypertoroidal_fourier_distribution import (
    LowRankHypertoroidalFourierDistribution,
)

from .abstract_filter import AbstractFilter
from .manifold_mixins import HypertoroidalFilterMixin


class LowRankHypertoroidalFourierFilter(AbstractFilter, HypertoroidalFilterMixin):
    """Tensor-train low-rank variant of the hypertoroidal IFF.

    This first prototype intentionally supports only ``transformation='identity'``.
    The SqFF prediction step requires recovering square-root Fourier coefficients
    after prediction and is left for later work.
    """

    def __init__(self, n_coefficients, transformation="identity", *, max_rank=None, rtol=0.0):
        if pyrecest.backend.__backend_name__ != "numpy":  # pylint: disable=no-member
            raise NotImplementedError("LowRankHypertoroidalFourierFilter is NumPy-only.")
        if transformation != "identity":
            raise NotImplementedError(
                "The first low-rank prototype supports only transformation='identity'."
            )
        if isinstance(n_coefficients, int):
            n_coefficients = (n_coefficients,)
        n_coefficients = tuple(int(n) for n in n_coefficients)
        self.max_rank = max_rank
        self.rtol = rtol
        initial_state = LowRankHypertoroidalFourierDistribution.uniform(
            n_coefficients, transformation="identity"
        )
        HypertoroidalFilterMixin.__init__(self)
        AbstractFilter.__init__(self, initial_state)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if not isinstance(new_state, LowRankHypertoroidalFourierDistribution):
            if not isinstance(new_state, HypertoroidalFourierDistribution):
                warnings.warn(
                    "setState:nonFourier: converting state to dense Fourier first.",
                    RuntimeWarning,
                )
                new_state = HypertoroidalFourierDistribution.from_distribution(
                    new_state,
                    self._filter_state.coeff_shape,
                    self._filter_state.transformation,
                )
            new_state = LowRankHypertoroidalFourierDistribution.from_dense(
                new_state, max_rank=self.max_rank, rtol=self.rtol
            )
        if new_state.transformation != "identity":
            raise NotImplementedError("Only identity-transformed low-rank states are supported.")
        self._filter_state = new_state

    def _convert_noise(self, distribution):
        if isinstance(distribution, LowRankHypertoroidalFourierDistribution):
            return distribution
        if isinstance(distribution, HypertoroidalFourierDistribution):
            return LowRankHypertoroidalFourierDistribution.from_dense(
                distribution, max_rank=self.max_rank, rtol=self.rtol
            )
        if isinstance(distribution, AbstractHypertoroidalDistribution):
            warnings.warn(
                "automaticConversion: transforming distribution to low-rank Fourier form.",
                RuntimeWarning,
            )
            return LowRankHypertoroidalFourierDistribution.from_distribution(
                distribution,
                self._filter_state.coeff_shape,
                "identity",
                max_rank=self.max_rank,
                rtol=self.rtol,
            )
        raise TypeError("distribution must be hypertoroidal.")

    def predict_identity(self, d_sys):
        """Predict for x_next = x + w mod 2*pi."""

        d_sys = self._convert_noise(d_sys)
        self._filter_state = self._filter_state.convolve(
            d_sys, max_rank=self.max_rank, rtol=self.rtol
        )

    def update_identity(self, d_meas, z):
        """Update for z = x + v mod 2*pi."""

        d_meas = self._convert_noise(d_meas)
        z = np.asarray(z)
        if z.shape != (self._filter_state.dim,):
            raise ValueError(f"z must have shape ({self._filter_state.dim},), got {z.shape}")
        likelihood = d_meas.shift(z)
        self._filter_state = self._filter_state.multiply(
            likelihood, max_rank=self.max_rank, rtol=self.rtol
        )
