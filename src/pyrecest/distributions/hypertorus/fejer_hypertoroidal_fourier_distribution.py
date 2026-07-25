"""Positive-kernel-reduced identity Fourier distributions on hypertori."""

from __future__ import annotations

import copy
import warnings
from typing import Any

import numpy as np
from pyrecest.backend import signal
from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)

from .fejer import (
    adaptive_kernel_reduce_coefficients,
    centered_coefficients,
    normalize_coefficient_shape,
    normalize_kernel_name,
    reduce_coefficients,
)


class FejerHypertoroidalFourierDistribution(HypertoroidalFourierDistribution):
    """Identity Fourier distribution with positive-kernel coefficient reduction.

    The class intentionally supports only the ``"identity"`` transformation.
    Multiplication of two identity Fourier distributions increases coefficient
    support. This subclass can either reduce the support unconditionally with a
    positive kernel such as Fejer or Fejer-Korovkin, or first try the ordinary
    sharp identity reduction and only damp coefficients if grid negativity is
    detected.
    """

    def __init__(
        self,
        coeff_mat,
        *,
        reduction_kernel: str = "fejer",
        adaptive_reduction: bool = False,
        min_value_tolerance: float = 1e-12,
        oversampling_factor: int = 1,
        exponent_search_steps: int = 24,
    ):
        self.reduction_kernel = normalize_kernel_name(reduction_kernel)
        self.adaptive_reduction = bool(adaptive_reduction)
        self.min_value_tolerance = float(min_value_tolerance)
        self.oversampling_factor = int(oversampling_factor)
        self.exponent_search_steps = int(exponent_search_steps)
        self.last_reduction_exponent: float | None = None
        self.last_used_prior_smoothing_fallback = False
        super().__init__(coeff_mat, transformation="identity")

    @property
    def reduction_options(self) -> dict[str, Any]:
        """Return constructor keyword arguments preserving reduction settings."""

        return {
            "reduction_kernel": self.reduction_kernel,
            "adaptive_reduction": self.adaptive_reduction,
            "min_value_tolerance": self.min_value_tolerance,
            "oversampling_factor": self.oversampling_factor,
            "exponent_search_steps": self.exponent_search_steps,
        }

    def _new_with_same_reduction(
        self, coeff_mat, *, reduction_exponent: float | None = None
    ) -> "FejerHypertoroidalFourierDistribution":
        result = type(self)(coeff_mat, **self.reduction_options)
        result.last_reduction_exponent = reduction_exponent
        result.last_used_prior_smoothing_fallback = (
            self.last_used_prior_smoothing_fallback
        )
        return result

    @classmethod
    def from_fourier_distribution(
        cls,
        distribution: HypertoroidalFourierDistribution,
        n_coefficients: int | tuple[int, ...] | None = None,
        *,
        apply_fejer: bool = True,
        reduction_kernel: str = "fejer",
        adaptive_reduction: bool = False,
        min_value_tolerance: float = 1e-12,
        oversampling_factor: int = 1,
        exponent_search_steps: int = 24,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Convert an identity HFD to a positive-kernel identity HFD."""

        if not isinstance(distribution, HypertoroidalFourierDistribution):
            raise TypeError("distribution must be a HypertoroidalFourierDistribution.")
        if distribution.transformation != "identity":
            raise ValueError(
                "FejerHypertoroidalFourierDistribution requires identity coefficients."
            )

        if n_coefficients is None:
            n_coefficients = distribution.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(
            n_coefficients, dim=distribution.dim
        )

        result = cls(
            centered_coefficients(distribution.coeff_mat, n_coefficients),
            reduction_kernel=reduction_kernel,
            adaptive_reduction=adaptive_reduction,
            min_value_tolerance=min_value_tolerance,
            oversampling_factor=oversampling_factor,
            exponent_search_steps=exponent_search_steps,
        )
        if apply_fejer:
            return result.fejer_reduce(n_coefficients)
        return result

    @classmethod
    def from_distribution(
        cls,
        distribution: AbstractHypertoroidalDistribution,
        n_coefficients: int | tuple[int, ...],
        *,
        apply_fejer: bool = True,
        reduction_kernel: str = "fejer",
        adaptive_reduction: bool = False,
        min_value_tolerance: float = 1e-12,
        oversampling_factor: int = 1,
        exponent_search_steps: int = 24,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Approximate a hypertoroidal distribution in positive-kernel identity form."""

        if isinstance(distribution, HypertoroidalFourierDistribution):
            return cls.from_fourier_distribution(
                distribution,
                n_coefficients,
                apply_fejer=apply_fejer,
                reduction_kernel=reduction_kernel,
                adaptive_reduction=adaptive_reduction,
                min_value_tolerance=min_value_tolerance,
                oversampling_factor=oversampling_factor,
                exponent_search_steps=exponent_search_steps,
            )
        if not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise TypeError(
                "distribution must be an AbstractHypertoroidalDistribution."
            )

        base = HypertoroidalFourierDistribution.from_distribution(
            distribution, n_coefficients, "identity"
        )
        return cls.from_fourier_distribution(
            base,
            n_coefficients,
            apply_fejer=apply_fejer,
            reduction_kernel=reduction_kernel,
            adaptive_reduction=adaptive_reduction,
            min_value_tolerance=min_value_tolerance,
            oversampling_factor=oversampling_factor,
            exponent_search_steps=exponent_search_steps,
        )

    @classmethod
    def from_function(
        cls,
        fun,
        n_coefficients: int | tuple[int, ...],
        *,
        apply_fejer: bool = True,
        reduction_kernel: str = "fejer",
        adaptive_reduction: bool = False,
        min_value_tolerance: float = 1e-12,
        oversampling_factor: int = 1,
        exponent_search_steps: int = 24,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Construct a positive-kernel identity HFD by sampling a vectorized function."""

        base = HypertoroidalFourierDistribution.from_function(
            fun, n_coefficients, "identity"
        )
        return cls.from_fourier_distribution(
            base,
            n_coefficients,
            apply_fejer=apply_fejer,
            reduction_kernel=reduction_kernel,
            adaptive_reduction=adaptive_reduction,
            min_value_tolerance=min_value_tolerance,
            oversampling_factor=oversampling_factor,
            exponent_search_steps=exponent_search_steps,
        )

    @classmethod
    def from_function_values(
        cls,
        fvals,
        n_coefficients: int | tuple[int, ...] | None = None,
        *,
        already_transformed: bool = False,
        apply_fejer: bool = True,
        reduction_kernel: str = "fejer",
        adaptive_reduction: bool = False,
        min_value_tolerance: float = 1e-12,
        oversampling_factor: int = 1,
        exponent_search_steps: int = 24,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Construct a positive-kernel identity HFD from regular-grid values."""

        base = HypertoroidalFourierDistribution.from_function_values(
            fvals,
            n_coefficients=n_coefficients,
            desired_transformation="identity",
            already_transformed=already_transformed,
        )
        target_shape = (
            base.coeff_mat.shape if n_coefficients is None else n_coefficients
        )
        return cls.from_fourier_distribution(
            base,
            target_shape,
            apply_fejer=apply_fejer,
            reduction_kernel=reduction_kernel,
            adaptive_reduction=adaptive_reduction,
            min_value_tolerance=min_value_tolerance,
            oversampling_factor=oversampling_factor,
            exponent_search_steps=exponent_search_steps,
        )

    def fejer_reduce(
        self, n_coefficients: int | tuple[int, ...] | None = None
    ) -> "FejerHypertoroidalFourierDistribution":
        """Center-crop/pad and reduce coefficients with the configured kernel."""

        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)
        coeff = self._reduce_coefficients(self.coeff_mat, n_coefficients)
        return self._new_with_same_reduction(
            coeff, reduction_exponent=self.last_reduction_exponent
        )

    def truncate(
        self, n_coefficients: int | tuple[int, ...], force_normalization: bool = False
    ):
        """Return a distribution with the requested centered coefficient shape.

        If the shape changes, reduction is performed with the configured kernel
        instead of sharp truncation. If the shape is unchanged, only optional
        normalization is performed, mirroring PyRecEst's truncate semantics.
        """

        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)
        if tuple(self.coeff_mat.shape) == n_coefficients:
            result = copy.deepcopy(self)
            if force_normalization:
                result.normalize_in_place(warn_unnorm=False)
            return result
        return self.fejer_reduce(n_coefficients)

    def multiply(self, f2: HypertoroidalFourierDistribution, n_coefficients=None):
        """Pointwise multiplication followed by configured coefficient reduction."""

        self._validate_compatible_identity(f2, "multiply")
        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)

        conv = signal.fftconvolve(self.coeff_mat, f2.coeff_mat, mode="full")
        self.last_used_prior_smoothing_fallback = False
        if self.adaptive_reduction and not self._has_positive_update_evidence(conv):
            smoothed_prior = reduce_coefficients(
                self.coeff_mat,
                self.coeff_mat.shape,
                kernel=self.reduction_kernel,
                exponent=1.0,
            )
            retry_conv = signal.fftconvolve(smoothed_prior, f2.coeff_mat, mode="full")
            if self._has_positive_update_evidence(retry_conv):
                conv = retry_conv
                self.last_used_prior_smoothing_fallback = True
        coeff = self._reduce_coefficients(conv, n_coefficients)
        return self._new_with_same_reduction(
            coeff, reduction_exponent=self.last_reduction_exponent
        )

    def convolve(self, f2: HypertoroidalFourierDistribution, n_coefficients=None):
        """Topology-aware convolution for additive noise in identity form.

        For equal coefficient shapes this keeps the ordinary IFF Hadamard-product
        prediction. Positive-kernel reduction is only used to align differing
        coefficient shapes.
        """

        self._validate_compatible_identity(f2, "convolve")
        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)

        f1_aligned = (
            self
            if tuple(self.coeff_mat.shape) == n_coefficients
            else self.fejer_reduce(n_coefficients)
        )
        if tuple(f2.coeff_mat.shape) == n_coefficients:
            f2_aligned = f2
        else:
            f2_aligned = type(self).from_fourier_distribution(
                f2, n_coefficients, apply_fejer=True, **self.reduction_options
            )

        c_conv = (2.0 * np.pi) ** self.dim * f1_aligned.coeff_mat * f2_aligned.coeff_mat
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Normalization:notNormalized")
            return self._new_with_same_reduction(c_conv)

    def shift(self, shift_by):
        """Shift the distribution on the hypertorus and keep reduction settings."""

        shifted = super().shift(shift_by)
        return type(self).from_fourier_distribution(
            shifted,
            shifted.coeff_mat.shape,
            apply_fejer=False,
            **self.reduction_options,
        )

    def _reduce_coefficients(self, coefficients, n_coefficients: tuple[int, ...]):
        if self.adaptive_reduction:
            coeff, exponent = adaptive_kernel_reduce_coefficients(
                coefficients,
                n_coefficients,
                kernel=self.reduction_kernel,
                min_value_tolerance=self.min_value_tolerance,
                oversampling_factor=self.oversampling_factor,
                exponent_search_steps=self.exponent_search_steps,
                return_exponent=True,
            )
            self.last_reduction_exponent = exponent
            return coeff

        self.last_reduction_exponent = 1.0 if self.reduction_kernel != "sharp" else 0.0
        return reduce_coefficients(
            coefficients, n_coefficients, kernel=self.reduction_kernel
        )

    def _has_positive_update_evidence(self, coefficients) -> bool:
        coeff_arr = np.asarray(coefficients)
        center = coeff_arr[tuple(side_length // 2 for side_length in coeff_arr.shape)]
        if not np.isfinite(center.real) or not np.isfinite(center.imag):
            return False
        if abs(center.imag) > max(1e-10, 1e-6 * abs(center.real)):
            return False
        return bool(center.real > self.min_value_tolerance)

    def _validate_compatible_identity(self, other, operation: str) -> None:
        if not isinstance(other, HypertoroidalFourierDistribution):
            raise TypeError(
                f"{operation}: other must be a HypertoroidalFourierDistribution."
            )
        if self.dim != other.dim:
            raise ValueError(f"{operation}: dimensions must match.")
        if self.transformation != "identity" or other.transformation != "identity":
            raise ValueError(
                f"{operation}: both distributions must use the identity transformation."
            )
