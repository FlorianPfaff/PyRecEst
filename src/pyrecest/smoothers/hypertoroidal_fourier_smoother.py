"""Backward-information smoothers for hypertoroidal Fourier distributions."""

from __future__ import annotations

import copy
import warnings

import pyrecest.backend
from pyrecest.backend import pi, sqrt, zeros
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)

from .abstract_smoother import AbstractSmoother


class HypertoroidalFourierSmoother(AbstractSmoother):
    """Fixed-interval backward-information smoother on hypertori.

    The smoother assumes the additive identity model

    ``x[t + 1] = x[t] + w[t]  mod 2*pi``.

    Given filtering posteriors ``p(x_t | z_1, ..., z_t)``, likelihoods
    ``p(z_t | x_t)``, and additive system-noise densities, it computes backward
    messages

    ``beta_t(x_t) = int p(x_{t+1} | x_t) p(z_{t+1} | x_{t+1}) beta_{t+1}(x_{t+1}) dx_{t+1}``

    and smoothed states proportional to ``filtered[t] * beta[t]``.

    Backward messages are likelihood-like functions. They are represented as
    normalized :class:`HypertoroidalFourierDistribution` instances for numerical
    convenience; their absolute scale is irrelevant for the final normalized
    smoothed densities.
    """

    def smooth(self, filtered_states, likelihoods, system_noises):
        """Run :meth:`smooth_identity`.

        This method implements the :class:`AbstractSmoother` interface.
        """

        return self.smooth_identity(filtered_states, likelihoods, system_noises)

    def smooth_identity(self, filtered_states, likelihoods, system_noises):
        """Smooth a sequence generated with additive identity dynamics.

        Parameters
        ----------
        filtered_states : sequence of HypertoroidalFourierDistribution
            Filtering posterior densities ``p(x_t | z_1, ..., z_t)``.
        likelihoods : sequence of HypertoroidalFourierDistribution
            Likelihood functions for the corresponding measurements. Only their
            relative scale matters; internally they are treated as normalized
            Fourier distributions.
        system_noises : HypertoroidalFourierDistribution or sequence
            Additive noise densities for transitions ``t -> t + 1``. A single
            distribution is reused for every transition.

        Returns
        -------
        smoothed_states : list of HypertoroidalFourierDistribution
            Smoothed posteriors ``p(x_t | z_1, ..., z_T)``.
        backward_messages : list of HypertoroidalFourierDistribution
            Backward information messages. Their scale is arbitrary.
        """

        self._check_backend()

        filtered_states = self._as_distribution_list(filtered_states, "filtered_states")
        likelihoods = self._as_distribution_list(likelihoods, "likelihoods")

        if len(likelihoods) != len(filtered_states):
            raise ValueError(
                "likelihoods must have the same length as filtered_states."
            )

        n_steps = len(filtered_states)
        reference = filtered_states[0]
        n_coefficients = reference.coeff_mat.shape

        filtered_states = [
            self._ensure_compatible(
                dist, reference, n_coefficients, f"filtered_states[{idx}]"
            )
            for idx, dist in enumerate(filtered_states)
        ]
        likelihoods = [
            self._ensure_compatible(
                dist, reference, n_coefficients, f"likelihoods[{idx}]"
            )
            for idx, dist in enumerate(likelihoods)
        ]
        system_noises = self._normalize_noise_sequence(
            system_noises,
            max(n_steps - 1, 0),
            reference,
            n_coefficients,
        )

        backward_messages: list[HypertoroidalFourierDistribution | None] = [
            None
        ] * n_steps
        smoothed_states: list[HypertoroidalFourierDistribution | None] = [
            None
        ] * n_steps

        backward_messages[-1] = self.constant_message_like(filtered_states[-1])
        smoothed_states[-1] = filtered_states[-1].multiply(
            backward_messages[-1], n_coefficients
        )

        for t in range(n_steps - 2, -1, -1):
            next_backward = backward_messages[t + 1]
            assert next_backward is not None

            future_message = likelihoods[t + 1].multiply(next_backward, n_coefficients)
            reversed_noise = self.reverse_frequencies(system_noises[t])
            backward_messages[t] = future_message.convolve(
                reversed_noise, n_coefficients
            )
            smoothed_states[t] = filtered_states[t].multiply(
                backward_messages[t], n_coefficients
            )

        return (
            [state for state in smoothed_states if state is not None],
            [message for message in backward_messages if message is not None],
        )

    @staticmethod
    def reverse_frequencies(
        distribution: HypertoroidalFourierDistribution,
    ) -> HypertoroidalFourierDistribution:
        """Return a distribution with coefficients indexed by negated frequencies.

        For identity coefficients this represents ``p(-x)``. For square-root
        coefficients it represents the square-root factor evaluated at ``-x``,
        which induces the reflected density after squaring.
        """

        if not isinstance(distribution, HypertoroidalFourierDistribution):
            raise TypeError("distribution must be a HypertoroidalFourierDistribution.")

        result = copy.deepcopy(distribution)
        result.coeff_mat = distribution.coeff_mat[
            (slice(None, None, -1),) * distribution.dim
        ]
        return result

    @staticmethod
    def constant_message_like(
        reference: HypertoroidalFourierDistribution,
    ) -> HypertoroidalFourierDistribution:
        """Return a constant backward message compatible with ``reference``."""

        if not isinstance(reference, HypertoroidalFourierDistribution):
            raise TypeError("reference must be a HypertoroidalFourierDistribution.")

        coeff_shape = reference.coeff_mat.shape
        dim = reference.dim
        center = tuple(s // 2 for s in coeff_shape)
        coeffs = zeros(coeff_shape, dtype=complex)

        if reference.transformation == "identity":
            coeffs[center] = 1.0 / (2.0 * pi) ** dim
        elif reference.transformation == "sqrt":
            coeffs[center] = 1.0 / sqrt((2.0 * pi) ** dim)
        else:
            raise ValueError(
                f"Unsupported transformation: {reference.transformation!r}"
            )

        return HypertoroidalFourierDistribution(coeffs, reference.transformation)

    @staticmethod
    def _check_backend() -> None:
        if pyrecest.backend.__backend_name__ in ("jax", "pytorch"):
            raise NotImplementedError(
                "HypertoroidalFourierSmoother is currently supported only for the NumPy/SciPy backend."
            )

    @staticmethod
    def _as_distribution_list(
        distributions, name: str
    ) -> list[HypertoroidalFourierDistribution]:
        if isinstance(distributions, HypertoroidalFourierDistribution):
            raise ValueError(f"{name} must be a sequence, not a single distribution.")
        try:
            distribution_list = list(distributions)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be a sequence of HypertoroidalFourierDistribution instances."
            ) from exc

        if len(distribution_list) == 0:
            raise ValueError(f"{name} must contain at least one distribution.")
        for idx, distribution in enumerate(distribution_list):
            if not isinstance(distribution, HypertoroidalFourierDistribution):
                raise TypeError(
                    f"{name}[{idx}] must be a HypertoroidalFourierDistribution."
                )
        return distribution_list

    @staticmethod
    def _ensure_compatible(
        distribution: HypertoroidalFourierDistribution,
        reference: HypertoroidalFourierDistribution,
        n_coefficients,
        name: str,
    ) -> HypertoroidalFourierDistribution:
        if distribution.dim != reference.dim:
            raise ValueError(
                f"{name} has dimension {distribution.dim}, expected {reference.dim}."
            )
        if distribution.transformation != reference.transformation:
            raise ValueError(
                f"{name} uses transformation {distribution.transformation!r}, expected {reference.transformation!r}."
            )
        if distribution.coeff_mat.shape != n_coefficients:
            warnings.warn(
                f"{name} has coefficient shape {distribution.coeff_mat.shape}; truncating/padding to {n_coefficients}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return distribution.truncate(n_coefficients)
        return distribution

    @classmethod
    def _normalize_noise_sequence(
        cls,
        system_noises,
        expected_length: int,
        reference: HypertoroidalFourierDistribution,
        n_coefficients,
    ) -> list[HypertoroidalFourierDistribution]:
        if expected_length == 0:
            return []

        if isinstance(system_noises, HypertoroidalFourierDistribution):
            noise_list = [system_noises] * expected_length
        else:
            try:
                noise_list = list(system_noises)
            except TypeError as exc:
                raise TypeError(
                    "system_noises must be a HypertoroidalFourierDistribution or a sequence of them."
                ) from exc

        if len(noise_list) != expected_length:
            raise ValueError(
                "system_noises must be a single distribution or contain one distribution per transition."
            )

        normalized = []
        for idx, noise in enumerate(noise_list):
            if not isinstance(noise, HypertoroidalFourierDistribution):
                raise TypeError(
                    f"system_noises[{idx}] must be a HypertoroidalFourierDistribution."
                )
            normalized.append(
                cls._ensure_compatible(
                    noise, reference, n_coefficients, f"system_noises[{idx}]"
                )
            )
        return normalized


HypertoroidalFourierBackwardInformationSmoother = HypertoroidalFourierSmoother
HFFSmoother = HypertoroidalFourierSmoother
