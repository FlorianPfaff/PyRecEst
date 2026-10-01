"""Fitzgerald-style cheap JPDA with normalized missed-detection weights."""

from math import log, log1p

import numpy as np
from scipy.special import logsumexp

from .joint_probabilistic_data_association_filter import (
    JointProbabilisticDataAssociationFilter,
)


class CheapJointProbabilisticDataAssociationFilter(
    JointProbabilisticDataAssociationFilter
):
    r"""Linear-Gaussian cheap JPDA, without joint-event enumeration.

    For gated measurement likelihoods ``L[i, j]`` and clutter intensities
    ``kappa[j]``, define ``w[i, j] = P_D * L[i, j] / ((1-P_D) * kappa[j])``.
    Writing ``r[i] = sum_j w[i, j]`` and ``c[j] = sum_i w[i, j]``, use

    .. math::

        \beta_{ij} = \frac{w_{ij}}{1+r_i+c_j-w_{ij}},\qquad
        \beta_{i0} = 1-\sum_j\beta_{ij}.

    This retains Fitzgerald's row/column competition approximation, but uses
    the complementary missed-detection mass, rather than the generally
    non-normalizing original choice ``1 / (1 + r[i])``. With scalar clutter,
    this is the likelihood-form formula with ``B=(1-P_D)*kappa/P_D``.
    It is an approximation, not exact JPDA or independent per-track PDA.

    Gating, Gaussian hypotheses, prediction, and moment-matched covariance
    updates are inherited from :class:`JointProbabilisticDataAssociationFilter`.
    All association sums are evaluated in log space. The association stage
    takes O(n_targets * n_meas) time and memory; no enumeration limit is used.
    Measurement/state dimensions are held fixed in this complexity statement.

    ``find_association_probabilities`` returns probabilities followed by a
    feasible, track-order greedy likelihood-ratio assignment (not joint MAP).
    This diagnostic is available as ``latest_greedy_association`` and does not
    affect the soft update. ``latest_map_association`` remains ``None`` because
    no joint MAP event is computed. Misses are encoded as ``-1``.

    Uses the same association parameters as JPDAF. ``max_enumerated_events``
    is accepted for configuration compatibility, but ignored. Only the NumPy
    backend and linear-Gaussian measurement models are supported. As in JPDAF,
    the miss prior is ``1-P_D``; gating does not introduce a separate ``P_G``.
    See ``docs/cheap-jpdaf.md`` for the normalization convention and limitations.
    """

    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        super().__init__(
            initial_prior,
            association_param,
            log_prior_estimates,
            log_posterior_estimates,
        )
        self.latest_greedy_association = None

    @staticmethod
    def _prepare_clutter_intensity(clutter_intensity, n_meas):
        clutter_intensity = (
            JointProbabilisticDataAssociationFilter._prepare_clutter_intensity(
                clutter_intensity, n_meas
            )
        )
        if not np.all(np.isfinite(clutter_intensity)):
            raise ValueError("clutter_intensity must be finite and strictly positive.")
        return clutter_intensity

    @staticmethod
    def _cheap_marginals(log_weights):
        """Compute normalized marginals from gated log detection-to-miss odds.

        ``-inf`` denotes a gated-out pair. The caller supplies a two-dimensional
        array containing only finite values or ``-inf``. Prefix/suffix log sums
        avoid subtracting a dominant pair from a column total, and do not share
        one global scaling factor across disconnected association components.
        """
        n_targets, n_meas = log_weights.shape
        probabilities = np.zeros((n_targets, n_meas + 1))
        if n_targets == 0:
            return probabilities
        if n_meas == 0:
            probabilities[:, 0] = 1.0
            return probabilities

        before = np.full_like(log_weights, -np.inf)
        after = np.full_like(log_weights, -np.inf)
        if n_targets > 1:
            before[1:] = np.logaddexp.accumulate(log_weights[:-1], axis=0)
            after[:-1] = np.logaddexp.accumulate(log_weights[:0:-1], axis=0)[::-1]
        log_competition = np.logaddexp(before, after)
        log_row_denominator = np.logaddexp(
            0.0, logsumexp(log_weights, axis=1, keepdims=True)
        )
        log_denominator = np.logaddexp(log_row_denominator, log_competition)
        probabilities[:, 1:] = np.exp(log_weights - log_denominator)

        # Algebraically 1 - sum(beta), but expressed as a sum of positive terms:
        # 1/(1+r) + sum_j [w_j/(1+r)] * [competition_j/(1+r+competition_j)].
        # This preserves small miss probabilities without cancellation.
        probabilities[:, 0] = np.exp(-log_row_denominator[:, 0]) + np.sum(
            np.exp(
                (log_weights - log_row_denominator)
                + (log_competition - log_denominator)
            ),
            axis=1,
        )
        return probabilities

    @staticmethod
    def _greedy_assignment(log_weights):
        """Return a feasible O(n_targets * n_meas) diagnostic, not joint MAP."""
        n_targets, n_meas = log_weights.shape
        assignment = np.full(n_targets, -1, dtype=int)
        available = np.ones(n_meas, dtype=bool)
        if n_meas == 0:
            return assignment
        for track_index, row in enumerate(log_weights):
            scores = np.where(available, row, -np.inf)
            measurement_index = int(np.argmax(scores))
            # A unit detection-to-miss likelihood ratio ties with a miss.
            if scores[measurement_index] > 0.0:
                assignment[track_index] = measurement_index
                available[measurement_index] = False
        return assignment

    def _compute_association_probabilities(
        self,
        log_likelihoods,
        eligible_measurements,
        detection_probability,
        clutter_intensity,
    ):
        """Replace the exact event solver, leaving Gaussian updates unchanged."""
        del eligible_measurements  # The gated log-likelihood matrix encodes these.
        if np.any(np.isnan(log_likelihoods)) or np.any(np.isposinf(log_likelihoods)):
            raise ValueError(
                "Gated log likelihoods must be finite or negative infinity."
            )
        log_weights = (
            log_likelihoods
            + log(detection_probability)
            - log1p(-detection_probability)
            - np.log(clutter_intensity)[None, :]
        )
        return self._cheap_marginals(log_weights), self._greedy_assignment(log_weights)

    def find_association_probabilities(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
        """Return normalized marginals and a greedy diagnostic, not joint MAP.

        Column zero is the missed-detection mass; remaining columns correspond
        to measurements. Empty banks retain JPDAF's legacy ``(0, 1)`` shape.
        """
        probabilities, greedy_assignment = super().find_association_probabilities(
            measurements,
            measurement_matrix,
            cov_mats_meas,
            warn_on_no_meas_for_track=warn_on_no_meas_for_track,
        )
        self.latest_association_probabilities = probabilities
        self.latest_greedy_association = greedy_assignment
        self.latest_map_association = None
        if probabilities.shape[0] == 0:
            self._latest_posterior_hypotheses = []
        return probabilities, greedy_assignment

    def find_association(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
        """Return the track-order greedy diagnostic, not a MAP joint event."""
        return super().find_association(
            measurements,
            measurement_matrix,
            cov_mats_meas,
            warn_on_no_meas_for_track=warn_on_no_meas_for_track,
        )


CheapJPDAF = CheapJointProbabilisticDataAssociationFilter
CJPDAF = CheapJointProbabilisticDataAssociationFilter

__all__ = ["CheapJointProbabilisticDataAssociationFilter", "CheapJPDAF", "CJPDAF"]
