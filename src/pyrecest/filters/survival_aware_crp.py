"""Survival-aware CRP-style association priors for multitarget tracking.

The classes in this module are deliberately small scoring utilities rather than
complete multi-object Bayes filters.  They expose a non-exchangeable,
CRP-inspired prior in which an existing track competes with birth and clutter
alternatives through survival, visibility, and measurement-compatibility
factors instead of through an exchangeable historical count alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _as_finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_nonnegative(value, name):
    value = _as_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


def _validate_positive(value, name):
    value = _as_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_probability(value, name):
    value = _as_finite_float(value, name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def _validate_nonnegative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


@dataclass(frozen=True)
class SurvivalAwareTrackEvidence:
    """Track evidence used by :class:`SurvivalAwareCRPAssociationPrior`.

    Parameters
    ----------
    mass : float, optional
        Effective historical evidence for the track.  This plays the role of the
        CRP table count, but the association prior can discount it through
        ``last_seen_steps`` and ``temporal_decay``.
    existence_probability : float, optional
        Predicted Bernoulli existence probability for the track.
    survival_probability : float, optional
        Probability that the track survives from the last prediction to the
        current association step.
    detection_probability : float, optional
        Detection probability conditioned on the target being visible.
    visibility_probability : float, optional
        Scene/sensor visibility probability.  This separates an actual missed
        target from a temporarily occluded target.
    kinematic_likelihood : float, optional
        Nonnegative kinematic compatibility between the measurement and the
        predicted track state, for example a Gaussian innovation likelihood or a
        gated score.
    appearance_likelihood : float, optional
        Nonnegative appearance or re-identification compatibility.
    last_seen_steps : int, optional
        Number of prediction/update steps since the target was last associated
        with a measurement.  Larger values reduce historical mass when the prior
        uses ``temporal_decay < 1``.
    """

    mass: float = 1.0
    existence_probability: float = 1.0
    survival_probability: float = 1.0
    detection_probability: float = 1.0
    visibility_probability: float = 1.0
    kinematic_likelihood: float = 1.0
    appearance_likelihood: float = 1.0
    last_seen_steps: int = 0

    def __post_init__(self):
        object.__setattr__(self, "mass", _validate_nonnegative(self.mass, "mass"))
        object.__setattr__(
            self,
            "existence_probability",
            _validate_probability(
                self.existence_probability,
                "existence_probability",
            ),
        )
        object.__setattr__(
            self,
            "survival_probability",
            _validate_probability(
                self.survival_probability,
                "survival_probability",
            ),
        )
        object.__setattr__(
            self,
            "detection_probability",
            _validate_probability(
                self.detection_probability,
                "detection_probability",
            ),
        )
        object.__setattr__(
            self,
            "visibility_probability",
            _validate_probability(
                self.visibility_probability,
                "visibility_probability",
            ),
        )
        object.__setattr__(
            self,
            "kinematic_likelihood",
            _validate_nonnegative(
                self.kinematic_likelihood,
                "kinematic_likelihood",
            ),
        )
        object.__setattr__(
            self,
            "appearance_likelihood",
            _validate_nonnegative(
                self.appearance_likelihood,
                "appearance_likelihood",
            ),
        )
        object.__setattr__(
            self,
            "last_seen_steps",
            _validate_nonnegative_integer(
                self.last_seen_steps,
                "last_seen_steps",
            ),
        )

    @property
    def visibility_aware_detection_probability(self):
        """Return ``p_D`` multiplied by the scene/sensor visibility probability."""

        return self.detection_probability * self.visibility_probability


@dataclass(frozen=True)
class SurvivalAwareAssociationProbabilities:
    """Normalized probabilities for existing-track, birth, and clutter outcomes."""

    existing_track_probabilities: tuple[float, ...]
    birth_probability: float
    clutter_probability: float = 0.0

    @property
    def as_tuple(self):
        """Return existing-track probabilities followed by birth and clutter."""

        return self.existing_track_probabilities + (
            self.birth_probability,
            self.clutter_probability,
        )

    @property
    def total_probability(self):
        """Return the sum of all normalized probabilities."""

        return (
            sum(self.existing_track_probabilities)
            + self.birth_probability
            + self.clutter_probability
        )


@dataclass(frozen=True)
class SurvivalAwareCRPAssociationPrior:
    """Non-exchangeable CRP-inspired association prior for target tracking.

    The ordinary CRP/Pitman--Yor predictive rule assigns the next observation to
    an occupied table according to ``n_k - d`` and to a new table according to
    ``alpha + d K``.  This class keeps that count structure, but replaces the
    raw table count with a track-specific score:

    ``discounted mass * existence * survival * detection * visibility``
    ``* kinematic compatibility * appearance compatibility``.

    The resulting object is best viewed as a CRP-inspired partition prior, not
    as a classical Dirichlet process, because the track scores are explicitly
    non-exchangeable and depend on tracking context.
    """

    concentration: float = 1.0
    discount: float = 0.0
    temporal_decay: float = 1.0
    minimum_total_weight: float = 1e-300

    def __post_init__(self):
        discount = _as_finite_float(self.discount, "discount")
        concentration = _as_finite_float(self.concentration, "concentration")
        temporal_decay = _validate_probability(self.temporal_decay, "temporal_decay")
        minimum_total_weight = _validate_positive(
            self.minimum_total_weight,
            "minimum_total_weight",
        )

        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must satisfy 0 <= discount < 1.")
        if concentration <= -discount:
            raise ValueError("concentration must satisfy concentration > -discount.")
        if discount == 0.0 and concentration <= 0.0:
            raise ValueError("concentration must be positive when discount is zero.")

        object.__setattr__(self, "concentration", concentration)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "temporal_decay", temporal_decay)
        object.__setattr__(self, "minimum_total_weight", minimum_total_weight)

    @staticmethod
    def _coerce_track_evidence(track_evidence):
        if isinstance(track_evidence, SurvivalAwareTrackEvidence):
            return track_evidence
        if isinstance(track_evidence, dict):
            return SurvivalAwareTrackEvidence(**track_evidence)
        raise TypeError(
            "track_evidence entries must be SurvivalAwareTrackEvidence instances "
            "or dictionaries accepted by SurvivalAwareTrackEvidence."
        )

    def effective_track_mass(self, track_evidence):
        """Return temporally discounted track mass before the Pitman--Yor discount."""

        track_evidence = self._coerce_track_evidence(track_evidence)
        return track_evidence.mass * (
            self.temporal_decay ** track_evidence.last_seen_steps
        )

    def existing_track_weight(self, track_evidence):
        """Return the unnormalized association weight for one existing track."""

        track_evidence = self._coerce_track_evidence(track_evidence)
        count_weight = max(
            self.effective_track_mass(track_evidence) - self.discount,
            0.0,
        )
        return (
            count_weight
            * track_evidence.existence_probability
            * track_evidence.survival_probability
            * track_evidence.visibility_aware_detection_probability
            * track_evidence.kinematic_likelihood
            * track_evidence.appearance_likelihood
        )

    def existing_track_weights(self, track_evidences):
        """Return unnormalized association weights for existing tracks."""

        return tuple(
            self.existing_track_weight(track_evidence)
            for track_evidence in track_evidences
        )

    def birth_weight(self, num_existing_tracks, base_birth_weight=1.0):
        """Return the unnormalized new-track weight.

        ``base_birth_weight`` can encode a spatial birth intensity or a
        measurement-specific birth likelihood.  The Pitman--Yor-style BNP part is
        ``concentration + discount * K``.
        """

        num_existing_tracks = _validate_nonnegative_integer(
            num_existing_tracks,
            "num_existing_tracks",
        )
        base_birth_weight = _validate_nonnegative(
            base_birth_weight,
            "base_birth_weight",
        )
        return base_birth_weight * (
            self.concentration + self.discount * num_existing_tracks
        )

    def predictive_assignment_weights(
        self,
        track_evidences,
        base_birth_weight=1.0,
        clutter_weight=0.0,
    ):
        """Return unnormalized weights for existing tracks, birth, and clutter."""

        existing_weights = self.existing_track_weights(tuple(track_evidences))
        birth_weight = self.birth_weight(
            len(existing_weights),
            base_birth_weight=base_birth_weight,
        )
        clutter_weight = _validate_nonnegative(clutter_weight, "clutter_weight")
        return existing_weights, birth_weight, clutter_weight

    def predictive_assignment_probabilities(
        self,
        track_evidences,
        base_birth_weight=1.0,
        clutter_weight=0.0,
    ):
        """Return normalized probabilities for an association decision."""

        existing_weights, birth_weight, clutter_weight = (
            self.predictive_assignment_weights(
                track_evidences,
                base_birth_weight=base_birth_weight,
                clutter_weight=clutter_weight,
            )
        )
        total_weight = sum(existing_weights) + birth_weight + clutter_weight
        if total_weight <= self.minimum_total_weight:
            raise ValueError("At least one association alternative must have positive weight.")

        return SurvivalAwareAssociationProbabilities(
            existing_track_probabilities=tuple(
                weight / total_weight for weight in existing_weights
            ),
            birth_probability=birth_weight / total_weight,
            clutter_probability=clutter_weight / total_weight,
        )

    @staticmethod
    def predict_existence_probability(
        existence_probability,
        survival_probability,
    ):
        """Return the prediction ``r^- = r p_S`` for a Bernoulli track."""

        existence_probability = _validate_probability(
            existence_probability,
            "existence_probability",
        )
        survival_probability = _validate_probability(
            survival_probability,
            "survival_probability",
        )
        return existence_probability * survival_probability

    @staticmethod
    def missed_detection_existence_probability(
        predicted_existence_probability,
        detection_probability,
        visibility_probability=1.0,
    ):
        """Return the visibility-aware Bernoulli missed-detection update.

        The update is

        ``r^+ = r^- (1 - p_D v) / (1 - r^- p_D v)``,

        where ``v`` is the probability that the target was visible to the sensor.
        Misses under occlusion therefore reduce existence less strongly than
        misses in clear visibility.
        """

        predicted_existence_probability = _validate_probability(
            predicted_existence_probability,
            "predicted_existence_probability",
        )
        detection_probability = _validate_probability(
            detection_probability,
            "detection_probability",
        )
        visibility_probability = _validate_probability(
            visibility_probability,
            "visibility_probability",
        )

        effective_detection_probability = (
            detection_probability * visibility_probability
        )
        numerator = predicted_existence_probability * (
            1.0 - effective_detection_probability
        )
        denominator = (
            1.0
            - predicted_existence_probability * effective_detection_probability
        )
        if denominator <= 0.0:
            return 0.0
        return numerator / denominator
