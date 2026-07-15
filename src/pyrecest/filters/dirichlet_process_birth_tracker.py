"""Dirichlet-process birth augmentation for multi-Bernoulli tracking."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyrecest.distributions import GaussianDistribution

from .multi_bernoulli_tracker import BernoulliComponent, MultiBernoulliTracker


@dataclass
class DirichletProcessBirthAtom:
    """Finite active approximation to one DP birth atom.

    The atom lives in state space. Its count is the Chinese-restaurant sufficient
    statistic used by :class:`DirichletProcessBirthMultiBernoulliTracker` when an
    unassigned measurement can either reuse an existing birth region or instantiate
    a new region.
    """

    mean: Any
    covariance: Any
    count: float = 1.0

    def __post_init__(self):
        self.mean = _as_vector(self.mean, "mean")
        self.covariance = _as_square_matrix(
            self.covariance,
            "covariance",
            self.mean.shape[0],
        )
        self.count = float(self.count)
        if self.count <= 0.0:
            raise ValueError("count must be positive")

    def copy(self):
        """Return a deep copy of the birth atom."""
        return copy.deepcopy(self)

    def measurement_likelihood(
        self,
        measurement,
        measurement_matrix,
        measurement_covariance,
    ):
        """Evaluate the atom's linear-Gaussian predictive likelihood."""
        measurement = _as_vector(measurement, "measurement")
        measurement_matrix = np.asarray(measurement_matrix, dtype=float)
        measurement_covariance = _as_square_matrix(
            measurement_covariance,
            "measurement_covariance",
            measurement.shape[0],
        )
        predicted_measurement = measurement_matrix @ self.mean
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_covariance
        )
        return float(
            GaussianDistribution(predicted_measurement, innovation_covariance).pdf(
                measurement
            )
        )

    def update_from_measurement(
        self,
        measurement,
        measurement_matrix,
        measurement_covariance,
    ):
        """Kalman-update the atom with a measurement assigned to this birth region."""
        measurement = _as_vector(measurement, "measurement")
        measurement_matrix = np.asarray(measurement_matrix, dtype=float)
        measurement_covariance = _as_square_matrix(
            measurement_covariance,
            "measurement_covariance",
            measurement.shape[0],
        )
        predicted_measurement = measurement_matrix @ self.mean
        innovation = measurement - predicted_measurement
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_covariance
        )
        gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.mean = self.mean + gain @ innovation
        updated_covariance = (
            self.covariance - gain @ measurement_matrix @ self.covariance
        )
        self.covariance = 0.5 * (updated_covariance + updated_covariance.T)
        self.count += 1.0


class DirichletProcessBirthMultiBernoulliTracker(MultiBernoulliTracker):
    """Multi-Bernoulli tracker with a Dirichlet-process-style birth model.

    The outer recursion is the lightweight PyRecEst multi-Bernoulli approximation:
    labels, existence probabilities, missed detections, pruning, and nearest-neighbor
    association are inherited from :class:`MultiBernoulliTracker`. This subclass only
    replaces the creation of measurement-driven births. Unassigned measurements compete
    against clutter and either reuse an active DP birth atom or instantiate a new atom.
    """

    DEFAULT_TRACKER_PARAM = {
        **MultiBernoulliTracker.DEFAULT_TRACKER_PARAM,
        "dp_concentration": 1.0,
        "dp_birth_threshold": 1.0,
        "dp_birth_clutter_intensity": None,
        "dp_reuse_existing_atoms": True,
        "dp_birth_atom_survival_probability": 1.0,
        "dp_birth_atom_pruning_threshold": 1e-6,
        "maximum_number_of_birth_atoms": None,
    }

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        initial_prior=None,
        tracker_param=None,
        birth_components=None,
        birth_atoms=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        self.birth_atoms = []
        self.last_birth_diagnostics = []
        super().__init__(
            initial_prior=initial_prior,
            tracker_param=tracker_param,
            birth_components=birth_components,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )
        if birth_atoms is not None:
            self.birth_atoms = self._normalize_birth_atoms(birth_atoms)

    @staticmethod
    def _normalize_birth_atoms(birth_atoms):
        normalized_atoms = []
        for atom in birth_atoms:
            if isinstance(atom, DirichletProcessBirthAtom):
                normalized_atoms.append(atom.copy())
            elif isinstance(atom, tuple) and len(atom) == 2:
                normalized_atoms.append(DirichletProcessBirthAtom(atom[0], atom[1]))
            elif isinstance(atom, tuple) and len(atom) == 3:
                normalized_atoms.append(
                    DirichletProcessBirthAtom(atom[0], atom[1], atom[2])
                )
            else:
                raise ValueError(
                    "birth_atoms must contain atoms or (mean, covariance[, count]) tuples"
                )
        return normalized_atoms

    def predict_linear(
        self,
        system_matrices,
        sys_noises,
        inputs=None,
        birth_components=None,
    ):
        """Predict targets and decay DP birth-atom counts."""
        super().predict_linear(
            system_matrices,
            sys_noises,
            inputs=inputs,
            birth_components=birth_components,
        )
        survival_probability = float(
            self.tracker_param["dp_birth_atom_survival_probability"]
        )
        if not 0.0 <= survival_probability <= 1.0:
            raise ValueError("dp_birth_atom_survival_probability must be in [0, 1]")
        for atom in self.birth_atoms:
            atom.count *= survival_probability
        self._prune_and_cap_birth_atoms()

    def update_linear(self, measurements, measurement_matrix, cov_mats_meas):
        """Update targets and reset DP-birth diagnostics for this scan."""
        self.last_birth_diagnostics = []
        return super().update_linear(measurements, measurement_matrix, cov_mats_meas)

    def get_birth_atoms(self, copy_atoms=True):
        """Return the active DP birth atoms."""
        if copy_atoms:
            return [atom.copy() for atom in self.birth_atoms]
        return self.birth_atoms

    def _create_birth_component_from_measurement(
        self,
        measurement,
        measurement_matrix,
        measurement_covariance,
        label=None,
    ):
        birth_covariance = self._resolve_birth_covariance(
            measurement,
            measurement_matrix,
            measurement_covariance,
        )
        if birth_covariance is None:
            return None

        measurement = _as_vector(measurement, "measurement")
        measurement_matrix = np.asarray(measurement_matrix, dtype=float)
        measurement_covariance = _as_square_matrix(
            measurement_covariance,
            "measurement_covariance",
            measurement.shape[0],
        )
        birth_mean = self._get_state_birth_mean(
            measurement,
            measurement_matrix,
            self.tracker_param["measurement_to_state_matrix"],
        )
        birth_mean = _as_vector(birth_mean, "birth_mean")
        birth_covariance = _as_square_matrix(
            birth_covariance,
            "birth_covariance",
            birth_mean.shape[0],
        )

        decision = self._select_birth_decision(
            measurement,
            measurement_matrix,
            measurement_covariance,
            birth_mean,
            birth_covariance,
        )
        self.last_birth_diagnostics.append(decision)
        if decision["action"] == "clutter":
            return None

        if decision["action"] == "new_atom":
            atom = DirichletProcessBirthAtom(birth_mean, birth_covariance)
            self.birth_atoms.append(atom)
            decision["atom_index"] = len(self.birth_atoms) - 1
        else:
            atom = self.birth_atoms[decision["atom_index"]]
            atom.update_from_measurement(
                measurement,
                measurement_matrix,
                measurement_covariance,
            )

        self._prune_and_cap_birth_atoms()
        return BernoulliComponent(
            decision["existence_probability"],
            GaussianDistribution(atom.mean, atom.covariance),
            label=label,
        )

    def _resolve_birth_covariance(
        self,
        measurement,
        measurement_matrix,
        measurement_covariance,
    ):
        birth_covariance = self.tracker_param["birth_covariance"]
        if birth_covariance is None:
            return None
        if callable(birth_covariance):
            birth_covariance = birth_covariance(
                measurement,
                measurement_matrix,
                measurement_covariance,
            )
        return birth_covariance

    def _select_birth_decision(
        self,
        measurement,
        measurement_matrix,
        measurement_covariance,
        birth_mean,
        birth_covariance,
    ):
        concentration = float(self.tracker_param["dp_concentration"])
        if concentration <= 0.0:
            raise ValueError("dp_concentration must be positive")

        base_likelihood = _linear_gaussian_likelihood(
            measurement,
            measurement_matrix,
            birth_mean,
            birth_covariance,
            measurement_covariance,
        )
        best_action = "new_atom"
        best_atom_index = None
        best_score = concentration * base_likelihood

        if self.tracker_param["dp_reuse_existing_atoms"]:
            for atom_index, atom in enumerate(self.birth_atoms):
                atom_likelihood = atom.measurement_likelihood(
                    measurement,
                    measurement_matrix,
                    measurement_covariance,
                )
                atom_score = atom.count * atom_likelihood
                if atom_score > best_score:
                    best_score = atom_score
                    best_action = "existing_atom"
                    best_atom_index = atom_index

        clutter_intensity = self._get_birth_clutter_intensity()
        birth_cap = self._clip_probability(
            self.tracker_param["birth_existence_probability"]
        )
        birth_odds = birth_cap * best_score / max(clutter_intensity, 1e-300)
        birth_probability = min(birth_cap, birth_odds / (1.0 + birth_odds))
        if birth_odds < float(self.tracker_param["dp_birth_threshold"]):
            return {
                "action": "clutter",
                "atom_index": None,
                "score": best_score,
                "base_likelihood": base_likelihood,
                "birth_odds": birth_odds,
                "existence_probability": 0.0,
                "clutter_intensity": clutter_intensity,
            }

        return {
            "action": best_action,
            "atom_index": best_atom_index,
            "score": best_score,
            "base_likelihood": base_likelihood,
            "birth_odds": birth_odds,
            "existence_probability": birth_probability,
            "clutter_intensity": clutter_intensity,
        }

    def _get_birth_clutter_intensity(self):
        dp_birth_clutter_intensity = self.tracker_param["dp_birth_clutter_intensity"]
        if dp_birth_clutter_intensity is not None:
            return max(float(dp_birth_clutter_intensity), 1e-300)
        try:
            return max(
                self._get_clutter_intensity(self.tracker_param["clutter_intensity"]),
                1e-300,
            )
        except ValueError as exc:
            raise ValueError(
                "Set dp_birth_clutter_intensity when clutter_intensity is measurement-dependent."
            ) from exc

    def _prune_and_cap_birth_atoms(self):
        pruning_threshold = float(self.tracker_param["dp_birth_atom_pruning_threshold"])
        self.birth_atoms = [
            atom for atom in self.birth_atoms if atom.count >= pruning_threshold
        ]
        maximum_number_of_birth_atoms = self.tracker_param[
            "maximum_number_of_birth_atoms"
        ]
        if maximum_number_of_birth_atoms is None:
            return
        maximum_number_of_birth_atoms = int(maximum_number_of_birth_atoms)
        if maximum_number_of_birth_atoms < 0:
            raise ValueError(
                "maximum_number_of_birth_atoms must be non-negative or None"
            )
        self.birth_atoms = sorted(
            self.birth_atoms,
            key=lambda atom: atom.count,
            reverse=True,
        )[:maximum_number_of_birth_atoms]


DPBirthMultiBernoulliTracker = DirichletProcessBirthMultiBernoulliTracker
DPBirthAtom = DirichletProcessBirthAtom


def _as_vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.ndim == 0:
        vector = vector.reshape(1)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return vector


def _as_square_matrix(value, name, dim):
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim == 0:
        matrix = np.eye(dim) * float(matrix)
    if matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    return matrix


def _linear_gaussian_likelihood(
    measurement,
    measurement_matrix,
    mean,
    covariance,
    measurement_covariance,
):
    predicted_measurement = measurement_matrix @ mean
    innovation_covariance = (
        measurement_matrix @ covariance @ measurement_matrix.T + measurement_covariance
    )
    return float(
        GaussianDistribution(predicted_measurement, innovation_covariance).pdf(
            measurement
        )
    )
