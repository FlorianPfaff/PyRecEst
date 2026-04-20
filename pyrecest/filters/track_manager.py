"""Generic explicit track lifecycle management for PyRecEst.

This module provides a reusable layer around single-target filters:

- :class:`TrackStatus` encodes track lifecycle state.
- :class:`Track` stores one filter instance plus bookkeeping.
- :class:`AssociationResult` stores the output of a data-association step.
- :class:`TrackManager` manages births, tentative/confirmed tracks, misses,
  and deletions.

The manager itself is intentionally agnostic to the application domain. Users
supply small callbacks for prediction, association, update, and initiation.
That keeps domain-specific measurement models, cost functions, and metadata out
of the library core.

A global-nearest-neighbor helper is included for cost-matrix-based association.
The helper uses SciPy's Hungarian implementation and therefore operates on
NumPy arrays.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from pyrecest.backend import empty, stack
from pyrecest.distributions import GaussianDistribution
from scipy.optimize import linear_sum_assignment

from .abstract_filter import AbstractFilter
from .abstract_multitarget_tracker import AbstractMultitargetTracker
from .kalman_filter import KalmanFilter

PredictorFn = Callable[..., None]
UpdaterFn = Callable[..., None]
InitiatorFn = Callable[..., AbstractFilter]
AssociatorFn = Callable[..., "AssociationResult"]
TrackPredicateFn = Callable[["Track"], bool]
TrackMetadataInitializerFn = Callable[..., Optional[Dict[str, Any]]]
CostMatrixBuilderFn = Callable[..., np.ndarray]


class TrackStatus(str, Enum):
    """Lifecycle status of a track."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


@dataclass
class Track:  # pylint: disable=too-many-instance-attributes
    """Container for one managed track."""

    track_id: int
    single_target_filter: AbstractFilter
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 1
    first_step: int = 0
    last_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def dim(self) -> int:
        """Return the state dimension of the underlying filter."""

        return self.single_target_filter.dim

    @property
    def filter_state(self):
        """Return the current filter state."""

        return self.single_target_filter.filter_state

    @property
    def is_alive(self) -> bool:
        """Return whether the track is still active."""

        return self.status != TrackStatus.DELETED

    @property
    def is_confirmed(self) -> bool:
        """Return whether the track is confirmed."""

        return self.status == TrackStatus.CONFIRMED

    def get_point_estimate(self):
        """Return the current point estimate of the track."""

        return self.single_target_filter.get_point_estimate()


@dataclass
class AssociationResult:
    """Output of a data-association step.

    All indices refer to the *active-track list* passed to the associator and
    the measurement list passed into :meth:`TrackManager.step`.
    """

    matches: List[Tuple[int, int]] = field(default_factory=list)
    unmatched_track_indices: Optional[List[int]] = None
    unmatched_measurement_indices: Optional[List[int]] = None
    cost_matrix: Optional[np.ndarray] = None


@dataclass
class TrackManagerStepResult:  # pylint: disable=too-many-instance-attributes
    """Summary of one :meth:`TrackManager.step` call."""

    step: int
    matches: List[Tuple[int, int]] = field(default_factory=list)
    missed_track_ids: List[int] = field(default_factory=list)
    born_track_ids: List[int] = field(default_factory=list)
    confirmed_track_ids: List[int] = field(default_factory=list)
    deleted_track_ids: List[int] = field(default_factory=list)
    unmatched_measurement_indices: List[int] = field(default_factory=list)
    association: Optional[AssociationResult] = None


class TrackManager(
    AbstractMultitargetTracker
):  # pylint: disable=too-many-instance-attributes
    """Explicit lifecycle manager around a bank of single-target filters.

    The manager does not assume a specific measurement modality or cost model.
    Instead it delegates problem-specific logic to small user-supplied hooks:

    ``predictor(track, **kwargs)``
        Advances the underlying filter state.
    ``associator(tracks, measurements, **kwargs) -> AssociationResult``
        Returns the chosen associations.
    ``updater(track, measurement, measurement_index=None, **kwargs)``
        Updates a matched track using one measurement.
    ``initiator(measurement, measurement_index=None, **kwargs) -> filter``
        Creates a new single-target filter from an unmatched measurement.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        predictor: Optional[PredictorFn] = None,
        updater: Optional[UpdaterFn] = None,
        initiator: Optional[InitiatorFn] = None,
        associator: Optional[AssociatorFn] = None,
        n_init: int = 2,
        max_misses: int = 1,
        allow_births: bool = True,
        confirm_condition: Optional[TrackPredicateFn] = None,
        delete_condition: Optional[TrackPredicateFn] = None,
        track_metadata_initializer: Optional[TrackMetadataInitializerFn] = None,
        extract_confirmed_only: bool = True,
        keep_history: bool = True,
        log_prior_estimates: bool = True,
        log_posterior_estimates: bool = True,
    ):
        super().__init__(
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )
        if n_init < 1:
            raise ValueError("n_init must be at least 1")
        if max_misses < 0:
            raise ValueError("max_misses must be non-negative")

        self.predictor = predictor
        self.updater = updater
        self.initiator = initiator
        self.associator = associator
        self.n_init = int(n_init)
        self.max_misses = int(max_misses)
        self.allow_births = bool(allow_births)
        self.confirm_condition = confirm_condition
        self.delete_condition = delete_condition
        self.track_metadata_initializer = track_metadata_initializer
        self.extract_confirmed_only = bool(extract_confirmed_only)
        self.keep_history = bool(keep_history)

        self.tracks: List[Track] = []
        self._next_track_id = 0
        self._current_step = -1

    @property
    def dim(self) -> int:
        """Return the state dimension of the first active track."""

        active_tracks = self.get_tracks(confirmed_only=False, include_deleted=False)
        if not active_tracks:
            raise ValueError(
                "Cannot provide state dimension when no active tracks exist."
            )
        return active_tracks[0].dim

    @property
    def filter_state(self):
        """Return copies of the filter states of the extracted tracks."""

        selected_tracks = self._select_tracks_for_output(None, include_deleted=False)
        return [copy.deepcopy(track.filter_state) for track in selected_tracks]

    @filter_state.setter
    def filter_state(self, new_state):
        self.tracks = []
        self._next_track_id = 0
        for filter_or_state in new_state:
            self.add_track(
                filter_or_state,
                step=self._current_step,
                status=TrackStatus.CONFIRMED,
            )
        if self.log_prior_estimates:
            self.store_prior_estimates()

    def get_tracks(
        self,
        confirmed_only: Optional[bool] = None,
        include_deleted: bool = False,
    ) -> List[Track]:
        """Return managed tracks matching the requested visibility flags."""

        if confirmed_only is None:
            confirmed_only = self.extract_confirmed_only

        selected_tracks = []
        for track in self.tracks:
            if not include_deleted and not track.is_alive:
                continue
            if confirmed_only and not track.is_confirmed:
                continue
            selected_tracks.append(track)
        return selected_tracks

    def get_number_of_targets(self, confirmed_only: Optional[bool] = None) -> int:
        """Return the number of extracted tracks."""

        return len(
            self.get_tracks(confirmed_only=confirmed_only, include_deleted=False)
        )

    def get_point_estimate(
        self,
        flatten_vector: bool = False,
        confirmed_only: Optional[bool] = None,
    ):
        """Return stacked point estimates of the extracted tracks."""

        selected_tracks = self.get_tracks(
            confirmed_only=confirmed_only,
            include_deleted=False,
        )
        if not selected_tracks:
            point_estimates = empty((0, 0))
        else:
            point_estimates = stack(
                [track.get_point_estimate() for track in selected_tracks],
                axis=1,
            )
        if flatten_vector:
            return point_estimates.flatten()
        return point_estimates

    def initialize_from_states(
        self,
        filters_or_states: Sequence[Any],
        step: int = 0,
        confirmed: bool = True,
        metadata_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> List[int]:
        """Create tracks directly from filters or filter states."""

        if metadata_list is not None and len(metadata_list) != len(filters_or_states):
            raise ValueError(
                "metadata_list must have the same length as filters_or_states"
            )

        track_ids = []
        for index, filter_or_state in enumerate(filters_or_states):
            metadata = None if metadata_list is None else metadata_list[index]
            track_ids.append(
                self.add_track(
                    filter_or_state,
                    step=step,
                    status=(
                        TrackStatus.CONFIRMED if confirmed else TrackStatus.TENTATIVE
                    ),
                    metadata=metadata,
                    history_event="initialized",
                )
            )
        if self.log_prior_estimates:
            self.store_prior_estimates()
        return track_ids

    def initialize_from_measurements(
        self,
        measurements: Sequence[Any],
        step: int = 0,
        confirmed: bool = False,
        **initiation_kwargs,
    ) -> List[int]:
        """Create tracks from measurements using ``self.initiator``."""

        if self.initiator is None:
            raise ValueError(
                "TrackManager.initialize_from_measurements requires an initiator"
            )

        track_ids = []
        for measurement_index, measurement in enumerate(measurements):
            track_ids.append(
                self._birth_track_from_measurement(
                    measurement,
                    step=step,
                    measurement_index=measurement_index,
                    status=(
                        TrackStatus.CONFIRMED if confirmed else TrackStatus.TENTATIVE
                    ),
                    **initiation_kwargs,
                )
            )
        if self.log_prior_estimates:
            self.store_prior_estimates()
        return track_ids

    def add_track(
        self,
        filter_or_state: Any,
        step: int = 0,
        status: TrackStatus = TrackStatus.TENTATIVE,
        metadata: Optional[Dict[str, Any]] = None,
        history_event: str = "born",
    ) -> int:
        """Add a new track and return its track id."""

        normalized_status = self._normalize_status(status)
        single_target_filter = self._normalize_filter(filter_or_state)
        track_id = self._next_track_id
        self._next_track_id += 1

        track = Track(
            track_id=track_id,
            single_target_filter=single_target_filter,
            status=normalized_status,
            hits=1,
            misses=0,
            age=1,
            first_step=int(step),
            last_step=int(step),
            metadata={} if metadata is None else copy.deepcopy(metadata),
        )
        self._record_history(track, int(step), history_event)
        self.tracks.append(track)
        return track_id

    def purge_deleted_tracks(self) -> int:
        """Physically remove deleted tracks and return the number removed."""

        original_count = len(self.tracks)
        self.tracks = [track for track in self.tracks if track.is_alive]
        return original_count - len(self.tracks)

    def step(
        self,
        measurements: Sequence[Any],
        step: Optional[int] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        association_kwargs: Optional[Dict[str, Any]] = None,
        update_kwargs: Optional[Dict[str, Any]] = None,
        initiation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TrackManagerStepResult:
        """Run one complete lifecycle step."""

        predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
        association_kwargs = (
            {} if association_kwargs is None else dict(association_kwargs)
        )
        update_kwargs = {} if update_kwargs is None else dict(update_kwargs)
        initiation_kwargs = {} if initiation_kwargs is None else dict(initiation_kwargs)

        if step is None:
            step = self._current_step + 1
        self._current_step = int(step)

        active_tracks = self.get_tracks(confirmed_only=False, include_deleted=False)
        for track in active_tracks:
            track.age += 1
            track.last_step = int(step)
            if self.predictor is not None:
                self.predictor(track, **predict_kwargs)
            self._record_history(track, int(step), "predicted")

        if self.log_prior_estimates:
            self.store_prior_estimates()

        association = self._associate(active_tracks, measurements, association_kwargs)
        result = TrackManagerStepResult(step=int(step), association=association)

        for track_index, measurement_index in association.matches:
            track = active_tracks[track_index]
            measurement = measurements[measurement_index]
            if self.updater is not None:
                self.updater(
                    track,
                    measurement,
                    measurement_index=measurement_index,
                    **update_kwargs,
                )
            track.hits += 1
            track.misses = 0
            if track.status == TrackStatus.TENTATIVE and self._should_confirm(track):
                track.status = TrackStatus.CONFIRMED
                result.confirmed_track_ids.append(track.track_id)
            self._record_history(
                track,
                int(step),
                "matched",
                measurement_index=int(measurement_index),
            )
            result.matches.append((track.track_id, int(measurement_index)))

        for track_index in association.unmatched_track_indices or []:
            track = active_tracks[track_index]
            track.misses += 1
            result.missed_track_ids.append(track.track_id)
            self._record_history(track, int(step), "missed")
            if self._should_delete(track):
                track.status = TrackStatus.DELETED
                result.deleted_track_ids.append(track.track_id)
                self._record_history(track, int(step), "deleted")

        result.unmatched_measurement_indices = list(
            association.unmatched_measurement_indices or []
        )

        if self.allow_births and self.initiator is not None:
            for measurement_index in result.unmatched_measurement_indices:
                measurement = measurements[measurement_index]
                born_track_id = self._birth_track_from_measurement(
                    measurement,
                    step=int(step),
                    measurement_index=int(measurement_index),
                    status=(
                        TrackStatus.CONFIRMED
                        if self.n_init <= 1
                        else TrackStatus.TENTATIVE
                    ),
                    **initiation_kwargs,
                )
                result.born_track_ids.append(born_track_id)

        if self.log_posterior_estimates:
            self.store_posterior_estimates()

        return result

    def _associate(
        self,
        active_tracks: Sequence[Track],
        measurements: Sequence[Any],
        association_kwargs: Dict[str, Any],
    ) -> AssociationResult:
        """Run association and normalize the result."""

        num_tracks = len(active_tracks)
        num_measurements = len(measurements)

        if num_tracks == 0:
            return AssociationResult(
                matches=[],
                unmatched_track_indices=[],
                unmatched_measurement_indices=list(range(num_measurements)),
                cost_matrix=None,
            )

        if num_measurements == 0:
            return AssociationResult(
                matches=[],
                unmatched_track_indices=list(range(num_tracks)),
                unmatched_measurement_indices=[],
                cost_matrix=None,
            )

        if self.associator is None:
            return AssociationResult(
                matches=[],
                unmatched_track_indices=list(range(num_tracks)),
                unmatched_measurement_indices=list(range(num_measurements)),
                cost_matrix=None,
            )

        association = self.associator(active_tracks, measurements, **association_kwargs)
        return self._normalize_association_result(
            association,
            num_tracks,
            num_measurements,
        )

    def _birth_track_from_measurement(
        self,
        measurement: Any,
        step: int,
        measurement_index: Optional[int] = None,
        status: TrackStatus = TrackStatus.TENTATIVE,
        **initiation_kwargs,
    ) -> int:
        if self.initiator is None:
            raise ValueError("TrackManager requires an initiator to create tracks")

        single_target_filter = self.initiator(
            measurement,
            measurement_index=measurement_index,
            **initiation_kwargs,
        )
        metadata: Optional[Dict[str, Any]] = None
        if self.track_metadata_initializer is not None:
            metadata = self.track_metadata_initializer(
                measurement,
                measurement_index=measurement_index,
                step=step,
                **initiation_kwargs,
            )
        return self.add_track(
            single_target_filter,
            step=step,
            status=status,
            metadata=metadata,
            history_event="born",
        )

    def _should_confirm(self, track: Track) -> bool:
        if self.confirm_condition is not None:
            return bool(self.confirm_condition(track))
        return track.hits >= self.n_init

    def _should_delete(self, track: Track) -> bool:
        if self.delete_condition is not None:
            return bool(self.delete_condition(track))
        return track.misses > self.max_misses

    def _select_tracks_for_output(
        self,
        confirmed_only: Optional[bool],
        include_deleted: bool,
    ) -> List[Track]:
        return self.get_tracks(
            confirmed_only=confirmed_only,
            include_deleted=include_deleted,
        )

    def _record_history(self, track: Track, step: int, event: str, **payload) -> None:
        if not self.keep_history:
            return
        event_record = {"step": int(step), "event": event}
        event_record.update(payload)
        track.history.append(event_record)

    @staticmethod
    def _normalize_status(status: TrackStatus) -> TrackStatus:
        if isinstance(status, TrackStatus):
            return status
        return TrackStatus(status)

    @staticmethod
    def _normalize_filter(filter_or_state: Any) -> AbstractFilter:
        if isinstance(filter_or_state, AbstractFilter):
            return copy.deepcopy(filter_or_state)
        if isinstance(filter_or_state, GaussianDistribution):
            return KalmanFilter(filter_or_state)
        if isinstance(filter_or_state, tuple) and len(filter_or_state) == 2:
            return KalmanFilter(filter_or_state)
        raise ValueError(
            "Expected an AbstractFilter, a GaussianDistribution, or a "
            "(mean, covariance) tuple as track initializer"
        )

    @staticmethod
    def _normalize_association_result(  # pylint: disable=too-many-branches
        association: AssociationResult,
        num_tracks: int,
        num_measurements: int,
    ) -> AssociationResult:
        if not isinstance(association, AssociationResult):
            raise ValueError("associator must return an AssociationResult")

        used_track_indices = set()
        used_measurement_indices = set()
        matches: List[Tuple[int, int]] = []

        for track_index, measurement_index in association.matches:
            track_index = int(track_index)
            measurement_index = int(measurement_index)
            if track_index < 0 or track_index >= num_tracks:
                raise ValueError("Track index in association is out of range")
            if measurement_index < 0 or measurement_index >= num_measurements:
                raise ValueError("Measurement index in association is out of range")
            if track_index in used_track_indices:
                raise ValueError("Each track may appear in at most one association")
            if measurement_index in used_measurement_indices:
                raise ValueError(
                    "Each measurement may appear in at most one association"
                )
            used_track_indices.add(track_index)
            used_measurement_indices.add(measurement_index)
            matches.append((track_index, measurement_index))

        if association.unmatched_track_indices is None:
            unmatched_track_indices = sorted(
                set(range(num_tracks)) - used_track_indices
            )
        else:
            unmatched_track_indices = sorted(
                {int(index) for index in association.unmatched_track_indices}
            )

        if association.unmatched_measurement_indices is None:
            unmatched_measurement_indices = sorted(
                set(range(num_measurements)) - used_measurement_indices
            )
        else:
            unmatched_measurement_indices = sorted(
                {int(index) for index in association.unmatched_measurement_indices}
            )

        if used_track_indices.intersection(unmatched_track_indices):
            raise ValueError("A track cannot be both matched and unmatched")
        if used_measurement_indices.intersection(unmatched_measurement_indices):
            raise ValueError("A measurement cannot be both matched and unmatched")

        if any(index < 0 or index >= num_tracks for index in unmatched_track_indices):
            raise ValueError("Unmatched track index is out of range")
        if any(
            index < 0 or index >= num_measurements
            for index in unmatched_measurement_indices
        ):
            raise ValueError("Unmatched measurement index is out of range")

        return AssociationResult(
            matches=matches,
            unmatched_track_indices=unmatched_track_indices,
            unmatched_measurement_indices=unmatched_measurement_indices,
            cost_matrix=association.cost_matrix,
        )


def solve_global_nearest_neighbor(  # pylint: disable=too-many-locals
    cost_matrix: np.ndarray,
    unassigned_track_cost: Any,
    unassigned_measurement_cost: Optional[Any] = None,
    invalid_cost: float = 1e12,
    dummy_dummy_cost: float = 0.0,
) -> AssociationResult:
    """Solve a global-nearest-neighbor assignment from a cost matrix.

    Parameters
    ----------
    cost_matrix:
        ``(n_tracks, n_measurements)`` matrix of assignment costs.
    unassigned_track_cost:
        Scalar or length-``n_tracks`` iterable specifying the cost of leaving a
        track unmatched.
    unassigned_measurement_cost:
        Scalar or length-``n_measurements`` iterable specifying the cost of
        leaving a measurement unmatched. If omitted, ``unassigned_track_cost``
        is reused.
    invalid_cost:
        Replacement for non-finite costs.
    dummy_dummy_cost:
        Cost placed in the dummy-dummy block. The default ``0.0`` matches the
        common rectangular-assignment interpretation. Some legacy trackers use a
        non-zero dummy-dummy cost to reproduce their historic gating semantics.
    """

    matrix = np.asarray(cost_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional")

    num_tracks, num_measurements = matrix.shape
    if unassigned_measurement_cost is None:
        unassigned_measurement_cost = unassigned_track_cost

    if num_tracks == 0:
        return AssociationResult(
            matches=[],
            unmatched_track_indices=[],
            unmatched_measurement_indices=list(range(num_measurements)),
            cost_matrix=matrix.copy(),
        )
    if num_measurements == 0:
        return AssociationResult(
            matches=[],
            unmatched_track_indices=list(range(num_tracks)),
            unmatched_measurement_indices=[],
            cost_matrix=matrix.copy(),
        )

    track_unassigned_costs = _coerce_cost_vector(
        unassigned_track_cost,
        num_tracks,
        name="unassigned_track_cost",
    )
    measurement_unassigned_costs = _coerce_cost_vector(
        unassigned_measurement_cost,
        num_measurements,
        name="unassigned_measurement_cost",
    )

    finite_matrix = matrix.copy()
    finite_matrix[~np.isfinite(finite_matrix)] = float(invalid_cost)

    augmented_cost = np.full(
        (num_tracks + num_measurements, num_measurements + num_tracks),
        float(invalid_cost),
    )
    augmented_cost[:num_tracks, :num_measurements] = finite_matrix

    for track_index in range(num_tracks):
        augmented_cost[track_index, num_measurements + track_index] = (
            track_unassigned_costs[track_index]
        )

    for measurement_index in range(num_measurements):
        augmented_cost[num_tracks + measurement_index, measurement_index] = (
            measurement_unassigned_costs[measurement_index]
        )

    augmented_cost[num_tracks:, num_measurements:] = float(dummy_dummy_cost)

    row_ind, col_ind = linear_sum_assignment(augmented_cost)

    matches: List[Tuple[int, int]] = []
    unmatched_track_indices: List[int] = []
    unmatched_measurement_indices: List[int] = []

    for row_index, col_index in zip(row_ind, col_ind):
        if row_index < num_tracks:
            if col_index < num_measurements:
                matches.append((int(row_index), int(col_index)))
            else:
                unmatched_track_indices.append(int(row_index))
        elif col_index < num_measurements:
            unmatched_measurement_indices.append(int(col_index))

    matched_track_indices = {track_index for track_index, _ in matches}
    matched_measurement_indices = {
        measurement_index for _, measurement_index in matches
    }

    unmatched_track_indices = sorted(
        set(unmatched_track_indices).union(
            set(range(num_tracks)) - matched_track_indices
        )
    )
    unmatched_measurement_indices = sorted(
        set(unmatched_measurement_indices).union(
            set(range(num_measurements)) - matched_measurement_indices
        )
    )

    return AssociationResult(
        matches=matches,
        unmatched_track_indices=unmatched_track_indices,
        unmatched_measurement_indices=unmatched_measurement_indices,
        cost_matrix=matrix.copy(),
    )


def build_global_nearest_neighbor_associator(
    cost_matrix_builder: CostMatrixBuilderFn,
    unassigned_track_cost: Any,
    unassigned_measurement_cost: Optional[Any] = None,
    invalid_cost: float = 1e12,
    dummy_dummy_cost: float = 0.0,
) -> AssociatorFn:
    """Create an associator from a cost-matrix builder.

    The resulting associator has the signature expected by
    :class:`TrackManager`.
    """

    def associator(tracks: Sequence[Track], measurements: Sequence[Any], **kwargs):
        cost_matrix = cost_matrix_builder(tracks, measurements, **kwargs)
        return solve_global_nearest_neighbor(
            cost_matrix,
            unassigned_track_cost=unassigned_track_cost,
            unassigned_measurement_cost=unassigned_measurement_cost,
            invalid_cost=invalid_cost,
            dummy_dummy_cost=dummy_dummy_cost,
        )

    return associator


def build_linear_gaussian_predictor(
    system_matrix: np.ndarray,
    sys_noise_cov: np.ndarray,
    sys_input: Optional[np.ndarray] = None,
) -> PredictorFn:
    """Create a linear/Gaussian prediction hook for filters supporting it."""

    system_matrix = np.asarray(system_matrix)
    sys_noise_cov = np.asarray(sys_noise_cov)
    sys_input_array = None if sys_input is None else np.asarray(sys_input)

    def predictor(track: Track, **kwargs) -> None:
        current_system_matrix = kwargs.get("system_matrix", system_matrix)
        current_sys_noise_cov = kwargs.get("sys_noise_cov", sys_noise_cov)
        current_sys_input = kwargs.get("sys_input", sys_input_array)
        cast(KalmanFilter, track.single_target_filter).predict_linear(
            current_system_matrix,
            current_sys_noise_cov,
            current_sys_input,
        )

    return predictor


def build_linear_gaussian_updater(
    measurement_matrix: np.ndarray,
    measurement_covariance: Any,
    measurement_getter: Optional[Callable[..., np.ndarray]] = None,
) -> UpdaterFn:
    """Create a linear/Gaussian update hook for filters supporting it."""

    measurement_matrix = np.asarray(measurement_matrix)

    if measurement_getter is None:

        def measurement_getter(measurement: Any, **kwargs) -> np.ndarray:
            del kwargs
            return np.asarray(measurement)

    def updater(
        track: Track,
        measurement: Any,
        measurement_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        measurement_vector = measurement_getter(
            measurement,
            measurement_index=measurement_index,
            track=track,
            **kwargs,
        )
        if callable(measurement_covariance):
            covariance = measurement_covariance(
                measurement,
                measurement_index=measurement_index,
                track=track,
                **kwargs,
            )
        else:
            covariance = measurement_covariance
        cast(KalmanFilter, track.single_target_filter).update_linear(
            np.asarray(measurement_vector),
            measurement_matrix,
            np.asarray(covariance),
        )

    return updater


def build_kalman_measurement_initiator(
    initial_covariance: Any,
    measurement_getter: Optional[Callable[..., np.ndarray]] = None,
    measurement_to_state_mapping: Optional[Any] = None,
) -> InitiatorFn:
    """Create a Kalman-filter initiator from measurements."""

    if measurement_getter is None:

        def measurement_getter(measurement: Any, **kwargs) -> np.ndarray:
            del kwargs
            return np.asarray(measurement)

    def initiator(
        measurement: Any,
        measurement_index: Optional[int] = None,
        **kwargs,
    ) -> AbstractFilter:
        measurement_vector = np.asarray(
            measurement_getter(
                measurement,
                measurement_index=measurement_index,
                **kwargs,
            )
        )

        if measurement_to_state_mapping is None:
            state_mean = measurement_vector
        elif callable(measurement_to_state_mapping):
            state_mean = measurement_to_state_mapping(
                measurement,
                measurement_index=measurement_index,
                **kwargs,
            )
        else:
            state_mean = np.asarray(measurement_to_state_mapping) @ measurement_vector

        if callable(initial_covariance):
            covariance = initial_covariance(
                measurement,
                measurement_index=measurement_index,
                **kwargs,
            )
        else:
            covariance = initial_covariance

        return KalmanFilter(
            GaussianDistribution(np.asarray(state_mean), np.asarray(covariance))
        )

    return initiator


def _coerce_cost_vector(cost: Any, length: int, name: str) -> np.ndarray:
    """Normalize scalar or iterable costs to a length-``length`` vector."""

    if np.isscalar(cost):
        return np.full(length, float(cost))

    vector = np.asarray(cost, dtype=float).reshape(-1)
    if vector.size != length:
        raise ValueError(f"{name} must be scalar or have length {length}")
    return vector


# pylint: disable=duplicate-code
__all__ = [
    "AssociationResult",
    "Track",
    "TrackManager",
    "TrackManagerStepResult",
    "TrackStatus",
    "build_global_nearest_neighbor_associator",
    "build_kalman_measurement_initiator",
    "build_linear_gaussian_predictor",
    "build_linear_gaussian_updater",
    "solve_global_nearest_neighbor",
]
