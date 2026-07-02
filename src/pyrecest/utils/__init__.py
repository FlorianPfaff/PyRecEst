"""Utility helpers for :mod:`pyrecest`."""

import math as _math

import numpy as _np

from . import assignment as _assignment_module
from . import multisession_assignment as _multisession_assignment_module
from . import roi_assignment as _roi_assignment_module
from ._multisession_assignment_labels import tracks_to_session_labels

_TEXT_SCALAR_TYPES = (str, bytes, bytearray, _np.str_, _np.bytes_)


def _murty_solution_with_full_assignment(solution, n_cols):
    if solution is None or "_full_assignment" in solution:
        return solution
    full_assignment = _assignment_module._array(solution["assignment"])
    for row_index, col_index in enumerate(full_assignment):
        if int(col_index) < 0:
            full_assignment[row_index] = n_cols + row_index
    patched_solution = dict(solution)
    patched_solution["_full_assignment"] = full_assignment
    return patched_solution


def _solve_subproblem_with_full_assignment(*args, **kwargs):
    solution = _assignment_module._solve_subproblem_without_full_assignment(
        *args, **kwargs
    )
    n_cols = args[2] if len(args) > 2 else kwargs["n_cols"]
    return _murty_solution_with_full_assignment(solution, n_cols)


if not hasattr(_assignment_module, "_solve_subproblem_without_full_assignment"):
    _assignment_module._solve_subproblem_without_full_assignment = (
        _assignment_module._solve_subproblem
    )
    _assignment_module._solve_subproblem = _solve_subproblem_with_full_assignment


def _normalize_roi_unmatched_value(unmatched_value):
    value_array = _roi_assignment_module.asarray(unmatched_value)
    if value_array.shape != ():
        raise ValueError("unmatched_value must be an integer.")

    scalar = value_array.item() if hasattr(value_array, "item") else value_array
    if isinstance(scalar, bool):
        raise ValueError("unmatched_value must be an integer.")
    if isinstance(scalar, int):
        return int(scalar)

    try:
        scalar_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("unmatched_value must be an integer.") from exc
    if not _math.isfinite(scalar_float) or not scalar_float.is_integer():
        raise ValueError("unmatched_value must be an integer.")
    return int(scalar_float)


def _assign_by_similarity_matrix_with_unmatched_value_validation(
    similarity_matrix,
    min_similarity=0.0,
    num_dummy=None,
    unmatched_value=-1,
    *,
    return_result=False,
):
    normalized_unmatched_value = _normalize_roi_unmatched_value(unmatched_value)
    similarities = _roi_assignment_module.asarray(
        similarity_matrix,
        dtype=_roi_assignment_module.float64,
    )
    if (
        similarities.ndim == 2
        and 0 <= normalized_unmatched_value < similarities.shape[1]
    ):
        raise ValueError(
            "unmatched_value must be outside the valid column index range."
        )

    return _roi_assignment_module._assign_by_similarity_matrix_without_unmatched_value_validation(
        similarity_matrix,
        min_similarity=min_similarity,
        num_dummy=num_dummy,
        unmatched_value=normalized_unmatched_value,
        return_result=return_result,
    )


if not hasattr(
    _roi_assignment_module,
    "_assign_by_similarity_matrix_without_unmatched_value_validation",
):
    _roi_assignment_module._assign_by_similarity_matrix_without_unmatched_value_validation = (
        _roi_assignment_module.assign_by_similarity_matrix
    )
    _roi_assignment_module.assign_by_similarity_matrix = (
        _assign_by_similarity_matrix_with_unmatched_value_validation
    )


def _validate_multisession_scalar_cost(name, value):
    message = f"{name} must be a finite scalar."
    try:
        value_array = _np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if (
        value_array.shape != ()
        or value_array.dtype == _np.bool_
        or value_array.dtype.kind in {"S", "U"}
    ):
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, _np.bool_) + _TEXT_SCALAR_TYPES):
        raise ValueError(message)
    try:
        scalar_float = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not _math.isfinite(scalar_float):
        raise ValueError(message)


min_cost_max_cardinality_assignment = (
    _assignment_module.min_cost_max_cardinality_assignment
)
murty_k_best_assignments = _assignment_module.murty_k_best_assignments
from .association_features import (
    CalibratedPairwiseAssociationModel,
    NamedPairwiseFeatureSchema,
    pairwise_feature_tensor,
)
from .association_models import LogisticPairwiseAssociationModel


def _patch_logistic_pairwise_backend_standardization() -> None:
    """Make association-model standardization use backend reduction contracts."""
    original_fit_standardization = LogisticPairwiseAssociationModel._fit_standardization
    if getattr(original_fit_standardization, "_pyrecest_backend_std_contract", False):
        return

    def _fit_standardization(self, features):
        import pyrecest.backend as _backend  # pylint: disable=import-outside-toplevel

        if self.standardize:
            feature_mean = _backend.mean(features, axis=0)
            feature_scale = _backend.std(features, axis=0)
            feature_scale = _backend.where(feature_scale > 0.0, feature_scale, 1.0)
        else:
            feature_mean = _backend.zeros(features.shape[1], dtype=_backend.float64)
            feature_scale = _backend.ones(features.shape[1], dtype=_backend.float64)

        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        return (features - feature_mean) / feature_scale

    _fit_standardization.__name__ = getattr(
        original_fit_standardization,
        "__name__",
        "_fit_standardization",
    )
    _fit_standardization.__doc__ = getattr(original_fit_standardization, "__doc__", None)
    _fit_standardization._pyrecest_backend_std_contract = True
    LogisticPairwiseAssociationModel._fit_standardization = _fit_standardization


def _patch_logistic_pairwise_scalar_prediction() -> None:
    """Make scalar prediction inputs valid for fitted one-feature models."""
    original_prepare_prediction_features = (
        LogisticPairwiseAssociationModel._prepare_prediction_features
    )
    if getattr(
        original_prepare_prediction_features,
        "_pyrecest_scalar_prediction_contract",
        False,
    ):
        return

    def _prepare_prediction_features(features, expected_feature_dimension):
        import pyrecest.backend as _backend  # pylint: disable=import-outside-toplevel

        features = _backend.asarray(features, dtype=_backend.float64)
        if features.ndim == 0:
            if expected_feature_dimension != 1:
                raise ValueError(
                    "A scalar prediction input is only valid for a fitted one-feature model"
                )
            flattened = features.reshape(1, 1)
            if not _backend.all(_backend.isfinite(flattened)):
                raise ValueError("features must be finite")
            return flattened, ()
        return original_prepare_prediction_features(features, expected_feature_dimension)

    _prepare_prediction_features.__name__ = getattr(
        original_prepare_prediction_features,
        "__name__",
        "_prepare_prediction_features",
    )
    _prepare_prediction_features.__doc__ = getattr(
        original_prepare_prediction_features,
        "__doc__",
        None,
    )
    _prepare_prediction_features._pyrecest_scalar_prediction_contract = True
    LogisticPairwiseAssociationModel._prepare_prediction_features = staticmethod(
        _prepare_prediction_features
    )


_patch_logistic_pairwise_backend_standardization()
_patch_logistic_pairwise_scalar_prediction()

from .candidate_pruning import (
    CandidatePruningConfig,
    candidate_mask_from_costs,
    candidate_pruning_config_from_mapping,
    prune_pairwise_cost_matrix,
)
from .history_recorder import HistoryRecorder
from .metrics import (
    anees,
    anis,
    chi_square_confidence_bounds,
    chi_square_confidence_interval,
    consistency_fraction,
    eot_shape_iou,
    extent_error,
    extent_intersection_over_union,
    extent_matrix_error,
    extent_wasserstein_distance,
    gaussian_wasserstein_distance,
    gospa_distance,
    iou_polygon,
    is_chi_square_consistent,
    is_within_chi_square_confidence_interval,
    mae,
    mospa_distance,
    mse,
    nees,
    nees_confidence_bounds,
    nees_confidence_interval,
    nis,
    nis_confidence_bounds,
    nis_confidence_interval,
    ospa_distance,
    rmse,
)
from .multisession_assignment import (
    MultiSessionAssignmentResult,
    solve_multisession_assignment,
)
from .multisession_assignment_observation_costs import (
    solve_multisession_assignment_with_observation_costs,
)
from .multisession_assignment_score import (
    solve_multisession_assignment_from_similarity,
    stitch_tracks_from_pairwise_scores,
    tracks_to_index_matrix,
)
from .nonrigid_point_set_registration import (
    ThinPlateSplineRegistrationResult,
    ThinPlateSplineTransform,
    estimate_thin_plate_spline,
    joint_tps_registration_assignment,
)
from .pairwise_covariance_features import (
    pairwise_covariance_shape_components,
    pairwise_mahalanobis_distances,
)
from .track_completion import (
    CandidateProvider,
    CandidateSessionProvider,
    CompletionCandidate,
    CompletionDirection,
    CompletionPath,
    CompletionStep,
    enumerate_fragment_completion_paths,
    occupied_observations_by_session,
    path_observations,
    path_sessions,
)
from .track_edit_whatif import (
    TrackEdit,
    TrackEditApplication,
    TrackEditDelta,
    apply_track_edit,
    rank_track_edits_by_delta,
    score_track_edit_delta,
    score_track_edits,
)
from .track_evaluation import (
    complete_track_set,
    normalize_track_matrix,
    pairwise_track_set,
    reference_fragment_counts,
    score_complete_tracks,
    score_false_continuations,
    score_fragmentation,
    score_pairwise_tracks,
    score_track_fragmentation,
    score_track_links,
    score_track_matrices,
    summarize_track_errors,
    summarize_tracks,
    track_error_ledger,
    track_lengths,
    track_pair_set,
)
from .track_metrics import (
    false_track_rate,
    missed_track_rate,
    score_false_tracks,
    score_missed_tracks,
    score_track_latency,
    score_track_outcomes,
    score_track_purity,
    track_latencies,
    track_purity,
)

_multisession_assignment_module.tracks_to_session_labels = tracks_to_session_labels
_multisession_assignment_module._validate_scalar_cost = _validate_multisession_scalar_cost

__all__ = [
    "MultiSessionAssignmentResult",
    "solve_multisession_assignment",
    "solve_multisession_assignment_from_similarity",
    "solve_multisession_assignment_with_observation_costs",
    "stitch_tracks_from_pairwise_scores",
    "tracks_to_index_matrix",
    "tracks_to_session_labels",
    "anis",
    "anees",
    "chi_square_confidence_bounds",
    "chi_square_confidence_interval",
    "consistency_fraction",
    "eot_shape_iou",
    "extent_error",
    "extent_intersection_over_union",
    "extent_matrix_error",
    "extent_wasserstein_distance",
    "false_track_rate",
    "gaussian_wasserstein_distance",
    "gospa_distance",
    "iou_polygon",
    "is_chi_square_consistent",
    "is_within_chi_square_confidence_interval",
    "mae",
    "missed_track_rate",
    "mospa_distance",
    "mse",
    "nees",
    "nees_confidence_bounds",
    "nees_confidence_interval",
    "nis",
    "nis_confidence_bounds",
    "nis_confidence_interval",
    "ospa_distance",
    "rmse",
    "score_false_tracks",
    "score_missed_tracks",
    "score_track_latency",
    "score_track_outcomes",
    "score_track_purity",
    "track_latencies",
    "track_purity",
    "TrackEdit",
    "TrackEditApplication",
    "TrackEditDelta",
    "apply_track_edit",
    "rank_track_edits_by_delta",
    "score_track_edit_delta",
    "score_track_edits",
    "CalibratedPairwiseAssociationModel",
    "LogisticPairwiseAssociationModel",
    "NamedPairwiseFeatureSchema",
    "pairwise_feature_tensor",
    "CandidatePruningConfig",
    "candidate_mask_from_costs",
    "candidate_pruning_config_from_mapping",
    "prune_pairwise_cost_matrix",
    "HistoryRecorder",
    "ThinPlateSplineRegistrationResult",
    "ThinPlateSplineTransform",
    "estimate_thin_plate_spline",
    "joint_tps_registration_assignment",
    "min_cost_max_cardinality_assignment",
    "murty_k_best_assignments",
    "pairwise_covariance_shape_components",
    "pairwise_mahalanobis_distances",
    "CompletionCandidate",
    "CompletionDirection",
    "CompletionPath",
    "CompletionStep",
    "CandidateProvider",
    "CandidateSessionProvider",
    "enumerate_fragment_completion_paths",
    "occupied_observations_by_session",
    "path_observations",
    "path_sessions",
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "reference_fragment_counts",
    "score_complete_tracks",
    "score_false_continuations",
    "score_fragmentation",
    "score_pairwise_tracks",
    "score_track_fragmentation",
    "score_track_links",
    "score_track_matrices",
    "summarize_track_errors",
    "summarize_tracks",
    "track_error_ledger",
    "track_lengths",
    "track_pair_set",
]
