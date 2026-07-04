"""Generic tracking event and replay-record helpers."""

from .event_records import (
    TrackingEvent,
    TrackingRecord,
    action_counts,
    event_from_measurement,
    record_from_update,
    records_to_dicts,
    records_to_matrix,
)

__all__ = [
    "TrackingEvent",
    "TrackingRecord",
    "action_counts",
    "event_from_measurement",
    "record_from_update",
    "records_to_dicts",
    "records_to_matrix",
]

from .hypothesis_replay import (
    HypothesisReplay,
    HypothesisReplayScore,
    InnovationConsistencyScoreConfig,
    rank_hypothesis_replays,
    rank_replayed_hypotheses,
    score_hypothesis_replay,
    scores_to_dicts,
)

__all__ += [
    "HypothesisReplay",
    "HypothesisReplayScore",
    "InnovationConsistencyScoreConfig",
    "rank_hypothesis_replays",
    "rank_replayed_hypotheses",
    "score_hypothesis_replay",
    "scores_to_dicts",
]

from .tracklet_graph import (
    Tracklet,
    TrackletEdge,
    TrackletGraphConfig,
    TrackletPath,
    build_tracklet_adjacency,
    constant_velocity_edge_cost,
    diverse_k_best_tracklet_paths,
    k_best_tracklet_paths,
    materialize_tracklet_path,
    path_jaccard,
    sort_tracklets,
    tracklet_paths_to_dicts,
)

__all__ += [
    "Tracklet",
    "TrackletEdge",
    "TrackletGraphConfig",
    "TrackletPath",
    "build_tracklet_adjacency",
    "constant_velocity_edge_cost",
    "diverse_k_best_tracklet_paths",
    "k_best_tracklet_paths",
    "materialize_tracklet_path",
    "path_jaccard",
    "sort_tracklets",
    "tracklet_paths_to_dicts",
]

from .measurement_reliability import (
    MeasurementReliabilityConfig,
    MeasurementReliabilityResult,
    ReliabilityWeightedMeasurement,
    apply_measurement_reliability,
    reliability_to_covariance_scale,
    scale_covariance_by_reliability,
)

__all__ += [
    "MeasurementReliabilityConfig",
    "MeasurementReliabilityResult",
    "ReliabilityWeightedMeasurement",
    "apply_measurement_reliability",
    "reliability_to_covariance_scale",
    "scale_covariance_by_reliability",
]

from .innovation_diagnostics import (
    InnovationDiagnostic,
    InnovationSummary,
    chi_square_gate_threshold,
    diagnostic_from_record,
    diagnostics_from_records,
    diagnostics_to_dicts,
    innovation_diagnostic,
    innovation_gate_threshold,
    linear_innovation_diagnostic,
    normalized_innovation_squared,
    summaries_to_dicts,
    summarize_innovation_diagnostics,
)

__all__ += [
    "InnovationDiagnostic",
    "InnovationSummary",
    "chi_square_gate_threshold",
    "diagnostic_from_record",
    "diagnostics_from_records",
    "diagnostics_to_dicts",
    "innovation_diagnostic",
    "innovation_gate_threshold",
    "linear_innovation_diagnostic",
    "normalized_innovation_squared",
    "summaries_to_dicts",
    "summarize_innovation_diagnostics",
]

from .residual_hypothesis_mht import (
    ResidualEditCandidate,
    ResidualHypothesis,
    ResidualMHTConfig,
    ResidualMHTPreset,
    enumerate_residual_hypotheses,
    hypotheses_to_dicts,
    hypothesis_to_dict,
    residual_mht_config_for_preset,
    select_residual_hypothesis,
)

__all__ += [
    "ResidualEditCandidate",
    "ResidualHypothesis",
    "ResidualMHTConfig",
    "ResidualMHTPreset",
    "enumerate_residual_hypotheses",
    "hypotheses_to_dicts",
    "hypothesis_to_dict",
    "residual_mht_config_for_preset",
    "select_residual_hypothesis",
]

from .residual_hypothesis_diagnostics import (
    candidate_to_dict,
    candidates_to_dicts,
    hypotheses_to_diagnostic_dicts,
    hypothesis_diagnostic_to_dict,
    selection_ledger_to_dicts,
)

__all__ += [
    "candidate_to_dict",
    "candidates_to_dicts",
    "hypotheses_to_diagnostic_dicts",
    "hypothesis_diagnostic_to_dict",
    "selection_ledger_to_dicts",
]

from .audit_guards import (
    ForbiddenKeyAccessError,
    GuardedMapping,
    assert_selector_invariant_under_forbidden_key_changes,
    guarded_mapping,
    guarded_mappings,
    poison_forbidden_keys,
    poison_forbidden_keys_in_mappings,
    strip_forbidden_keys,
    strip_forbidden_keys_from_mappings,
)

__all__ += [
    "ForbiddenKeyAccessError",
    "GuardedMapping",
    "assert_selector_invariant_under_forbidden_key_changes",
    "guarded_mapping",
    "guarded_mappings",
    "poison_forbidden_keys",
    "poison_forbidden_keys_in_mappings",
    "strip_forbidden_keys",
    "strip_forbidden_keys_from_mappings",
]

from .ellipse_geometry import (
    canonicalize_ellipse_axes,
    canonicalize_ellipse_shape,
    ellipse_angle_delta,
    ellipse_extent_matrix,
    ellipse_shape_canonicalization_transform,
    extent_matrix_from_shape,
    project_symmetric_covariance,
    rotation_matrix_2d,
    shape_from_extent_matrix,
    symmetrize,
    wrap_ellipse_angle_to_reference,
)

__all__ += [
    "canonicalize_ellipse_axes",
    "canonicalize_ellipse_shape",
    "ellipse_angle_delta",
    "ellipse_extent_matrix",
    "ellipse_shape_canonicalization_transform",
    "extent_matrix_from_shape",
    "project_symmetric_covariance",
    "rotation_matrix_2d",
    "shape_from_extent_matrix",
    "symmetrize",
    "wrap_ellipse_angle_to_reference",
]
