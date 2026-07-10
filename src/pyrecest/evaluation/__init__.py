from math import isfinite as _isfinite
from math import isinf as _isinf

from . import model_comparison as _model_comparison
from .check_and_fix_config import check_and_fix_config
from .configure_for_filter import configure_for_filter
from .determine_all_deviations import determine_all_deviations
from .evaluate_for_file import evaluate_for_file
from .evaluate_for_simulation_config import evaluate_for_simulation_config
from .evaluate_for_variables import evaluate_for_variables
from .generate_groundtruth import generate_groundtruth
from .generate_measurements import generate_measurements
from .generate_simulated_scenarios import generate_simulated_scenarios
from .get_axis_label import get_axis_label
from .get_distance_function import get_distance_function
from .get_extract_mean import get_extract_mean
from .implicit_surfaces import (
    classify_inside_outside,
    surface_band_mask,
    surface_band_probability_from_signed_distance,
    surface_gradients,
    surface_residuals,
    surface_variances,
)
from .iterate_configs_and_runs import iterate_configs_and_runs
from .model_comparison import (
    add_evidence_margin_columns,
    classify_evidence_margin,
    cluster_bootstrap_margin_summary,
    evidence_margin_table,
    grouped_claim_gate_summary,
    grouped_paired_model_margin_summary,
    infer_paired_model_group_cols,
    leave_one_group_out_summary,
    paired_model_margin_decisions,
    paired_model_margin_summary,
    paired_model_margin_threshold_sweep,
    select_paired_model_margin_threshold,
)
from .pareto import (
    constraint_mask,
    equal_quality_selection,
    is_pareto_front,
    pareto_front_indices,
    record_dominates,
    select_under_constraints,
)
from .perform_predict_update_cycles import perform_predict_update_cycles
from .plot_results import plot_results
from .point_set_metrics import (
    as_point_set,
    chamfer_distance,
    deterministic_subsample,
    distance_quantiles,
    nearest_neighbor_distances,
    point_set_geometry_summary,
    precision_recall_curve,
    precision_recall_fscore,
)
from .selection import (
    protected_tail_topk_mask,
    quantile_tail_mask,
    quantile_tail_threshold,
    retained_count_from_fraction,
    sanitized_score_vector,
    tail_rescue_quota_count,
    tail_rescue_topk_mask,
    top_count_mask,
    top_fraction_mask,
)
from .simulation_database import simulation_database
from .summarize_filter_results import summarize_filter_results

_ORIGINAL_CLASSIFIER_ATTR = "_pyrecest_original_classify_evidence_margin"
_ORIGINAL_DECISIONS_ATTR = "_pyrecest_original_paired_model_margin_decisions"

if not hasattr(_model_comparison, _ORIGINAL_CLASSIFIER_ATTR):
    setattr(
        _model_comparison,
        _ORIGINAL_CLASSIFIER_ATTR,
        classify_evidence_margin,
    )
if not hasattr(_model_comparison, _ORIGINAL_DECISIONS_ATTR):
    setattr(
        _model_comparison,
        _ORIGINAL_DECISIONS_ATTR,
        paired_model_margin_decisions,
    )

_original_classify_evidence_margin = getattr(
    _model_comparison, _ORIGINAL_CLASSIFIER_ATTR
)
_original_paired_model_margin_decisions = getattr(
    _model_comparison, _ORIGINAL_DECISIONS_ATTR
)


def _classify_evidence_margin(delta_log_evidence: float) -> str:
    value = float(delta_log_evidence)
    if _isinf(value) and value > 0.0:
        return "decisive"
    return _original_classify_evidence_margin(delta_log_evidence)


def _validated_margin_threshold(margin_threshold):
    threshold = float(margin_threshold)
    if threshold < 0.0 or not _isfinite(threshold):
        raise ValueError("margin_threshold must be finite and non-negative")
    return threshold


def _paired_model_margin_decisions(
    scores,
    *,
    positive_model,
    reference_model,
    margin_threshold=0.0,
    group_cols=("session", "event_index"),
    evidence_col="log_evidence",
    model_col="model",
    true_model_col=None,
    positive_true_label=None,
):
    return _original_paired_model_margin_decisions(
        scores,
        positive_model=positive_model,
        reference_model=reference_model,
        margin_threshold=_validated_margin_threshold(margin_threshold),
        group_cols=group_cols,
        evidence_col=evidence_col,
        model_col=model_col,
        true_model_col=true_model_col,
        positive_true_label=positive_true_label,
    )


_model_comparison.classify_evidence_margin = _classify_evidence_margin
classify_evidence_margin = _classify_evidence_margin
_model_comparison.paired_model_margin_decisions = _paired_model_margin_decisions
paired_model_margin_decisions = _paired_model_margin_decisions

__all__ = [
    "generate_groundtruth",
    "generate_measurements",
    "simulation_database",
    "check_and_fix_config",
    "configure_for_filter",
    "perform_predict_update_cycles",
    "iterate_configs_and_runs",
    "determine_all_deviations",
    "get_axis_label",
    "get_distance_function",
    "get_extract_mean",
    "summarize_filter_results",
    "generate_simulated_scenarios",
    "plot_results",
    "evaluate_for_file",
    "evaluate_for_simulation_config",
    "evaluate_for_variables",
    "classify_inside_outside",
    "surface_band_mask",
    "surface_band_probability_from_signed_distance",
    "surface_gradients",
    "surface_residuals",
    "surface_variances",
    "add_evidence_margin_columns",
    "classify_evidence_margin",
    "cluster_bootstrap_margin_summary",
    "evidence_margin_table",
    "grouped_claim_gate_summary",
    "grouped_paired_model_margin_summary",
    "infer_paired_model_group_cols",
    "leave_one_group_out_summary",
    "paired_model_margin_decisions",
    "paired_model_margin_summary",
    "paired_model_margin_threshold_sweep",
    "select_paired_model_margin_threshold",
    "constraint_mask",
    "equal_quality_selection",
    "is_pareto_front",
    "pareto_front_indices",
    "record_dominates",
    "select_under_constraints",
    "as_point_set",
    "chamfer_distance",
    "deterministic_subsample",
    "distance_quantiles",
    "nearest_neighbor_distances",
    "point_set_geometry_summary",
    "precision_recall_curve",
    "precision_recall_fscore",
    "protected_tail_topk_mask",
    "quantile_tail_mask",
    "quantile_tail_threshold",
    "retained_count_from_fraction",
    "sanitized_score_vector",
    "tail_rescue_quota_count",
    "tail_rescue_topk_mask",
    "top_count_mask",
    "top_fraction_mask",
]
