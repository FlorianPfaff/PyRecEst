"""Generic model-comparison helpers for log-evidence tables.

The functions in this module operate on ordinary ``pandas.DataFrame`` objects.
They deliberately avoid application-specific vocabulary such as replay events,
rats, maps, or paper rows.  Downstream projects provide their own model labels,
grouping columns, cluster identifiers, and threshold values.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

EVIDENCE_MARGIN_CATEGORIES: tuple[tuple[str, float], ...] = (
    ("tie", 1.0),
    ("weak", 3.0),
    ("strong", 10.0),
    ("decisive", np.inf),
)

DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS = (0.0, 1.0, 3.0, 5.5, 10.0, 20.0)
DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_BOOTSTRAP_RANDOM_SEED = 1

__all__ = [
    "DEFAULT_BOOTSTRAP_RANDOM_SEED",
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_MARGIN_SENSITIVITY_THRESHOLDS",
    "EVIDENCE_MARGIN_CATEGORIES",
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
]


def _successful_rows(scores: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return scores.copy()
    if "status" in scores.columns:
        return scores[scores["status"].astype(str).eq("success")].copy()
    return scores.copy()


def _comparable_rows(scores: pd.DataFrame) -> pd.DataFrame:
    ok = _successful_rows(scores)
    if ok.empty:
        return ok
    if "evidence_comparable" in ok.columns:
        ok = ok[ok["evidence_comparable"].fillna(False).astype(bool)].copy()
    return ok


def classify_evidence_margin(delta_log_evidence: float) -> str:
    """Classify a non-negative log-evidence margin into qualitative buckets."""

    value = float(delta_log_evidence)
    if not np.isfinite(value):
        return "missing"
    for label, upper in EVIDENCE_MARGIN_CATEGORIES:
        if value <= upper:
            return label
    return "decisive"


def evidence_margin_table(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("session", "event_index"),
    evidence_col: str = "log_evidence",
    model_col: str = "model",
) -> pd.DataFrame:
    """Return one best-vs-runner-up evidence-margin row per group."""

    group_cols = tuple(group_cols)
    columns = [
        *group_cols,
        "best_model_by_evidence",
        "second_best_model_by_evidence",
        "best_log_evidence",
        "second_best_log_evidence",
        "evidence_margin_to_second_best",
        "evidence_margin_category",
        "models_compared",
    ]
    ok = _comparable_rows(scores)
    if ok.empty:
        return pd.DataFrame(columns=columns)
    missing = [
        column
        for column in (*group_cols, evidence_col, model_col)
        if column not in ok.columns
    ]
    if missing:
        raise KeyError(f"scores is missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        ranked = group.dropna(subset=[evidence_col]).sort_values(
            evidence_col, ascending=False
        )
        if ranked.empty:
            continue
        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None
        best_value = float(best[evidence_col])
        second_value = float(second[evidence_col]) if second is not None else np.nan
        margin = best_value - second_value if second is not None else np.inf
        row = {
            column: value for column, value in zip(group_cols, key_tuple, strict=True)
        }
        row.update(
            {
                "best_model_by_evidence": str(best[model_col]),
                "second_best_model_by_evidence": (
                    "" if second is None else str(second[model_col])
                ),
                "best_log_evidence": best_value,
                "second_best_log_evidence": second_value,
                "evidence_margin_to_second_best": float(margin),
                "evidence_margin_category": classify_evidence_margin(margin),
                "models_compared": int(len(ranked)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def add_evidence_margin_columns(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("session", "event_index"),
) -> pd.DataFrame:
    """Merge event-level evidence-margin diagnostics back into score rows."""

    if scores.empty:
        return scores.copy()
    margins = evidence_margin_table(scores, group_cols=group_cols)
    if margins.empty:
        out = scores.copy()
        out["evidence_margin_to_second_best"] = np.nan
        out["evidence_margin_category"] = "missing"
        return out
    return scores.merge(margins, on=list(group_cols), how="left")


def paired_model_margin_decisions(
    scores: pd.DataFrame,
    *,
    positive_model: str,
    reference_model: str,
    margin_threshold: float = 0.0,
    group_cols: Sequence[str] = ("session", "event_index"),
    evidence_col: str = "log_evidence",
    model_col: str = "model",
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    """Classify paired model wins using a symmetric log-evidence margin."""

    threshold = float(margin_threshold)
    if threshold < 0.0:
        raise ValueError("margin_threshold must be non-negative")
    group_cols = tuple(group_cols)
    columns = [
        *group_cols,
        "positive_model",
        "reference_model",
        "positive_log_evidence",
        "reference_log_evidence",
        "positive_minus_reference_log_evidence",
        "margin_threshold",
        "margin_decision",
        "positive_model_claimed",
    ]
    if true_model_col:
        columns.extend(
            [
                true_model_col,
                "positive_true_label",
                "true_is_positive",
                "margin_binary_correct",
            ]
        )

    ok = _comparable_rows(scores)
    if ok.empty:
        return pd.DataFrame(columns=columns)
    missing = [
        column
        for column in (*group_cols, evidence_col, model_col)
        if column not in ok.columns
    ]
    if true_model_col and true_model_col not in ok.columns:
        missing.append(true_model_col)
    if missing:
        raise KeyError(f"scores is missing required columns: {missing}")

    positive_label = positive_true_label or _model_family_label(positive_model)
    rows: list[dict[str, object]] = []
    for key, group in ok.groupby(list(group_cols), sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        paired = group[
            group[model_col].astype(str).isin([positive_model, reference_model])
        ]
        pivot = paired.dropna(subset=[evidence_col]).drop_duplicates(
            model_col, keep="last"
        )
        by_model = pivot.set_index(model_col)
        if (
            positive_model not in by_model.index
            or reference_model not in by_model.index
        ):
            continue
        positive_value = float(by_model.loc[positive_model, evidence_col])
        reference_value = float(by_model.loc[reference_model, evidence_col])
        delta = positive_value - reference_value
        if delta == 0.0:
            decision = "ambiguous"
            positive_claimed = False
        elif delta >= threshold:
            decision = positive_model
            positive_claimed = True
        elif delta <= -threshold:
            decision = reference_model
            positive_claimed = False
        else:
            decision = "ambiguous"
            positive_claimed = False
        row = {
            column: value for column, value in zip(group_cols, key_tuple, strict=True)
        }
        row.update(
            {
                "positive_model": positive_model,
                "reference_model": reference_model,
                "positive_log_evidence": positive_value,
                "reference_log_evidence": reference_value,
                "positive_minus_reference_log_evidence": float(delta),
                "margin_threshold": threshold,
                "margin_decision": decision,
                "positive_model_claimed": bool(positive_claimed),
            }
        )
        if true_model_col:
            true_label = _unique_text_value(group[true_model_col])
            true_is_positive = _model_family_label(true_label) == _model_family_label(
                positive_label
            )
            row.update(
                {
                    true_model_col: true_label,
                    "positive_true_label": positive_label,
                    "true_is_positive": bool(true_is_positive),
                    "margin_binary_correct": bool(positive_claimed)
                    == bool(true_is_positive),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def infer_paired_model_group_cols(scores: pd.DataFrame) -> tuple[str, ...]:
    """Infer columns that define one paired model-decision unit."""

    columns: list[str] = []
    for candidate in ("matrix_id", "random_seed", "session"):
        if candidate in scores.columns:
            columns.append(candidate)
    if "simulation_event_index" in scores.columns:
        columns.append("simulation_event_index")
    elif "event_index" in scores.columns:
        columns.append("event_index")
    if not columns:
        raise KeyError("could not infer paired event grouping columns")
    return tuple(columns)


def paired_model_margin_summary(
    decisions: pd.DataFrame,
    *,
    group_cols: Sequence[str] = (),
    true_model_col: str | None = None,
) -> pd.DataFrame:
    """Summarize a paired margin-decision table, optionally by groups."""

    group_cols = tuple(group_cols)
    base_columns = [
        *group_cols,
        "events",
        "positive_model",
        "reference_model",
        "margin_threshold",
        "positive_raw_wins",
        "reference_raw_wins",
        "raw_ties",
        "positive_raw_win_fraction",
        "positive_model_claims",
        "reference_model_claims",
        "ambiguous_events",
        "positive_claim_fraction",
        "reference_claim_fraction",
        "ambiguous_fraction",
        "mean_positive_minus_reference_log_evidence",
        "median_positive_minus_reference_log_evidence",
    ]
    true_columns = [
        "thresholded_binary_accuracy",
        "positive_true_events",
        "reference_true_events",
        "positive_true_claimed_events",
        "reference_true_rejected_events",
        "positive_claim_recall",
        "reference_specificity",
        "false_positive_claims",
        "false_negative_claims",
    ]
    columns = base_columns + (true_columns if true_model_col else [])
    if decisions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    groups = (
        [((), decisions)]
        if not group_cols
        else decisions.groupby(list(group_cols), sort=True)
    )
    for key, group in groups:
        key_tuple = key if isinstance(key, tuple) else (key,)
        delta = pd.to_numeric(
            group["positive_minus_reference_log_evidence"], errors="coerce"
        )
        events = int(len(group))
        positive_claims = int(
            group["positive_model_claimed"].fillna(False).astype(bool).sum()
        )
        reference_claims = int(
            (group["margin_decision"] == group["reference_model"]).sum()
        )
        ambiguous = int((group["margin_decision"] == "ambiguous").sum())
        row = {
            column: value for column, value in zip(group_cols, key_tuple, strict=True)
        }
        row.update(
            {
                "events": events,
                "positive_model": str(group["positive_model"].dropna().iloc[0]),
                "reference_model": str(group["reference_model"].dropna().iloc[0]),
                "margin_threshold": float(group["margin_threshold"].dropna().iloc[0]),
                "positive_raw_wins": int((delta > 0.0).sum()),
                "reference_raw_wins": int((delta < 0.0).sum()),
                "raw_ties": int((delta == 0.0).sum()),
                "positive_raw_win_fraction": float((delta > 0.0).mean()),
                "positive_model_claims": positive_claims,
                "reference_model_claims": reference_claims,
                "ambiguous_events": ambiguous,
                "positive_claim_fraction": float(positive_claims / max(events, 1)),
                "reference_claim_fraction": float(reference_claims / max(events, 1)),
                "ambiguous_fraction": float(ambiguous / max(events, 1)),
                "mean_positive_minus_reference_log_evidence": float(delta.mean()),
                "median_positive_minus_reference_log_evidence": float(delta.median()),
            }
        )
        if true_model_col and true_model_col in group:
            correct = group["margin_binary_correct"].fillna(False).astype(bool)
            true_positive = group["true_is_positive"].fillna(False).astype(bool)
            claims = group["positive_model_claimed"].fillna(False).astype(bool)
            row.update(
                {
                    "thresholded_binary_accuracy": float(correct.mean()),
                    "positive_true_events": int(true_positive.sum()),
                    "reference_true_events": int((~true_positive).sum()),
                    "positive_true_claimed_events": int((claims & true_positive).sum()),
                    "reference_true_rejected_events": int(
                        (~claims & ~true_positive).sum()
                    ),
                    "positive_claim_recall": _safe_ratio(
                        int((claims & true_positive).sum()), int(true_positive.sum())
                    ),
                    "reference_specificity": _safe_ratio(
                        int((~claims & ~true_positive).sum()),
                        int((~true_positive).sum()),
                    ),
                    "false_positive_claims": int((claims & ~true_positive).sum()),
                    "false_negative_claims": int((~claims & true_positive).sum()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def grouped_paired_model_margin_summary(
    decisions: pd.DataFrame,
    *,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Alias for grouped paired-margin summaries used by downstream reports."""

    return paired_model_margin_summary(decisions, group_cols=tuple(group_cols))


def paired_model_margin_threshold_sweep(
    scores: pd.DataFrame,
    *,
    positive_model: str,
    reference_model: str,
    thresholds: Sequence[float],
    group_cols: Sequence[str] | None = None,
    summary_group_cols: Sequence[str] = (),
    evidence_col: str = "log_evidence",
    model_col: str = "model",
    true_model_col: str | None = None,
    positive_true_label: str | None = None,
) -> pd.DataFrame:
    """Summarize paired margin decisions over candidate thresholds."""

    paired_group_cols = (
        tuple(group_cols)
        if group_cols is not None
        else infer_paired_model_group_cols(scores)
    )
    rows: list[pd.DataFrame] = []
    for threshold in thresholds:
        decisions = paired_model_margin_decisions(
            scores,
            positive_model=positive_model,
            reference_model=reference_model,
            margin_threshold=float(threshold),
            group_cols=paired_group_cols,
            evidence_col=evidence_col,
            model_col=model_col,
            true_model_col=true_model_col,
            positive_true_label=positive_true_label,
        )
        summary = paired_model_margin_summary(
            decisions,
            group_cols=tuple(summary_group_cols),
            true_model_col=true_model_col,
        )
        summary["group_cols"] = ",".join(paired_group_cols)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    sort_cols = ["margin_threshold", *tuple(summary_group_cols)]
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(sort_cols, kind="stable")
        .reset_index(drop=True)
    )


def select_paired_model_margin_threshold(
    threshold_sweep: pd.DataFrame,
    *,
    max_false_positive_claims: int = 0,
    min_positive_claim_recall: float = 0.0,
) -> pd.DataFrame:
    """Select the smallest threshold satisfying synthetic specificity gates."""

    if threshold_sweep.empty:
        return pd.DataFrame(
            [
                {
                    "selection_status": "empty_threshold_sweep",
                    "selected_margin_threshold": np.nan,
                }
            ]
        )
    if (
        "false_positive_claims" not in threshold_sweep
        or "positive_claim_recall" not in threshold_sweep
    ):
        raise KeyError(
            "threshold_sweep must include true-model false-positive and recall columns"
        )
    sweep = threshold_sweep.copy()
    sweep["passes_threshold_gate"] = (
        pd.to_numeric(sweep["false_positive_claims"], errors="coerce").fillna(np.inf)
        <= int(max_false_positive_claims)
    ) & (
        pd.to_numeric(sweep["positive_claim_recall"], errors="coerce").fillna(-np.inf)
        >= float(min_positive_claim_recall)
    )
    if sweep["passes_threshold_gate"].any():
        selected = (
            sweep[sweep["passes_threshold_gate"]]
            .sort_values("margin_threshold", kind="stable")
            .head(1)
            .copy()
        )
        selected["selection_status"] = "passed_specificity_gate"
    else:
        selected = (
            sweep.sort_values(
                ["false_positive_claims", "positive_claim_recall", "margin_threshold"],
                ascending=[True, False, True],
                kind="stable",
            )
            .head(1)
            .copy()
        )
        selected["selection_status"] = "fallback_no_gate_pass"
    selected["selected_margin_threshold"] = selected["margin_threshold"].astype(float)
    selected["max_false_positive_claims"] = int(max_false_positive_claims)
    selected["min_positive_claim_recall"] = float(min_positive_claim_recall)
    return selected.reset_index(drop=True)


def leave_one_group_out_summary(
    frame: pd.DataFrame,
    *,
    group_col: str,
    summary_fn: Callable[[pd.DataFrame], pd.DataFrame],
    held_out_col: str = "held_out_group",
) -> pd.DataFrame:
    """Apply a summary after holding out each value of ``group_col``."""

    if frame.empty:
        return pd.DataFrame()
    if group_col not in frame:
        raise KeyError(f"frame is missing required group column {group_col!r}")
    rows: list[pd.DataFrame] = []
    for group_value in sorted(frame[group_col].dropna().astype(str).unique()):
        retained = frame[frame[group_col].astype(str) != group_value]
        summary = summary_fn(retained)
        if summary.empty:
            continue
        summary = summary.copy()
        summary.insert(0, held_out_col, group_value)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(held_out_col)
        .reset_index(drop=True)
    )


def cluster_bootstrap_margin_summary(
    frame: pd.DataFrame,
    *,
    cluster_col: str,
    delta_col: str,
    positive_claim_col: str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES,
    random_seed: int = DEFAULT_BOOTSTRAP_RANDOM_SEED,
) -> pd.DataFrame:
    """Return cluster-bootstrap intervals for paired evidence-margin summaries."""

    columns = [
        "bootstrap_unit",
        "bootstrap_replicates",
        "random_seed",
        "observed_events",
        "observed_clusters",
        "observed_positive_raw_win_fraction",
        "positive_raw_win_fraction_ci95_low",
        "positive_raw_win_fraction_ci95_high",
        "observed_positive_claim_fraction",
        "positive_claim_fraction_ci95_low",
        "positive_claim_fraction_ci95_high",
        "observed_mean_delta",
        "mean_delta_ci95_low",
        "mean_delta_ci95_high",
        "probability_mean_delta_gt_0",
        "observed_median_delta",
        "median_delta_ci95_low",
        "median_delta_ci95_high",
        "probability_median_delta_gt_0",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = [
        column
        for column in (cluster_col, delta_col, positive_claim_col)
        if column not in frame
    ]
    if missing:
        raise KeyError(f"frame is missing required columns: {missing}")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")

    clusters = sorted(frame[cluster_col].dropna().astype(str).unique())
    if not clusters:
        return pd.DataFrame(columns=columns)
    by_cluster = {
        cluster: frame[frame[cluster_col].astype(str).eq(cluster)]
        for cluster in clusters
    }
    observed = _bootstrap_margin_metrics(
        frame, delta_col=delta_col, positive_claim_col=positive_claim_col
    )
    rng = np.random.default_rng(int(random_seed))
    replicate_rows = []
    for _ in range(int(n_bootstrap)):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat(
            [by_cluster[cluster] for cluster in sampled], ignore_index=True
        )
        replicate_rows.append(
            _bootstrap_margin_metrics(
                sample, delta_col=delta_col, positive_claim_col=positive_claim_col
            )
        )
    replicates = pd.DataFrame(replicate_rows)

    def ci(metric: str, q: float) -> float:
        return float(np.nanquantile(replicates[metric].to_numpy(dtype=float), q))

    return pd.DataFrame(
        [
            {
                "bootstrap_unit": str(cluster_col),
                "bootstrap_replicates": int(n_bootstrap),
                "random_seed": int(random_seed),
                "observed_events": int(len(frame)),
                "observed_clusters": int(len(clusters)),
                "observed_positive_raw_win_fraction": observed[
                    "positive_raw_win_fraction"
                ],
                "positive_raw_win_fraction_ci95_low": ci(
                    "positive_raw_win_fraction", 0.025
                ),
                "positive_raw_win_fraction_ci95_high": ci(
                    "positive_raw_win_fraction", 0.975
                ),
                "observed_positive_claim_fraction": observed["positive_claim_fraction"],
                "positive_claim_fraction_ci95_low": ci(
                    "positive_claim_fraction", 0.025
                ),
                "positive_claim_fraction_ci95_high": ci(
                    "positive_claim_fraction", 0.975
                ),
                "observed_mean_delta": observed["mean_delta"],
                "mean_delta_ci95_low": ci("mean_delta", 0.025),
                "mean_delta_ci95_high": ci("mean_delta", 0.975),
                "probability_mean_delta_gt_0": float(
                    (replicates["mean_delta"] > 0.0).mean()
                ),
                "observed_median_delta": observed["median_delta"],
                "median_delta_ci95_low": ci("median_delta", 0.025),
                "median_delta_ci95_high": ci("median_delta", 0.975),
                "probability_median_delta_gt_0": float(
                    (replicates["median_delta"] > 0.0).mean()
                ),
            }
        ],
        columns=columns,
    )


def grouped_claim_gate_summary(
    summary: pd.DataFrame,
    *,
    group_col: str,
    claim_fraction_col: str,
    forbidden_claims_col: str | None = None,
    mean_delta_col: str | None = None,
    median_delta_col: str | None = None,
    min_claim_fraction: float = 0.5,
) -> pd.DataFrame:
    """Return generic pass/fail gates for grouped claim robustness."""

    columns = ["gate", "passed", "observed", "criterion", "details"]
    rows: list[dict[str, object]] = []

    def add(
        gate: str, passed: bool, observed: object, criterion: str, details: str = ""
    ) -> None:
        rows.append(
            {
                "gate": gate,
                "passed": bool(passed),
                "observed": observed,
                "criterion": criterion,
                "details": details,
            }
        )

    if summary.empty:
        add("groups_present", False, 0, f"{group_col} groups > 0")
        result = pd.DataFrame(rows, columns=columns)
    else:
        missing = [
            column
            for column in (group_col, claim_fraction_col)
            if column not in summary
        ]
        for optional in (forbidden_claims_col, mean_delta_col, median_delta_col):
            if optional and optional not in summary:
                missing.append(optional)
        if missing:
            raise KeyError(f"summary is missing required columns: {missing}")
        add(
            "groups_present",
            True,
            int(summary[group_col].nunique()),
            f"{group_col} groups > 0",
        )
        min_claim = float(
            pd.to_numeric(summary[claim_fraction_col], errors="coerce").min()
        )
        add(
            "all_groups_claim_majority",
            min_claim > float(min_claim_fraction),
            f"{min_claim:.6g}",
            f"min {claim_fraction_col} > {float(min_claim_fraction):g}",
        )
        if forbidden_claims_col:
            max_forbidden = int(
                pd.to_numeric(summary[forbidden_claims_col], errors="coerce")
                .fillna(np.inf)
                .max()
            )
            add(
                "all_groups_no_forbidden_claims",
                max_forbidden == 0,
                max_forbidden,
                f"max {forbidden_claims_col} == 0",
            )
        if mean_delta_col:
            min_mean = float(
                pd.to_numeric(summary[mean_delta_col], errors="coerce").min()
            )
            add(
                "all_groups_mean_delta_positive",
                min_mean > 0.0,
                f"{min_mean:.6g}",
                f"min {mean_delta_col} > 0",
            )
        if median_delta_col:
            min_median = float(
                pd.to_numeric(summary[median_delta_col], errors="coerce").min()
            )
            add(
                "all_groups_median_delta_positive",
                min_median > 0.0,
                f"{min_median:.6g}",
                f"min {median_delta_col} > 0",
            )
        result = pd.DataFrame(rows, columns=columns)

    overall = pd.DataFrame(
        [
            {
                "gate": "overall",
                "passed": bool(result["passed"].all()) if not result.empty else False,
                "observed": (
                    f"{int(result['passed'].sum())}/{len(result)} gates passed"
                    if not result.empty
                    else "0/0"
                ),
                "criterion": "all grouped claim gates pass",
                "details": "",
            }
        ],
        columns=columns,
    )
    return pd.concat([result, overall], ignore_index=True)


def _bootstrap_margin_metrics(
    frame: pd.DataFrame,
    *,
    delta_col: str,
    positive_claim_col: str,
) -> dict[str, float]:
    delta = pd.to_numeric(frame[delta_col], errors="coerce")
    return {
        "positive_raw_win_fraction": float((delta > 0.0).mean()),
        "positive_claim_fraction": float(
            frame[positive_claim_col].fillna(False).astype(bool).mean()
        ),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
    }


def _unique_text_value(values: pd.Series) -> str:
    unique = [str(value) for value in values.dropna().unique()]
    if not unique:
        return ""
    return unique[0]


def _model_family_label(value: str) -> str:
    text = str(value).lower()
    if "momentum" in text:
        return "momentum"
    if "diffusion" in text:
        return "diffusion"
    return text


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)
