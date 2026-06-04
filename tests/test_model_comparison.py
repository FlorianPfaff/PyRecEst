import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation.model_comparison import (
    add_evidence_margin_columns,
    cluster_bootstrap_margin_summary,
    evidence_margin_table,
    grouped_claim_gate_summary,
    infer_paired_model_group_cols,
    leave_one_group_out_summary,
    paired_model_margin_decisions,
    paired_model_margin_summary,
    paired_model_margin_threshold_sweep,
    select_paired_model_margin_threshold,
)


def test_paired_model_margin_decisions_summary_and_threshold_selection():
    scores = pd.DataFrame(
        [
            _score("s1", 0, "positive", 10.0, true_model="positive"),
            _score("s1", 0, "reference", 4.0, true_model="positive"),
            _score("s1", 1, "positive", 3.0, true_model="reference"),
            _score("s1", 1, "reference", 5.0, true_model="reference"),
            _score("s2", 2, "positive", 4.0, true_model="positive"),
            _score("s2", 2, "reference", 2.5, true_model="positive"),
        ]
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="positive",
        reference_model="reference",
        margin_threshold=2.0,
        true_model_col="true_model",
        positive_true_label="positive",
    )
    assert decisions["margin_decision"].tolist() == [
        "positive",
        "reference",
        "ambiguous",
    ]
    assert decisions["positive_model_claimed"].tolist() == [True, False, False]

    summary = paired_model_margin_summary(decisions, true_model_col="true_model")
    row = summary.iloc[0]
    assert row["events"] == 3
    assert row["positive_model_claims"] == 1
    assert row["reference_model_claims"] == 1
    assert row["ambiguous_events"] == 1
    assert row["false_positive_claims"] == 0
    assert row["positive_claim_recall"] == pytest.approx(0.5)

    sweep = paired_model_margin_threshold_sweep(
        scores,
        positive_model="positive",
        reference_model="reference",
        thresholds=(0.0, 2.0, 5.0),
        group_cols=("session", "event_index"),
        true_model_col="true_model",
        positive_true_label="positive",
    )
    selected = select_paired_model_margin_threshold(sweep, max_false_positive_claims=0)
    assert selected.iloc[0]["selection_status"] == "passed_specificity_gate"
    assert selected.iloc[0]["selected_margin_threshold"] == 0.0


def test_grouped_leave_one_out_and_cluster_bootstrap_are_generic():
    decisions = pd.DataFrame(
        [
            _decision("Rat1", 10.0, True),
            _decision("Rat1", 4.0, False),
            _decision("Rat2", 8.0, True),
            _decision("Rat2", 7.0, True),
            _decision("Rat3", 9.0, True),
            _decision("Rat3", -1.0, False),
        ]
    )

    grouped = paired_model_margin_summary(decisions, group_cols=("rat",))
    gates = grouped_claim_gate_summary(
        grouped,
        group_col="rat",
        claim_fraction_col="positive_claim_fraction",
        forbidden_claims_col="reference_model_claims",
        mean_delta_col="mean_positive_minus_reference_log_evidence",
        median_delta_col="median_positive_minus_reference_log_evidence",
        min_claim_fraction=0.0,
    )
    assert bool(gates.set_index("gate").loc["all_groups_mean_delta_positive", "passed"])

    leave_one = leave_one_group_out_summary(
        decisions,
        group_col="rat",
        held_out_col="held_out_rat",
        summary_fn=lambda frame: paired_model_margin_summary(frame),
    )
    assert set(leave_one["held_out_rat"]) == {"Rat1", "Rat2", "Rat3"}
    assert (leave_one["events"] == 4).all()

    bootstrap = cluster_bootstrap_margin_summary(
        decisions,
        cluster_col="rat",
        delta_col="positive_minus_reference_log_evidence",
        positive_claim_col="positive_model_claimed",
        n_bootstrap=100,
        random_seed=7,
    )
    assert bootstrap.iloc[0]["bootstrap_unit"] == "rat"
    assert bootstrap.iloc[0]["observed_clusters"] == 3
    assert bootstrap.iloc[0]["probability_mean_delta_gt_0"] > 0.9


def test_best_vs_second_best_margin_table_and_group_inference():
    scores = pd.DataFrame(
        [
            _score("s1", 0, "a", 0.0),
            _score("s1", 0, "b", 3.0),
            _score("s1", 0, "c", 1.0),
            _score("s1", 1, "a", 2.0, comparable=False),
            _score("s1", 1, "b", 1.0),
            _score("s1", 1, "c", 5.0),
        ]
    )
    margins = evidence_margin_table(scores)
    assert margins["best_model_by_evidence"].tolist() == ["b", "c"]
    assert margins["evidence_margin_to_second_best"].tolist() == [2.0, 4.0]

    augmented = add_evidence_margin_columns(scores)
    assert "evidence_margin_category" in augmented
    assert infer_paired_model_group_cols(scores) == ("session", "event_index")


def _score(
    session: str,
    event_index: int,
    model: str,
    log_evidence: float,
    *,
    true_model: str | None = None,
    comparable: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "status": "success",
        "session": session,
        "event_index": event_index,
        "model": model,
        "log_evidence": log_evidence,
        "evidence_comparable": comparable,
    }
    if true_model is not None:
        row["true_model"] = true_model
    return row


def _decision(rat: str, delta: float, positive_claimed: bool) -> dict[str, object]:
    return {
        "rat": rat,
        "positive_model": "positive",
        "reference_model": "reference",
        "margin_threshold": 2.0,
        "positive_minus_reference_log_evidence": delta,
        "positive_model_claimed": positive_claimed,
        "margin_decision": "positive" if positive_claimed else "ambiguous",
    }
