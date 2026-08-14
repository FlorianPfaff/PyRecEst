import pandas as pd
from pyrecest.evaluation.model_comparison import (
    evidence_margin_table,
    paired_model_margin_decisions,
)


def _score(model, evidence, comparable):
    return {
        "status": "success",
        "session": "s1",
        "event_index": 0,
        "model": model,
        "log_evidence": evidence,
        "evidence_comparable": comparable,
    }


def test_text_false_does_not_enter_evidence_ranking():
    scores = pd.DataFrame(
        [
            _score("incomparable", 10.0, "False"),
            _score("best_valid", 5.0, "TRUE"),
            _score("runner_up", 4.0, True),
        ]
    )

    margins = evidence_margin_table(scores)

    assert margins["best_model_by_evidence"].tolist() == ["best_valid"]
    assert margins["second_best_model_by_evidence"].tolist() == ["runner_up"]
    assert margins["evidence_margin_to_second_best"].tolist() == [1.0]
    assert margins["models_compared"].tolist() == [2]


def test_comparability_flags_fail_closed_for_unknown_serializations():
    scores = pd.DataFrame(
        [
            _score("yes", 3.0, "yes"),
            _score("one", 2.0, 1),
            _score("unknown", 100.0, "definitely"),
            _score("two", 90.0, 2),
            _score("missing", 80.0, pd.NA),
        ]
    )

    margins = evidence_margin_table(scores)

    assert margins["best_model_by_evidence"].tolist() == ["yes"]
    assert margins["second_best_model_by_evidence"].tolist() == ["one"]
    assert margins["models_compared"].tolist() == [2]


def test_text_false_does_not_enter_paired_model_decisions():
    scores = pd.DataFrame(
        [
            _score("positive", 100.0, "False"),
            _score("reference", 2.0, "True"),
        ]
    )

    decisions = paired_model_margin_decisions(
        scores,
        positive_model="positive",
        reference_model="reference",
    )

    assert decisions.empty
