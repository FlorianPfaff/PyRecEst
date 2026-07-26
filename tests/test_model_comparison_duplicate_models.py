import pandas as pd
from pyrecest.evaluation.model_comparison import evidence_margin_table


def test_evidence_margin_uses_distinct_models_for_runner_up():
    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "s1",
                "event_index": 0,
                "model": "a",
                "log_evidence": 10.0,
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "s1",
                "event_index": 0,
                "model": "b",
                "log_evidence": 7.0,
                "evidence_comparable": True,
            },
            {
                "status": "success",
                "session": "s1",
                "event_index": 0,
                "model": "a",
                "log_evidence": 9.0,
                "evidence_comparable": True,
            },
        ]
    )

    margins = evidence_margin_table(scores)

    assert margins["best_model_by_evidence"].tolist() == ["a"]
    assert margins["second_best_model_by_evidence"].tolist() == ["b"]
    assert margins["best_log_evidence"].tolist() == [9.0]
    assert margins["second_best_log_evidence"].tolist() == [7.0]
    assert margins["evidence_margin_to_second_best"].tolist() == [2.0]
    assert margins["models_compared"].tolist() == [2]
