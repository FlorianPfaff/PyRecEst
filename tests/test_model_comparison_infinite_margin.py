import numpy as np
import pandas as pd
from pyrecest.evaluation.model_comparison import (
    classify_evidence_margin,
    evidence_margin_table,
)


def test_infinite_evidence_margin_is_decisive_not_missing():
    assert classify_evidence_margin(np.inf) == "decisive"
    assert classify_evidence_margin(-np.inf) == "missing"
    assert classify_evidence_margin(np.nan) == "missing"

    scores = pd.DataFrame(
        [
            {
                "status": "success",
                "session": "s1",
                "event_index": 0,
                "model": "only_model",
                "log_evidence": 7.0,
                "evidence_comparable": True,
            }
        ]
    )

    margins = evidence_margin_table(scores)

    assert margins["best_model_by_evidence"].tolist() == ["only_model"]
    assert np.isposinf(margins.loc[0, "evidence_margin_to_second_best"])
    assert margins["evidence_margin_category"].tolist() == ["decisive"]
