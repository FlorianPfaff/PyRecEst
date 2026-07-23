import importlib

import pandas as pd
import pyrecest.evaluation as evaluation


def test_model_comparison_hotfixes_are_reload_idempotent():
    module = evaluation

    for _ in range(2):
        module = importlib.reload(module)

        assert module.classify_evidence_margin(2.0) == "weak"
        assert module.classify_evidence_margin(float("inf")) == "decisive"

        decisions = module.paired_model_margin_decisions(
            pd.DataFrame(),
            positive_model="positive",
            reference_model="reference",
        )
        assert decisions.empty
