import numpy as np
import pandas as pd
from pyrecest.evaluation.model_comparison import grouped_claim_gate_summary


def test_grouped_claim_gate_summary_treats_missing_forbidden_counts_as_failed_gate():
    summary = pd.DataFrame(
        [
            {
                "rat": "Rat1",
                "positive_claim_fraction": 1.0,
                "reference_model_claims": np.nan,
            }
        ]
    )

    gates = grouped_claim_gate_summary(
        summary,
        group_col="rat",
        claim_fraction_col="positive_claim_fraction",
        forbidden_claims_col="reference_model_claims",
        min_claim_fraction=0.5,
    )

    by_gate = gates.set_index("gate")
    assert by_gate.loc["all_groups_no_forbidden_claims", "observed"] == "missing"
    assert not bool(by_gate.loc["all_groups_no_forbidden_claims", "passed"])
    assert not bool(by_gate.loc["overall", "passed"])
