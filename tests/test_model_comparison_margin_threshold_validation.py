import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation import paired_model_margin_decisions


@pytest.mark.parametrize(
    "bad_threshold",
    [
        True,
        np.bool_(False),
        "1.0",
        b"1.0",
        1.0 + 0.0j,
        np.datetime64("2026-01-01"),
        np.timedelta64(1, "D"),
        np.asarray([1.0]),
        np.ma.masked,
        np.ma.array(1.0, mask=True),
    ],
)
def test_paired_model_margin_decisions_rejects_invalid_threshold_scalars(
    bad_threshold,
):
    with pytest.raises(
        ValueError,
        match="margin_threshold must be finite and non-negative",
    ):
        paired_model_margin_decisions(
            pd.DataFrame(),
            positive_model="positive",
            reference_model="reference",
            margin_threshold=bad_threshold,
        )


def test_paired_model_margin_decisions_accepts_unmasked_scalar_threshold():
    decisions = paired_model_margin_decisions(
        pd.DataFrame(),
        positive_model="positive",
        reference_model="reference",
        margin_threshold=np.ma.array(1.5, mask=False),
    )

    assert decisions.empty
