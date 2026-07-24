import numpy as np
import pytest
from pyrecest.evaluation.get_extract_mean import get_extract_mean


class FailingArrayScalar:
    def __init__(self, exception_type):
        self.exception_type = exception_type

    def __array__(self, dtype=None):
        del dtype
        raise self.exception_type("cannot convert")


@pytest.mark.parametrize("mtt_scenario", ["False", "true", 1, [False]])
def test_get_extract_mean_rejects_non_boolean_mtt_scenario(mtt_scenario):
    with pytest.raises(ValueError, match="mtt_scenario must be a bool"):
        get_extract_mean("euclidean", mtt_scenario=mtt_scenario)


@pytest.mark.parametrize(
    "mtt_scenario",
    [
        np.ma.masked,
        np.ma.array(True, mask=True),
        np.ma.array(False, mask=True),
    ],
)
def test_get_extract_mean_rejects_masked_boolean_mtt_scenario(mtt_scenario):
    with pytest.raises(ValueError, match="mtt_scenario must be a bool"):
        get_extract_mean("euclidean", mtt_scenario=mtt_scenario)


@pytest.mark.parametrize("exception_type", [RuntimeError, OverflowError])
def test_get_extract_mean_normalizes_array_conversion_failures(exception_type):
    with pytest.raises(ValueError, match="mtt_scenario must be a bool"):
        get_extract_mean(
            "euclidean",
            mtt_scenario=FailingArrayScalar(exception_type),
        )


def test_get_extract_mean_accepts_unmasked_boolean_mtt_scenario():
    extract_mean = get_extract_mean(
        "euclidean", mtt_scenario=np.ma.array(True, mask=False)
    )

    assert extract_mean([1, 2]) == [1, 2]
