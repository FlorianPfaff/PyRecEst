import pytest
from pyrecest.evaluation.get_extract_mean import get_extract_mean


@pytest.mark.parametrize("mtt_scenario", ["False", "true", 1, [False]])
def test_get_extract_mean_rejects_non_boolean_mtt_scenario(mtt_scenario):
    with pytest.raises(ValueError, match="mtt_scenario must be a bool"):
        get_extract_mean("euclidean", mtt_scenario=mtt_scenario)
