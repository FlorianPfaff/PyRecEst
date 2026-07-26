import pytest
from pyrecest.scenarios import _to_float_list


def test_to_float_list_rejects_noniterable_nonnumeric_values():
    with pytest.raises(ValueError, match="measurement must contain numeric values"):
        _to_float_list(object(), name="measurement", reject_text_or_bool=True)
