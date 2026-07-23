import pytest
from pyrecest.backend_support._pytorch_sort_numpy_contract import normalize_sort_axis


@pytest.mark.parametrize("axis", [True, False])
def test_pytorch_sort_axis_rejects_python_bool(axis):
    with pytest.raises(TypeError, match="integer"):
        normalize_sort_axis(axis)
