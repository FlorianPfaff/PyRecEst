import importlib.util

import pyrecest.backend as backend
import pytest
from pyrecest.backend_tools import get_backend_name


@pytest.mark.backend_portable
def test_raw_pytorch_cumulative_out_contract_with_numpy_public_backend():
    if get_backend_name() != "numpy":
        pytest.skip("raw PyTorch/non-PyTorch public backend contract needs NumPy")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    from pyrecest._backend import pytorch as pytorch_backend

    values = pytorch_backend.reshape(pytorch_backend.arange(1, 7), (2, 3))
    out_sum = pytorch_backend.zeros((2, 3), dtype=values.dtype)
    out_prod = pytorch_backend.zeros((2, 3), dtype=values.dtype)

    result_sum = pytorch_backend.cumsum(values, axis=1, out=out_sum)
    result_prod = pytorch_backend.cumprod(values, axis=1, out=out_prod)

    assert result_sum is out_sum
    assert result_prod is out_prod
    assert pytorch_backend.to_numpy(result_sum).tolist() == [[1, 3, 6], [4, 9, 15]]
    assert pytorch_backend.to_numpy(result_prod).tolist() == [[1, 2, 6], [4, 20, 120]]

    flat = pytorch_backend.cumsum(
        values,
        out=pytorch_backend.zeros((6,), dtype=values.dtype),
    )
    assert pytorch_backend.to_numpy(flat).tolist() == [1, 3, 6, 10, 15, 21]
