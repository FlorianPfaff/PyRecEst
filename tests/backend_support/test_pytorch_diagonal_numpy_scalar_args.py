import numpy as np
import pytest
from pyrecest._backend import _common as common


def test_common_diagonal_accepts_numpy_scalar_integer_arguments_for_torch():
    torch = pytest.importorskip("torch")

    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    diag = common.diagonal(
        values,
        offset=np.array(1),
        axis1=np.array(1),
        axis2=np.array(2),
    )

    assert diag.cpu().numpy().tolist() == [[2.0, 6.0], [8.0, 12.0]]
