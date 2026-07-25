import numpy as np
import numpy.testing as npt
import pytest
from pyrecest._backend import _common as common


def _to_numpy(value):
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(value)


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([1.0, 2.0], [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        ([[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        (np.arange(24.0).reshape(2, 3, 4), np.arange(20.0).reshape(4, 5)),
        (2.0, [[5.0, 6.0], [7.0, 8.0]]),
    ],
)
def test_common_dot_matches_numpy_contract(left, right):
    expected = np.dot(np.asarray(left), np.asarray(right))

    result = common.dot(left, right)

    npt.assert_allclose(_to_numpy(result), expected)
