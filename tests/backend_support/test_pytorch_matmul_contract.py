import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
@pytest.mark.parametrize(
    ("left", "right", "expected_shape", "expected"),
    [
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            (),
            32.0,
        ),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [7.0, 8.0, 9.0],
            (2,),
            [50.0, 122.0],
        ),
        (
            [7.0, 8.0],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            (3,),
            [39.0, 54.0, 69.0],
        ),
    ],
)
def test_pytorch_matmul_matches_numpy_for_vector_operands(
    left, right, expected_shape, expected
):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        f"""
import pyrecest.backend as backend

left = backend.array({left!r})
right = backend.array({right!r})
expected = backend.array({expected!r})
actual = backend.matmul(left, right)

assert tuple(actual.shape) == {expected_shape!r}
assert backend.allclose(actual, expected)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
