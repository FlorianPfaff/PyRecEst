import pyrecest.backend as backend
import pytest


def _as_plain_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_matmul_accepts_matrix_like_lists():
    result = backend.matmul([[1, 2], [3, 4]], [[5], [6]])

    assert _as_plain_python(result) == [[17], [39]]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 32.0),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [7.0, 8.0, 9.0],
            [50.0, 122.0],
        ),
        ([7.0, 8.0], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [39.0, 54.0, 69.0]),
    ],
)
def test_matmul_accepts_numpy_style_vector_operands(left, right, expected):
    result = backend.matmul(left, right)

    assert _as_plain_python(result) == expected


def test_matmul_preserves_numpy_style_out_argument_for_lists():
    output = backend.empty((2, 1), dtype=backend.float64)

    result = backend.matmul([[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]], out=output)

    if backend.__backend_name__ == "jax":
        assert _as_plain_python(result) == [[17.0], [39.0]]
    else:
        assert result is output
        assert _as_plain_python(output) == [[17.0], [39.0]]
