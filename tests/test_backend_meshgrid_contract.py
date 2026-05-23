import pyrecest.backend as backend


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_meshgrid_accepts_numpy_style_array_like_axes():
    rows, cols = backend.meshgrid([0, 1], range(2), indexing="ij")

    assert _to_python(rows) == [[0, 0], [1, 1]]
    assert _to_python(cols) == [[0, 1], [0, 1]]


def test_meshgrid_accepts_scalar_axes():
    rows, cols = backend.meshgrid(1, [2, 3], indexing="ij")

    assert _to_python(rows) == [[1, 1]]
    assert _to_python(cols) == [[2, 3]]
