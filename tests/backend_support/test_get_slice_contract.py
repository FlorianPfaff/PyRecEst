import pyrecest.backend as backend


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_get_slice_accepts_array_like_inputs():
    result = backend.get_slice([[0, 1, 2], [3, 4, 5]], ((0, 1), (2, 0)))

    assert _to_python(result) == [2, 3]


def test_get_slice_accepts_grouped_list_indices():
    values = backend.reshape(backend.arange(30), (3, 10))

    result = backend.get_slice(values, [[0, 2], [8, 9]])

    assert _to_python(result) == [8, 29]
