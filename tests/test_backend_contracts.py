import pyrecest.backend as backend
import pytest
from pyrecest._backend.capabilities import get_unsupported_functions
from pyrecest.backend import array, linalg

_MATRIX = array([[1.0, 0.0], [0.0, 1.0]])


def _linalg_call_args(name):
    return {
        "fractional_matrix_power": (_MATRIX, 0.5),
        "is_single_matrix_pd": (_MATRIX,),
        "logm": (_MATRIX,),
        "quadratic_assignment": (_MATRIX, _MATRIX, {}),
        "solve_sylvester": (_MATRIX, _MATRIX, _MATRIX),
    }[name]


def test_declared_linalg_unsupported_functions_raise_not_implemented():
    unsupported = get_unsupported_functions(backend.__backend_name__, "linalg")
    if not unsupported:
        pytest.skip("active backend has no declared unsupported linalg functions")

    for name in unsupported:
        with pytest.raises(NotImplementedError):
            getattr(linalg, name)(*_linalg_call_args(name))


def test_pytorch_to_numpy_detaches_tensors_requiring_grad():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend behavior")

    import torch

    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    converted = backend.to_numpy(tensor)

    assert converted.tolist() == [1.0, 2.0]


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_numpy_vmap_rejects_mismatched_leading_dimensions():
    if backend.__backend_name__ != "numpy":
        pytest.skip("NumPy-specific vmap regression test")

    vmapped = backend.vmap(lambda x, y: x + y)

    with pytest.raises(ValueError, match="same size in the first dimension"):
        vmapped(array([[1.0], [2.0]]), array([[1.0]]))


def test_default_reductions_reduce_all_elements():
    values = array([[2.0, 3.0], [4.0, 5.0]])

    assert float(_to_python(backend.min(values))) == 2.0
    assert float(_to_python(backend.prod(values))) == 120.0


def test_default_cumprod_flattens_all_elements():
    values = array([[2.0, 3.0], [4.0, 5.0]])

    assert _to_python(backend.cumprod(values)) == [2.0, 6.0, 24.0, 120.0]


def test_take_preserves_singleton_dimensions_for_vector_indices():
    values = array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]
    )

    result = backend.take(values, [0], axis=0)

    assert result.shape == (1, 1, 3)
    assert _to_python(result) == [[[1.0, 2.0, 3.0]]]


def test_take_removes_axis_for_scalar_indices():
    values = array(
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]
    )

    result = backend.take(values, 0, axis=0)

    assert result.shape == (1, 3)
    assert _to_python(result) == [[1.0, 2.0, 3.0]]


def test_cross_uses_trailing_vector_axis_for_batched_vectors():
    first = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    second = array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = backend.cross(first, second)

    assert _to_python(result) == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]


def test_get_slice_uses_axis_grouped_indices_contract():
    values = array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        ]
    )

    result = backend.get_slice(values, ((0, 2), (8, 9)))

    assert _to_python(result) == [8, 29]


def test_scatter_add_preserves_input_values_and_uses_facade_signature():
    values = array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    indices = backend.asarray([[0, 2], [1, 2]], dtype=backend.int64)
    updates = array([[2.0, 3.0], [4.0, 5.0]])

    result = backend.scatter_add(values, 1, indices, updates)

    assert _to_python(result) == [[3.0, 1.0, 4.0], [1.0, 5.0, 6.0]]


def test_as_dtype_string_lookup_is_available():
    assert backend.as_dtype("float64") is not None


def test_ravel_tril_indices_returns_flat_indices():
    assert _to_python(backend.ravel_tril_indices(3)) == [0, 3, 4, 6, 7, 8]


def test_triangular_matrix_helpers_return_compact_vectors():
    values = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    assert _to_python(backend.tril_to_vec(values)) == [1, 4, 5, 7, 8, 9]
    assert _to_python(backend.triu_to_vec(values)) == [1, 2, 3, 5, 6, 9]
    assert _to_python(backend.triu_to_vec(values, k=1)) == [2, 3, 6]


def test_triangular_matrix_helpers_preserve_batch_dimensions():
    values = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    assert _to_python(backend.tril_to_vec(values)) == [[1, 3, 4], [5, 7, 8]]
    assert _to_python(backend.triu_to_vec(values)) == [[1, 2, 4], [5, 6, 8]]


def test_divide_ignore_div_zero_accepts_scalars_and_integer_inputs():
    assert _to_python(backend.divide(array([1.0, 2.0]), 0.0, ignore_div_zero=True)) == [0.0, 0.0]
    assert _to_python(backend.divide(1.0, array([0.0, 2.0]), ignore_div_zero=True)) == [0.0, 0.5]

    numerator = backend.asarray([1, 2], dtype=backend.int32)
    denominator = backend.asarray([1, 0], dtype=backend.int32)

    assert _to_python(backend.divide(numerator, denominator, ignore_div_zero=True)) == [1.0, 0.0]


def test_vec_to_diag_preserves_leading_singleton_batch_dimension():
    result = backend.vec_to_diag(array([[1.0, 2.0]]))

    assert result.shape == (1, 2, 2)
    assert _to_python(result) == [[[1.0, 0.0], [0.0, 2.0]]]


def test_tril_and_triu_to_vec_skip_masked_entries():
    values = array([[1.0, 2.0], [3.0, 4.0]])

    assert _to_python(backend.tril_to_vec(values)) == [1.0, 3.0, 4.0]
    assert _to_python(backend.triu_to_vec(values)) == [1.0, 2.0, 4.0]


def test_set_diag_preserves_leading_batch_dimensions():
    result = backend.set_diag(backend.zeros((1, 2, 2)), array([[1.0, 2.0]]))

    assert result.shape == (1, 2, 2)
    assert _to_python(result) == [[[1.0, 0.0], [0.0, 2.0]]]


def test_set_diag_accepts_rectangular_matrices():
    result = backend.set_diag(backend.ones((2, 3)), array([5.0, 6.0]))

    assert result.shape == (2, 3)
    assert _to_python(result) == [[5.0, 1.0, 1.0], [1.0, 6.0, 1.0]]


def test_batched_mat_from_diag_triu_tril_preserves_leading_dimensions():
    result = backend.mat_from_diag_triu_tril(
        array([[1.0, 2.0], [5.0, 6.0]]),
        array([[3.0], [7.0]]),
        array([[4.0], [8.0]]),
    )

    assert result.shape == (2, 2, 2)
    assert _to_python(result) == [
        [[1.0, 3.0], [4.0, 2.0]],
        [[5.0, 7.0], [8.0, 6.0]],
    ]


def test_batched_matvec_pairs_leading_dimensions():
    matrix = array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]]])
    vector = array([[1.0, 2.0], [3.0, 4.0]])

    result = backend.matvec(matrix, vector)

    assert result.shape == (2, 2)
    assert _to_python(result) == [[1.0, 2.0], [6.0, 12.0]]


def test_batched_dot_uses_last_axis_inner_product():
    first = array([[1.0, 2.0], [3.0, 4.0]])
    second = array([[5.0, 6.0], [7.0, 8.0]])

    result = backend.dot(first, second)

    assert result.shape == (2,)
    assert _to_python(result) == [17.0, 53.0]


def test_batched_dot_accepts_high_rank_right_operand():
    first = array([1.0, 2.0])
    second = array(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )

    result = backend.dot(first, second)

    assert result.shape == (2, 3)
    assert _to_python(result) == [[2.0, 8.0, 14.0], [20.0, 26.0, 32.0]]


def test_batched_outer_pairs_leading_dimensions():
    first = array([[1.0, 2.0], [3.0, 4.0]])
    second = array([[5.0, 6.0], [7.0, 8.0]])

    result = backend.outer(first, second)

    assert result.shape == (2, 2, 2)
    assert _to_python(result) == [
        [[5.0, 6.0], [10.0, 12.0]],
        [[21.0, 24.0], [28.0, 32.0]],
    ]


def test_outer_accepts_high_rank_right_operand():
    first = array([1.0, 2.0])
    second = array(
        [
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
        ]
    )

    result = backend.outer(first, second)

    assert result.shape == (2, 3, 2, 2)
    assert _to_python(result) == [
        [
            [[0.0, 1.0], [0.0, 2.0]],
            [[2.0, 3.0], [4.0, 6.0]],
            [[4.0, 5.0], [8.0, 10.0]],
        ],
        [
            [[6.0, 7.0], [12.0, 14.0]],
            [[8.0, 9.0], [16.0, 18.0]],
            [[10.0, 11.0], [20.0, 22.0]],
        ],
    ]
