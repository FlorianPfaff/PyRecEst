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


def _skip_if_linalg_function_unsupported(name):
    if name in get_unsupported_functions(backend.__backend_name__, "linalg"):
        pytest.skip(f"{name} is unsupported on the active backend")


def test_declared_linalg_unsupported_functions_raise_not_implemented():
    unsupported = get_unsupported_functions(backend.__backend_name__, "linalg")
    if not unsupported:
        pytest.skip("active backend has no declared unsupported linalg functions")

    for name in unsupported:
        with pytest.raises(NotImplementedError):
            getattr(linalg, name)(*_linalg_call_args(name))


def test_is_single_matrix_pd_rejects_vector_inputs():
    _skip_if_linalg_function_unsupported("is_single_matrix_pd")

    assert linalg.is_single_matrix_pd(array([1.0, 2.0])) is False


def test_is_single_matrix_pd_rejects_batched_inputs():
    _skip_if_linalg_function_unsupported("is_single_matrix_pd")

    batched_identity = array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )

    assert linalg.is_single_matrix_pd(batched_identity) is False


def test_pytorch_to_numpy_detaches_tensors_requiring_grad():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend behavior")

    import torch

    tensor = torch.tensor([1.0, 2.0], requires_grad=True)
    converted = backend.to_numpy(tensor)

    assert converted.tolist() == [1.0, 2.0]


def test_pytorch_to_numpy_resolves_conjugate_views():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific conjugate-view conversion")

    values = backend.conj(array([1.0 + 2.0j, 3.0 - 4.0j]))
    converted = backend.to_numpy(values)

    assert converted.tolist() == [1.0 - 2.0j, 3.0 + 4.0j]


def test_pytorch_logm_backward_handles_high_rank_batches():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific logm gradient regression test")

    torch = pytest.importorskip("torch")

    values = torch.eye(2, dtype=torch.float64).repeat(2, 3, 1, 1)
    values.requires_grad_(True)

    result = linalg.logm(values)
    result.sum().backward()

    assert values.grad is not None
    assert values.grad.shape == values.shape
    assert bool(torch.isfinite(values.grad).all())


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_pytorch_custom_gradient_accepts_keyword_arguments_and_defaults():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific custom_gradient regression test")

    torch = pytest.importorskip("torch")

    @backend.autodiff.custom_gradient(lambda x, scale=1.0, offset=0.0: 2.0 * scale * x)
    def quadratic(x, scale=1.0, offset=0.0):
        return scale * x * x + offset

    x = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    result = quadratic(x, scale=3.0)
    result.backward()

    assert float(result.detach()) == pytest.approx(12.0)
    assert float(x.grad) == pytest.approx(12.0)


def test_pytorch_cov_bias_accepts_array_weights():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific covariance regression test")

    result = backend.cov(
        array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        bias=True,
        aweights=array([1.0, 2.0, 1.0]),
    )

    assert result.shape == (2, 2)
    assert _to_python(result) == [[0.5, 1.0], [1.0, 2.0]]


def test_pytorch_cov_bias_honors_frequency_weights():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific covariance regression test")

    result = backend.cov(
        array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        bias=True,
        fweights=backend.asarray([1, 2, 1], dtype=backend.int64),
    )

    assert result.shape == (2, 2)
    assert _to_python(result) == [[0.5, 1.0], [1.0, 2.0]]


def test_array_like_sequence_helpers_accept_python_lists():
    assert _to_python(backend.concatenate(([1, 2], [3, 4]), axis=0)) == [1, 2, 3, 4]
    assert _to_python(backend.stack(([1, 2], [3, 4]), axis=0)) == [[1, 2], [3, 4]]
    assert _to_python(backend.flip([1, 2, 3], axis=0)) == [3, 2, 1]
    assert _to_python(backend.sort([3, 1, 2])) == [1, 2, 3]
    assert _to_python(backend.unique([2, 1, 2, 1])) == [1, 2]


def test_array_like_matrix_helpers_accept_python_lists():
    assert _to_python(backend.squeeze([[[1], [2]]], axis=2)) == [[1, 2]]
    assert _to_python(backend.trace([[1, 2], [3, 4]])) == 5
    assert _to_python(backend.diagonal([[1, 2], [3, 4]])) == [1, 4]
    assert _to_python(backend.tril([[1, 2], [3, 4]])) == [[1, 0], [3, 4]]
    assert _to_python(backend.triu([[1, 2], [3, 4]])) == [[1, 2], [0, 4]]


def test_cov_accepts_array_like_inputs():
    result = backend.cov([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])

    assert result.shape == (2, 2)
    assert bool(_to_python(backend.allclose(result, array([[1.0, 2.0], [2.0, 4.0]]))))


def test_one_hot_accepts_list_labels_and_returns_uint8_indicators():
    result = backend.one_hot([0, 2], 3)

    assert result.shape == (2, 3)
    assert str(backend.to_numpy(result).dtype) == "uint8"
    assert _to_python(result) == [[1, 0, 0], [0, 0, 1]]


def test_trace_uses_trailing_matrix_axes_for_batches():
    values = array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        ]
    )

    result = backend.trace(values)

    assert result.shape == (2,)
    assert _to_python(result) == [15.0, 42.0]


def test_divide_accepts_array_like_inputs():
    assert _to_python(backend.divide([2.0, 4.0], [1.0, 2.0])) == [2.0, 2.0]


def test_nonzero_returns_numpy_style_coordinate_tuple():
    result = backend.nonzero(array([[0, 1], [2, 0]]))

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert _to_python(result[0]) == [0, 1]
    assert _to_python(result[1]) == [1, 0]


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


def test_mean_accepts_numpy_axis_keyword_and_integer_inputs():
    values = array([[1, 2], [3, 4]])

    axis_result = backend.mean(values, axis=0)
    keepdims_result = backend.mean(values, axis=1, keepdims=True)

    assert _to_python(axis_result) == [2.0, 3.0]
    assert _to_python(keepdims_result) == [[1.5], [3.5]]


def test_arg_reductions_accept_numpy_axis_keepdims_and_array_like_inputs():
    values = [[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]]

    assert _to_python(backend.argmax(values, axis=1)) == [1, 2]
    assert _to_python(backend.argmin(values, axis=0)) == [0, 1, 0]
    assert _to_python(backend.argmax(values, axis=1, keepdims=True)) == [[1], [2]]


def test_arg_reductions_support_boolean_inputs():
    values = array([[False, True], [False, False]])

    assert _to_python(backend.argmax(values, axis=1)) == [1, 0]
    assert _to_python(backend.argmin(values, axis=1)) == [0, 0]


def test_sum_keepdims_without_axis_matches_numpy_contract():
    values = array([[1.0, 2.0], [3.0, 4.0]])

    result = backend.sum(values, keepdims=True)

    assert result.shape == (1, 1)
    assert _to_python(result) == [[10.0]]


def test_empty_axis_tuple_reductions_match_numpy_contract():
    values = array([[1.0, 2.0], [3.0, 4.0]])

    summed = backend.sum(values, axis=())
    averaged = backend.mean(values, axis=())
    standardized = backend.std(values, axis=())

    assert summed.shape == values.shape
    assert averaged.shape == values.shape
    assert standardized.shape == values.shape
    assert _to_python(summed) == [[1.0, 2.0], [3.0, 4.0]]
    assert _to_python(averaged) == [[1.0, 2.0], [3.0, 4.0]]
    assert _to_python(standardized) == [[0.0, 0.0], [0.0, 0.0]]


def test_prod_accepts_tuple_axis():
    values = array([[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]])

    result = backend.prod(values, axis=(0, 2))

    assert result.shape == (2,)
    assert _to_python(result) == [252.0, 1440.0]


def test_reductions_accept_numpy_integer_axis():
    np = pytest.importorskip("numpy")
    values = array([[1.0, 2.0], [3.0, 4.0]])

    result = backend.max(values, axis=np.int64(0))

    assert _to_python(result) == [3.0, 4.0]


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


def test_take_defaults_to_flattened_input():
    values = array([[0, 1, 2], [3, 4, 5]])

    result = backend.take(values, [[0, 2], [5, 1]])

    assert result.shape == (2, 2)
    assert _to_python(result) == [[0, 2], [5, 1]]


def test_take_preserves_multidimensional_index_shape():
    values = array([[0, 1, 2], [3, 4, 5]])

    result = backend.take(values, [[0, 2], [1, 0]], axis=1)

    assert result.shape == (2, 2, 2)
    assert _to_python(result) == [[[0, 2], [1, 0]], [[3, 5], [4, 3]]]


def test_take_wraps_negative_indices_in_raise_mode():
    values = array([[0, 1, 2], [3, 4, 5]])

    result = backend.take(values, [-1, 0], axis=1)

    assert result.shape == (2, 2)
    assert _to_python(result) == [[2, 0], [5, 3]]


def test_pytorch_take_raises_for_out_of_bounds_indices():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific take bounds regression test")

    values = array([0, 1, 2])

    with pytest.raises(IndexError):
        backend.take(values, [3])


def test_pad_accepts_scalar_pad_width():
    result = backend.pad(array([1, 2, 3]), 1)

    assert result.shape == (5,)
    assert _to_python(result) == [0, 1, 2, 3, 0]


def test_pad_treats_length_two_pad_width_as_before_after_for_each_axis():
    result = backend.pad(array([1, 2, 3]), (1, 2))

    assert result.shape == (6,)
    assert _to_python(result) == [0, 1, 2, 3, 0, 0]


def test_pad_broadcasts_single_axis_pair_to_all_axes():
    result = backend.pad(array([[1, 2], [3, 4]]), ((1, 2),))

    assert result.shape == (5, 5)
    assert _to_python(result) == [
        [0, 0, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_pad_uses_per_axis_pad_pairs_in_numpy_order():
    result = backend.pad(array([[1, 2], [3, 4]]), ((1, 0), (0, 2)))

    assert result.shape == (3, 4)
    assert _to_python(result) == [
        [0, 0, 0, 0],
        [1, 2, 0, 0],
        [3, 4, 0, 0],
    ]


def test_pad_rejects_negative_widths():
    with pytest.raises(ValueError):
        backend.pad(array([1, 2, 3]), (-1, 0))


def test_pytorch_pad_reflects_spatial_dimensions_with_unpadded_leading_axes():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific nonconstant pad regression test")

    values = backend.reshape(backend.arange(9.0), (1, 1, 3, 3))

    result = backend.pad(values, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")

    assert result.shape == (1, 1, 5, 5)
    assert _to_python(result[0, 0]) == [
        [4.0, 3.0, 4.0, 5.0, 4.0],
        [1.0, 0.0, 1.0, 2.0, 1.0],
        [4.0, 3.0, 4.0, 5.0, 4.0],
        [7.0, 6.0, 7.0, 8.0, 7.0],
        [4.0, 3.0, 4.0, 5.0, 4.0],
    ]


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


def test_split_integer_sections_reject_uneven_division():
    values = array([0, 1, 2, 3, 4])

    with pytest.raises(ValueError, match="equal division"):
        backend.split(values, 2)


def test_hsplit_splits_along_second_axis_for_high_rank_arrays():
    values = backend.reshape(backend.arange(24), (2, 3, 4))

    result = backend.hsplit(values, 3)

    assert [part.shape for part in result] == [(2, 1, 4), (2, 1, 4), (2, 1, 4)]
    assert [_to_python(part) for part in result] == [
        [[[0, 1, 2, 3]], [[12, 13, 14, 15]]],
        [[[4, 5, 6, 7]], [[16, 17, 18, 19]]],
        [[[8, 9, 10, 11]], [[20, 21, 22, 23]]],
    ]


def test_hsplit_accepts_cut_indices():
    values = backend.reshape(backend.arange(12), (2, 3, 2))

    result = backend.hsplit(values, [1, 2])

    assert [part.shape for part in result] == [(2, 1, 2), (2, 1, 2), (2, 1, 2)]
    assert [_to_python(part) for part in result] == [
        [[[0, 1]], [[6, 7]]],
        [[[2, 3]], [[8, 9]]],
        [[[4, 5]], [[10, 11]]],
    ]


def test_as_dtype_string_lookup_is_available():
    assert backend.as_dtype("float64") is not None


def test_ravel_tril_indices_returns_flat_indices():
    assert _to_python(backend.ravel_tril_indices(3)) == [0, 3, 4, 6, 7, 8]


def test_triangular_index_helpers_return_coordinate_tuples():
    tril = backend.tril_indices(3)
    triu = backend.triu_indices(3)

    assert isinstance(tril, tuple)
    assert isinstance(triu, tuple)
    assert len(tril) == 2
    assert len(triu) == 2

    tril_rows, tril_cols = tril
    triu_rows, triu_cols = triu
    assert _to_python(tril_rows) == [0, 1, 1, 2, 2, 2]
    assert _to_python(tril_cols) == [0, 0, 1, 0, 1, 2]
    assert _to_python(triu_rows) == [0, 0, 0, 1, 1, 2]
    assert _to_python(triu_cols) == [0, 1, 2, 1, 2, 2]


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
    assert _to_python(backend.divide(array([1.0, 2.0]), 0.0, ignore_div_zero=True)) == [
        0.0,
        0.0,
    ]
    assert _to_python(backend.divide(1.0, array([0.0, 2.0]), ignore_div_zero=True)) == [
        0.0,
        0.5,
    ]

    numerator = backend.asarray([1, 2], dtype=backend.int32)
    denominator = backend.asarray([1, 0], dtype=backend.int32)

    assert _to_python(backend.divide(numerator, denominator, ignore_div_zero=True)) == [
        1.0,
        0.0,
    ]


def test_pytorch_divide_ignore_div_zero_only_masks_zero_denominators():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific divide regression test")

    masked = backend.divide([1.0, 2.0], [0.0, 4.0], ignore_div_zero=True)
    assert _to_python(masked) == [0.0, 0.5]

    overflow = backend.divide(
        backend.asarray([1e308], dtype=backend.float64),
        backend.asarray([1e-308], dtype=backend.float64),
        ignore_div_zero=True,
    )
    assert _to_python(backend.isinf(overflow)) == [True]


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


def test_matvec_accepts_array_like_inputs():
    result = backend.matvec(
        [[1.0, 2.0], [3.0, 4.0]],
        [5.0, 6.0],
    )

    assert _to_python(result) == [17.0, 39.0]


def test_matvec_accepts_high_rank_vector_batches_for_shared_matrix():
    matrix = array([[1.0, 2.0], [3.0, 4.0]])
    vectors = array(
        [
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
        ]
    )

    result = backend.matvec(matrix, vectors)

    assert result.shape == (2, 3, 2)
    assert _to_python(result) == [
        [[17.0, 39.0], [23.0, 53.0], [29.0, 67.0]],
        [[35.0, 81.0], [41.0, 95.0], [47.0, 109.0]],
    ]


def test_batched_dot_uses_last_axis_inner_product():
    first = array([[1.0, 2.0], [3.0, 4.0]])
    second = array([[5.0, 6.0], [7.0, 8.0]])

    result = backend.dot(first, second)

    assert result.shape == (2,)
    assert _to_python(result) == [17.0, 53.0]


def test_dot_accepts_scalar_operands():
    assert _to_python(backend.dot(2.0, 3.0)) == 6.0
    assert _to_python(backend.dot(2.0, array([1.0, 2.0]))) == [2.0, 4.0]
    assert _to_python(backend.dot(array([1.0, 2.0]), 3.0)) == [3.0, 6.0]
    assert _to_python(backend.dot(array(2.0), array(3.0))) == 6.0


def test_dot_accepts_array_like_inputs():
    assert _to_python(backend.dot([1.0, 2.0], [3.0, 4.0])) == 11.0


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


def test_outer_accepts_array_like_inputs():
    result = backend.outer([1.0, 2.0], [3.0, 4.0])

    assert _to_python(result) == [[3.0, 4.0], [6.0, 8.0]]


def test_outer_accepts_scalar_operands():
    assert _to_python(backend.outer(2.0, array([1.0, 2.0]))) == [2.0, 4.0]
    assert _to_python(backend.outer(array([1.0, 2.0]), 3.0)) == [3.0, 6.0]


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
