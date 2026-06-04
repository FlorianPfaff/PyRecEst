import numpy as np
from pyrecest.filters import SparseTransitionRowCache, cached_sparse_transition_rows


def test_sparse_transition_row_cache_reuses_rows_and_reports_diagnostics():
    cache: SparseTransitionRowCache[tuple[int, int]] = SparseTransitionRowCache()
    build_calls = 0

    def builder() -> tuple[np.ndarray, np.ndarray]:
        nonlocal build_calls
        build_calls += 1
        return np.array([0, 1]), np.array([0.25, 0.75])

    row1 = cache.get_or_build((1, 2), builder)
    row2 = cache.get_or_build((1, 2), builder)

    assert build_calls == 1
    assert row1 is row2
    assert cache.entries == 1
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.diagnostics() == {
        "transition_row_cache_entries": 1,
        "transition_row_cache_hits": 1,
        "transition_row_cache_misses": 1,
    }


def test_cached_sparse_transition_rows_bypasses_none_keys():
    cache: SparseTransitionRowCache = SparseTransitionRowCache()
    build_calls: list[int] = []

    def builder(state: int) -> tuple[np.ndarray, np.ndarray]:
        build_calls.append(state)
        return np.array([state]), np.array([1.0])

    rows, used_cache = cached_sparse_transition_rows(
        [1, 1, 2, 2],
        builder,
        lambda state: None if state == 2 else state,
        cache=cache,
    )

    assert used_cache is cache
    assert [row[0].item() for row in rows] == [1, 1, 2, 2]
    assert build_calls == [1, 2, 2]
    assert cache.entries == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_sparse_second_order_grid_accepts_external_transition_cache():
    from pyrecest.filters import sparse_second_order_grid_evidence

    cache: SparseTransitionRowCache[tuple[int, int, int]] = SparseTransitionRowCache()
    log_likelihood = np.log(np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.5, 0.5]]))

    def initial_pair_initializer(_scaled):
        return np.array([0, 1]), np.array([0, 1]), np.array([0.5, 0.5]), [1, 1]

    def transition_row_builder(_prev, curr, _transition_index):
        return np.array([curr]), np.array([1.0])

    result = sparse_second_order_grid_evidence(
        log_likelihood,
        initial_pair_initializer,
        transition_row_builder,
        transition_cache_key_builder=lambda prev, curr, transition_index: (
            prev,
            curr,
            transition_index,
        ),
        transition_row_cache=cache,
    )

    assert result.diagnostics["transition_row_cache_entries"] == cache.entries
    assert result.diagnostics["transition_row_cache_misses"] == cache.misses
