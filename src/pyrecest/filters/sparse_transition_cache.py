"""Reusable sparse transition-row cache utilities."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

SparseTransitionRow = tuple[np.ndarray, np.ndarray]
SparseTransitionRowBuilder = Callable[[], SparseTransitionRow]
KeyT = TypeVar("KeyT", bound=Hashable)
StateT = TypeVar("StateT")


@dataclass
class SparseTransitionRowCache(Generic[KeyT]):
    """Cache sparse transition rows keyed by caller-defined hashable values."""

    rows: dict[KeyT, SparseTransitionRow] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_build(
        self, key: KeyT, builder: SparseTransitionRowBuilder
    ) -> SparseTransitionRow:
        """Return the cached row for ``key`` or build/cache it on demand."""

        try:
            row = self.rows[key]
        except KeyError:
            row = builder()
            self.rows[key] = row
            self.misses += 1
            return row
        self.hits += 1
        return row

    @property
    def entries(self) -> int:
        """Number of cached transition rows."""

        return len(self.rows)

    def clear(self) -> None:
        """Remove cached rows and reset hit/miss counters."""

        self.rows.clear()
        self.hits = 0
        self.misses = 0

    def diagnostics(self, prefix: str = "transition_row_cache") -> dict[str, int]:
        """Return cache diagnostics with a configurable key prefix."""

        return {
            f"{prefix}_entries": int(self.entries),
            f"{prefix}_hits": int(self.hits),
            f"{prefix}_misses": int(self.misses),
        }


def cached_sparse_transition_rows(
    source_states: Iterable[StateT],
    row_builder: Callable[[StateT], SparseTransitionRow],
    cache_key_builder: Callable[[StateT], Hashable | None],
    *,
    cache: SparseTransitionRowCache[Hashable] | None = None,
) -> tuple[list[SparseTransitionRow], SparseTransitionRowCache[Hashable]]:
    """Build transition rows with optional caller-defined row caching."""

    row_cache: SparseTransitionRowCache[Hashable]
    row_cache = cache if cache is not None else SparseTransitionRowCache()
    rows: list[SparseTransitionRow] = []
    for state in source_states:
        key = cache_key_builder(state)
        if key is None:
            rows.append(row_builder(state))
        else:
            rows.append(
                row_cache.get_or_build(key, lambda state=state: row_builder(state))
            )
    return rows, row_cache


__all__ = [
    "SparseTransitionRow",
    "SparseTransitionRowBuilder",
    "SparseTransitionRowCache",
    "cached_sparse_transition_rows",
]
