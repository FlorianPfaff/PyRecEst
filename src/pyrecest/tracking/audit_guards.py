"""Generic audit/no-leakage guards for benchmarked selectors.

The helpers in this module are intentionally domain-agnostic.  They are useful
when a benchmark ledger contains both selection features and audit-only columns
such as ground-truth labels or score deltas.  Selectors can be run against
``GuardedMapping`` rows to fail fast if they access forbidden keys, and their
outputs can be checked for invariance after forbidden columns are stripped or
poisoned.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


class ForbiddenKeyAccessError(KeyError):
    """Raised when code accesses a forbidden audit-only key."""


@dataclass(frozen=True)
class GuardedMapping(Mapping[str, Any]):
    """Read-only mapping that raises on access to forbidden keys."""

    mapping: Mapping[str, Any]
    forbidden_keys: frozenset[str]

    def __getitem__(self, key: str) -> Any:
        if str(key) in self.forbidden_keys:
            raise ForbiddenKeyAccessError(str(key))
        return self.mapping[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.mapping if str(key) not in self.forbidden_keys)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __contains__(self, key: object) -> bool:
        if str(key) in self.forbidden_keys:
            raise ForbiddenKeyAccessError(str(key))
        return key in self.mapping

    def get(self, key: str, default: Any = None) -> Any:
        if str(key) in self.forbidden_keys:
            raise ForbiddenKeyAccessError(str(key))
        return self.mapping.get(key, default)


def guarded_mapping(
    mapping: Mapping[str, Any],
    forbidden_keys: Iterable[str],
) -> GuardedMapping:
    """Wrap one mapping so forbidden audit keys fail on access."""

    return GuardedMapping(mapping=mapping, forbidden_keys=frozenset(map(str, forbidden_keys)))


def guarded_mappings(
    rows: Iterable[Mapping[str, Any]],
    forbidden_keys: Iterable[str],
) -> tuple[GuardedMapping, ...]:
    """Wrap a sequence of mapping rows with :class:`GuardedMapping`."""

    frozen = frozenset(map(str, forbidden_keys))
    return tuple(GuardedMapping(mapping=row, forbidden_keys=frozen) for row in rows)


def strip_forbidden_keys(
    mapping: Mapping[str, Any],
    forbidden_keys: Iterable[str],
) -> dict[str, Any]:
    """Return a copy of ``mapping`` without forbidden audit-only keys."""

    forbidden = frozenset(map(str, forbidden_keys))
    return {str(key): value for key, value in mapping.items() if str(key) not in forbidden}


def strip_forbidden_keys_from_mappings(
    rows: Iterable[Mapping[str, Any]],
    forbidden_keys: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Return copies of rows with forbidden audit-only keys removed."""

    forbidden = frozenset(map(str, forbidden_keys))
    return tuple(strip_forbidden_keys(row, forbidden) for row in rows)


def poison_forbidden_keys(
    mapping: Mapping[str, Any],
    forbidden_keys: Iterable[str],
    *,
    poison_value: Any = "__PYRECEST_FORBIDDEN_AUDIT_VALUE__",
) -> dict[str, Any]:
    """Return a copy with forbidden keys overwritten by a sentinel value."""

    forbidden = frozenset(map(str, forbidden_keys))
    output = {str(key): value for key, value in mapping.items()}
    for key in forbidden:
        if key in output:
            output[key] = poison_value
    return output


def poison_forbidden_keys_in_mappings(
    rows: Iterable[Mapping[str, Any]],
    forbidden_keys: Iterable[str],
    *,
    poison_value: Any = "__PYRECEST_FORBIDDEN_AUDIT_VALUE__",
) -> tuple[dict[str, Any], ...]:
    """Return copies of rows with forbidden audit-only keys poisoned."""

    forbidden = frozenset(map(str, forbidden_keys))
    return tuple(
        poison_forbidden_keys(row, forbidden, poison_value=poison_value) for row in rows
    )


def _comparison_is_true(comparison: Any) -> bool:
    """Reduce scalar or array-like equality results to one boolean."""

    reduce_all = getattr(comparison, "all", None)
    if callable(reduce_all):
        try:
            comparison = reduce_all()
        except (TypeError, ValueError, RuntimeError):
            return False
    try:
        return bool(comparison)
    except (TypeError, ValueError, RuntimeError):
        return False


def _results_equal(left: Any, right: Any) -> bool:
    """Compare selector outputs, treating aligned NaN values as unchanged."""

    try:
        comparison = (left == right) | ((left != left) & (right != right))
    except (TypeError, ValueError, RuntimeError):
        return False
    return _comparison_is_true(comparison)


def assert_selector_invariant_under_forbidden_key_changes(
    selector: Callable[[Sequence[Mapping[str, Any]]], T],
    rows: Sequence[Mapping[str, Any]],
    forbidden_keys: Iterable[str],
    *,
    normalize: Callable[[T], Any] | None = None,
    poison_value: Any = "__PYRECEST_FORBIDDEN_AUDIT_VALUE__",
) -> None:
    """Assert selector output is unchanged by removing/poisoning audit keys.

    The selector is evaluated on the original rows, rows with forbidden columns
    stripped, rows with forbidden columns poisoned, and guarded rows that raise
    if forbidden columns are accessed.  ``normalize`` may be provided when the
    selector returns a rich object and only a stable identifier should be
    compared. Scalar and array-valued normalized outputs are supported.
    """

    forbidden = frozenset(map(str, forbidden_keys))
    normalize_result = normalize or (lambda result: result)
    baseline = normalize_result(selector(rows))
    stripped = normalize_result(selector(strip_forbidden_keys_from_mappings(rows, forbidden)))
    poisoned = normalize_result(
        selector(
            poison_forbidden_keys_in_mappings(
                rows,
                forbidden,
                poison_value=poison_value,
            )
        )
    )
    guarded = normalize_result(selector(guarded_mappings(rows, forbidden)))
    if not all(
        _results_equal(baseline, candidate)
        for candidate in (stripped, poisoned, guarded)
    ):
        raise AssertionError(
            "selector output changed when forbidden audit-only keys were removed, "
            "poisoned, or guarded"
        )
