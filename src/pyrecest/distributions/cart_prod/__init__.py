"""Cartesian-product distributions and compatibility fixes."""

from __future__ import annotations

from . import se2_bingham_distribution as _se2_bingham_distribution


def _validate_se2_bingham_sample_count(n) -> int:
    """Validate SE(2) Bingham sample counts without binary64 rounding."""
    count_array = _se2_bingham_distribution.np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    count = count_array.item()
    if isinstance(count, (bool, _se2_bingham_distribution.np.bool_)):
        raise ValueError("n must be an integer, not a boolean")

    try:
        count_int = int(count)
        is_exact_integer = bool(count == count_int)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    if not is_exact_integer:
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


_se2_bingham_distribution._validate_positive_sample_count = (
    _validate_se2_bingham_sample_count
)
