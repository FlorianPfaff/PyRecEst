"""Backend compatibility for fully unmasked NumPy masked arrays."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from . import validation as _validation


def _unwrap_fully_unmasked_array(value: Any) -> Any:
    """Expose ordinary NumPy data only when no entries are masked."""
    if isinstance(value, np.ma.MaskedArray) and not np.ma.is_masked(value):
        return np.asarray(value.data)
    return value


def install_masked_array_backend_compat() -> None:
    """Allow backend conversion of fully unmasked ``MaskedArray`` inputs."""
    current = _validation._as_backend_array
    if getattr(current, "_pyrecest_masked_array_backend_compat", False):
        return

    @wraps(current)
    def wrapped(value: Any, name: str):
        return current(_unwrap_fully_unmasked_array(value), name)

    wrapped._pyrecest_masked_array_backend_compat = True
    _validation._as_backend_array = wrapped
