"""Compatibility wrapper for :mod:`pyrecest.utils.metrics`."""

from __future__ import annotations

import math as _math
import runpy
from pathlib import Path

_metrics = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "metrics.py"), run_name=__name__
)


def _validate_order_cutoff(order: float, cutoff: float) -> tuple[float, float]:
    order = float(order)
    cutoff = float(cutoff)
    if not _math.isfinite(order) or order < 1.0:
        raise ValueError("order must be finite and at least 1")
    if cutoff <= 0.0 or not _math.isfinite(cutoff):
        raise ValueError("cutoff must be a finite positive number")
    return order, cutoff


_metrics["_validate_order_cutoff"] = _validate_order_cutoff
for _name, _value in _metrics.items():
    if not (_name.startswith("__") and _name != "__all__"):
        globals()[_name] = _value
__all__ = _metrics["__all__"]
