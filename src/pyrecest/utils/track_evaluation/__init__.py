from decimal import Decimal
from fractions import Fraction
import runpy
from pathlib import Path

import numpy as np

d = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "track_evaluation.py"), run_name=__name__
)
o = d["_optional_int_candidate"]


def _fractional_exact_number(v):
    if isinstance(v, Decimal):
        return v != v.to_integral_value()
    if isinstance(v, Fraction):
        return v.denominator != 1
    return False


def f(v):
    if isinstance(v, np.ndarray):
        if v.ndim != 0:
            return d["_MISSING"]
        v = v.item()
    if isinstance(v, (bool, np.bool_)):
        return d["_MISSING"]
    if _fractional_exact_number(v):
        return d["_MISSING"]
    return o(v)


d["_optional_int_candidate"] = f
for n in d["__all__"]:
    globals()[n] = d[n]
__all__ = d["__all__"]
