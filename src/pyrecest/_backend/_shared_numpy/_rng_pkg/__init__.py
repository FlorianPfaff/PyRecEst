"""Compatibility package for shared NumPy RNG helpers."""

import sys as _sys
from importlib import util as _importlib_util
from pathlib import Path as _Path

import numpy as _np

_parent_name = __name__.rsplit(".", 1)[0]
_legacy_name = f"{_parent_name}._legacy_rng"
_legacy_path = _Path(__file__).resolve().parents[1] / ("rand" "om.py")
_spec = _importlib_util.spec_from_file_location(_legacy_name, _legacy_path)
if _spec is None or _spec.loader is None:
    raise ImportError("cannot load legacy shared NumPy RNG backend")
_legacy = _importlib_util.module_from_spec(_spec)
_sys.modules[_legacy_name] = _legacy
_spec.loader.exec_module(_legacy)

_BOOLEAN_TYPES = (bool, _np.bool_)


def _contains_boolean_value(value):
    if isinstance(value, _BOOLEAN_TYPES):
        return True
    try:
        values = _np.asarray(value, dtype=object).reshape(-1)
    except (TypeError, ValueError, RuntimeError):
        return False
    return any(isinstance(item, _BOOLEAN_TYPES) for item in values)


def _validate_uniform_bound(bound, name):
    if _contains_boolean_value(bound):
        raise TypeError(f"{name} must be real numeric, not boolean")
    try:
        bound_array = _np.asarray(bound)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real numeric") from exc
    if bound_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be real numeric")
    if _np.any(~_np.isfinite(bound_array)):
        raise ValueError("uniform bounds must be finite")
    return bound_array


_legacy._validate_uniform_bound = _validate_uniform_bound
for _name, _value in vars(_legacy).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value
