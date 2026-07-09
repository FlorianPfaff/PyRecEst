"""PyTorch linear-algebra tolerance validation contract patch."""

from __future__ import annotations


def patch_pytorch_linalg_tolerance_contract() -> None:
    """Reject temporal NumPy tolerance values before scalar unwrapping."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    linalg = getattr(raw_pytorch, "linalg", None)
    original_normalizer = getattr(linalg, "_normalize_linalg_tolerance", None)
    if original_normalizer is None:
        return
    if getattr(original_normalizer, "_pyrecest_temporal_tolerance_contract", False):
        return

    temporal_types = (np.datetime64, np.timedelta64)
    error_message = "linalg tolerances must be real numeric"

    def _has_temporal_dtype(value) -> bool:
        try:
            return np.asarray(value).dtype.kind in {"M", "m"}
        except (TypeError, ValueError, RuntimeError):
            return False

    def _contains_temporal_values(value) -> bool:
        if isinstance(value, temporal_types):
            return True
        try:
            values = np.asarray(value, dtype=object).reshape(-1)
        except (TypeError, ValueError, RuntimeError):
            return False
        return any(isinstance(item, temporal_types) for item in values)

    def _normalize_linalg_tolerance(value, reference=None):
        if value is not None and not torch.is_tensor(value):
            if _has_temporal_dtype(value) or _contains_temporal_values(value):
                raise TypeError(error_message)
        return original_normalizer(value, reference)

    _normalize_linalg_tolerance.__name__ = getattr(
        original_normalizer,
        "__name__",
        "_normalize_linalg_tolerance",
    )
    _normalize_linalg_tolerance.__doc__ = getattr(original_normalizer, "__doc__", None)
    _normalize_linalg_tolerance._pyrecest_temporal_tolerance_contract = True
    linalg._normalize_linalg_tolerance = _normalize_linalg_tolerance
