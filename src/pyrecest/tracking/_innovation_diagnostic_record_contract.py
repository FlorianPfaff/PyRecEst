"""Round-trip contract for serialized innovation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from . import innovation_diagnostics as _diagnostics

# pylint: disable=protected-access

_ORIGINAL_AS_FINITE_REAL_ARRAY = _diagnostics._as_finite_real_array


def _record_value(
    record: Mapping[str, Any], preferred_key: str, canonical_key: str
) -> Any:
    """Return a configured record field, falling back to its canonical name."""

    if preferred_key in record:
        return record[preferred_key]
    return record.get(canonical_key)


def _contains_masked_value(value: Any) -> bool:
    """Return whether ``value`` contains genuinely masked NumPy entries."""

    if np.ma.is_masked(value):
        return True
    if np.ma.isMaskedArray(value):
        if bool(np.any(np.ma.getmaskarray(value))):
            return True
        value = np.ma.getdata(value)
    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return False
        return any(_contains_masked_value(item) for item in value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return any(_contains_masked_value(item) for item in value)
    return False


def _as_finite_real_array(value: Any, name: str) -> np.ndarray:
    """Validate finite real arrays without discarding mask metadata."""

    if _contains_masked_value(value):
        raise ValueError(f"{name} must contain finite real numeric values")
    return _ORIGINAL_AS_FINITE_REAL_ARRAY(value, name)


def _optional_finite_real_array(value: Any, name: str) -> np.ndarray | None:
    """Deserialize an optional diagnostic array."""

    if value is None:
        return None
    return _diagnostics._as_finite_real_array(value, name).copy()


def diagnostic_from_record(
    record: Mapping[str, Any],
    *,
    source_key: str = "source",
    time_key: str = "time_s",
    action_key: str = "update_action",
    accepted_key: str = "accepted",
    nis_key: str = "nis",
    residual_norm_key: str = "residual_norm_m",
    measurement_dim_key: str = "measurement_dim",
    gate_threshold_key: str = "gate_threshold",
) -> _diagnostics.InnovationDiagnostic:
    """Restore a diagnostic from legacy records or ``InnovationDiagnostic.to_dict``."""

    residual = _optional_finite_real_array(record.get("residual"), "residual")
    innovation_covariance = _optional_finite_real_array(
        record.get("innovation_covariance"),
        "innovation_covariance",
    )

    measurement_dim_value = _record_value(
        record,
        measurement_dim_key,
        "measurement_dim",
    )
    if measurement_dim_value is None:
        measurement_dim_value = 0 if residual is None else residual.reshape(-1).size
    measurement_dim = _diagnostics._nonnegative_integer(
        measurement_dim_value,
        measurement_dim_key,
    )

    raw_metadata = record.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, Any] = {}
    elif isinstance(raw_metadata, Mapping):
        metadata = dict(raw_metadata)
    else:
        # Preserve permissive legacy behavior for non-mapping metadata values.
        metadata = {"metadata": raw_metadata}

    excluded = {
        source_key,
        time_key,
        action_key,
        accepted_key,
        nis_key,
        residual_norm_key,
        measurement_dim_key,
        gate_threshold_key,
        "source",
        "time",
        "action",
        "accepted",
        "nis",
        "residual_norm",
        "measurement_dim",
        "gate_threshold",
        "residual",
        "innovation_covariance",
        "metadata",
        "mahalanobis_distance",
    }
    metadata.update(
        {key: value for key, value in record.items() if key not in excluded}
    )

    action = _record_value(record, action_key, "action")
    source = _record_value(record, source_key, "source")
    return _diagnostics.InnovationDiagnostic(
        measurement_dim=measurement_dim,
        nis=_diagnostics._optional_float(_record_value(record, nis_key, "nis")),
        residual_norm=_diagnostics._optional_float(
            _record_value(record, residual_norm_key, "residual_norm")
        ),
        gate_threshold=_diagnostics._optional_float(
            _record_value(record, gate_threshold_key, "gate_threshold")
        ),
        accepted=_diagnostics._optional_bool(
            _record_value(record, accepted_key, "accepted")
        ),
        action=None if action is None else str(action),
        source=None if source is None else str(source),
        time=_diagnostics._optional_float(_record_value(record, time_key, "time")),
        residual=None if residual is None else residual.reshape(-1),
        innovation_covariance=innovation_covariance,
        metadata=metadata,
    )


def install_innovation_diagnostic_record_contract() -> None:
    """Install round-trip-safe record and array validation contracts."""

    record_marker = "_pyrecest_round_trip_record_contract"
    setattr(diagnostic_from_record, record_marker, True)
    current_record_loader = _diagnostics.diagnostic_from_record
    if not getattr(current_record_loader, record_marker, False):
        _diagnostics.diagnostic_from_record = diagnostic_from_record

    array_marker = "_pyrecest_mask_preserving_array_contract"
    setattr(_as_finite_real_array, array_marker, True)
    current_array_validator = _diagnostics._as_finite_real_array
    if not getattr(current_array_validator, array_marker, False):
        _diagnostics._as_finite_real_array = _as_finite_real_array


__all__ = ["install_innovation_diagnostic_record_contract"]
