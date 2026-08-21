"""Compatibility wrapper for Leopardi equal-area sampling utilities."""

from __future__ import annotations

import runpy
from pathlib import Path

_module_globals = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "leopardi_sampler.py"),
    run_name=__name__,
)
_original_get_equal_area_caps = _module_globals["get_equal_area_caps"]
_legacy_globals = _original_get_equal_area_caps.__globals__


def get_equal_area_caps(dim, N, symmetric: bool = False):
    """Return equal-area caps while preserving even symmetric collar counts."""
    if not symmetric or dim == 1 or N <= 2:
        return _original_get_equal_area_caps(dim, N, symmetric=symmetric)

    get_polar_cap_colatitude = _legacy_globals["get_polar_cap_colatitude"]
    get_ideal_collar_angle = _legacy_globals["get_ideal_collar_angle"]
    get_ideal_region_counts = _legacy_globals["get_ideal_region_counts"]
    round_region_counts = _legacy_globals["round_region_counts"]
    get_cap_colatitudes = _legacy_globals["get_cap_colatitudes"]

    c_polar = get_polar_cap_colatitude(dim, N)
    ideal_angle = get_ideal_collar_angle(dim, N)
    if not bool(ideal_angle > 0):
        return _original_get_equal_area_caps(dim, N, symmetric=symmetric)

    ratio_half = 0.5 * (_legacy_globals["pi"] - 2 * c_polar) / ideal_angle
    # A symmetric partition needs at least one collar in each hemisphere.
    n_half = _legacy_globals["max"](
        _legacy_globals["array"](
            (1, _legacy_globals["round"](ratio_half)),
            dtype=_legacy_globals["int32"],
        )
    )
    n_collars = int(2 * n_half)

    region_counts = get_ideal_region_counts(dim, N, c_polar, n_collars)
    n_regions = round_region_counts(region_counts)
    cap_colatitudes = get_cap_colatitudes(dim, N, c_polar, n_regions)
    return cap_colatitudes, n_regions


# Legacy functions resolve global names through their original execution dictionary.
# Patch that dictionary so internal calls use the corrected implementation as well.
_legacy_globals["get_equal_area_caps"] = get_equal_area_caps
_module_globals["get_equal_area_caps"] = get_equal_area_caps
for _name, _value in _module_globals.items():
    if not _name.startswith("__"):
        globals()[_name] = _value
