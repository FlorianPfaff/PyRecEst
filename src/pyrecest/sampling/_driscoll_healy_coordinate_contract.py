"""Coordinate-convention correction for Driscoll-Healy spherical grids."""

from __future__ import annotations

from functools import wraps

from pyrecest.backend import pi

from . import hyperspherical_sampler as _implementation

_MARKER = "_pyrecest_driscoll_healy_colatitude_contract"


def install_driscoll_healy_coordinate_contract() -> None:
    """Convert SHTOOLS latitude rows to PyRecEst colatitudes."""

    sampler_type = _implementation.DriscollHealySampler
    current = sampler_type.get_grid_spherical_coordinates
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def corrected(self, grid_density_parameter):
        azimuths, latitudes, description = current(self, grid_density_parameter)
        colatitudes = pi / 2.0 - latitudes
        return azimuths, colatitudes, description

    setattr(corrected, _MARKER, True)
    sampler_type.get_grid_spherical_coordinates = corrected
