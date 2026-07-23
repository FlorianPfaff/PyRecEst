from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.sampling.hyperspherical_sampler import (
    HealpixHopfSampler,
    SphericalFibonacciSampler,
)


def _fake_healpy() -> ModuleType:
    module = ModuleType("healpy")
    module.nside2npix = lambda nside: 12 * int(nside) ** 2
    module.pix2ang = lambda nside, index, nest=True: (np.pi / 2.0, 0.0)
    return module


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="HealpixHopfSampler uses in-place grid construction on this backend",
)
def test_healpix_hopf_accepts_numpy_integer_scalar(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "healpy", _fake_healpy())

    grid, description = HealpixHopfSampler().get_grid(np.int64(0))

    assert grid.shape == (72, 4)
    assert description["layer-parameter"] == [0]


@pytest.mark.parametrize(
    "grid_density_parameter",
    (
        True,
        np.bool_(False),
        -1,
        0.5,
        np.nan,
        np.inf,
        "0",
        np.str_("0"),
        [],
        [0, 0],
        [0, True],
        [0, 4, 8],
    ),
)
def test_healpix_hopf_rejects_invalid_density_before_import(
    grid_density_parameter,
) -> None:
    with pytest.raises(ValueError, match="grid_density_parameter"):
        HealpixHopfSampler().get_grid(grid_density_parameter)


@pytest.mark.parametrize("grid_density_parameter", ("4", np.str_("4")))
def test_spherical_fibonacci_rejects_text_density(grid_density_parameter) -> None:
    with pytest.raises(ValueError, match="grid_density_parameter"):
        SphericalFibonacciSampler().get_grid_spherical_coordinates(
            grid_density_parameter
        )
