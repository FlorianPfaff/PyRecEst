import numpy as np
import pytest
from pyrecest.filters import HypertoroidalParticleFilter


def test_constructor_accepts_numpy_integer_scalar_arrays():
    hpf = HypertoroidalParticleFilter(
        np.array(5, dtype=np.int64),
        np.array(2, dtype=np.int64),
    )

    assert hpf.filter_state.d.shape == (5, 2)


@pytest.mark.parametrize(
    ("argument_name", "n_particles", "dim"),
    [
        ("n_particles", np.array(True), 3),
        ("n_particles", np.array(0), 3),
        ("n_particles", np.array(-1), 3),
        ("n_particles", np.array(1.5), 3),
        ("dim", 5, np.array(True)),
        ("dim", 5, np.array(0)),
        ("dim", 5, np.array(-1)),
        ("dim", 5, np.array(1.5)),
    ],
)
def test_constructor_rejects_invalid_numpy_scalar_arrays(
    argument_name,
    n_particles,
    dim,
):
    with pytest.raises(ValueError, match=argument_name):
        HypertoroidalParticleFilter(n_particles, dim)
