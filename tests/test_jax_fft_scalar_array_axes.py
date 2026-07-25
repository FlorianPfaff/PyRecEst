import numpy as np
import pytest


def test_jax_fft_scalar_axis_smoke():
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    values = jnp.arange(4.0)
    axis = jnp.array(0)
    expected = fft.rfft(values, axis=0)
    actual = fft.rfft(values, axis=axis)

    assert actual.shape == expected.shape
    assert actual.tolist() == expected.tolist()


def test_jax_fft_scalar_length_arrays_match_integer_length():
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    values = jnp.arange(4.0)
    expected = fft.rfft(values, n=3)

    for n in (np.array(3), jnp.array(3)):
        actual = fft.rfft(values, n=n)
        assert actual.shape == expected.shape
        assert actual.tolist() == expected.tolist()


def test_jax_fft_rejects_non_singleton_length_arrays():
    jnp = pytest.importorskip("jax.numpy")
    from pyrecest._backend.jax import fft

    values = jnp.arange(4.0)

    for n in (
        np.array([3, 4]),
        np.array([[3, 4]]),
        jnp.array([3, 4]),
        jnp.array([[3, 4]]),
    ):
        with pytest.raises(TypeError):
            fft.rfft(values, n=n)
