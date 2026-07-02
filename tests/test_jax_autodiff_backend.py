import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from pyrecest._backend.jax import autodiff  # noqa: E402


def test_elementwise_grad_vector_output_returns_elementwise_derivative():
    grad_fn = autodiff.elementwise_grad(lambda x: x**2)

    result = grad_fn(jnp.array([1.0, 2.0, -3.0]))

    assert jnp.allclose(result, jnp.array([2.0, 4.0, -6.0]))


def test_elementwise_grad_scalar_output_returns_regular_gradient():
    grad_fn = autodiff.elementwise_grad(lambda x: jnp.sum(x**2))

    result = grad_fn(jnp.array([1.0, 2.0, -3.0]))

    assert jnp.allclose(result, jnp.array([2.0, 4.0, -6.0]))


def test_value_and_grad_respects_argnums_contract():
    value_grad_fn = autodiff.value_and_grad(lambda x, y: jnp.sum(x * y), argnums=1)

    value, grad = value_grad_fn(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))

    assert jnp.allclose(value, 11.0)
    assert jnp.allclose(grad, jnp.array([1.0, 2.0]))


def test_value_and_grad_accepts_keyword_arguments():
    value_grad_fn = autodiff.value_and_grad(lambda x, *, scale: jnp.sum(scale * x**2))

    value, grad = value_grad_fn(jnp.array([1.0, -2.0]), scale=3.0)

    assert jnp.allclose(value, 15.0)
    assert jnp.allclose(grad, jnp.array([6.0, -12.0]))


def test_custom_gradient_overrides_jax_default_derivative():
    @autodiff.custom_gradient(lambda x: 3.0 * jnp.ones_like(x))
    def squared(x):
        return x**2

    result = jax.grad(lambda x: jnp.sum(squared(x)))(jnp.array([2.0, 3.0]))

    assert jnp.allclose(result, jnp.array([3.0, 3.0]))


def test_custom_gradient_supports_multiple_arguments_and_keyword_calls():
    @autodiff.custom_gradient(
        lambda x, *, scale=1.0: scale * jnp.ones_like(x),
        lambda x, *, scale=1.0: -jnp.ones_like(scale),
    )
    def scaled_square(x, *, scale=1.0):
        return scale * jnp.sum(x**2)

    grad_x, grad_scale = jax.grad(
        lambda x, scale: scaled_square(x, scale=scale),
        argnums=(0, 1),
    )(jnp.array([2.0, 3.0]), 5.0)

    assert jnp.allclose(grad_x, jnp.array([5.0, 5.0]))
    assert jnp.allclose(grad_scale, -1.0)


def test_custom_gradient_contracts_vector_output_for_scalar_argument():
    @autodiff.custom_gradient(lambda x: jnp.array([2.0, 3.0]))
    def affine_pair(x):
        return jnp.array([2.0 * x, 3.0 * x])

    _, pullback = jax.vjp(affine_pair, 4.0)
    (gradient,) = pullback(jnp.array([10.0, 1.0]))

    assert jnp.allclose(gradient, 23.0)


def test_custom_gradient_contracts_full_jacobian_with_cotangent():
    jacobian_matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    @autodiff.custom_gradient(lambda x: jacobian_matrix)
    def linear_map(x):
        return jacobian_matrix @ x

    _, pullback = jax.vjp(linear_map, jnp.array([1.0, 1.0]))
    (gradient,) = pullback(jnp.array([5.0, 7.0]))

    assert jnp.allclose(gradient, jnp.array([26.0, 38.0]))


@pytest.mark.parametrize(
    "function_name",
    [
        "hessian",
        "hessian_vec",
        "jacobian_vec",
        "jacobian_and_hessian",
        "value_jacobian_and_hessian",
        "value_and_jacobian",
    ],
)
def test_unsupported_autodiff_functions_raise_not_implemented(function_name):
    with pytest.raises(NotImplementedError, match="not supported in this JAX backend"):
        getattr(autodiff, function_name)(lambda x: x)
