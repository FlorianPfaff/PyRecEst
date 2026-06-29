"""
Wrapper around jax functions to be consistent with backends.
Based on autodiff.py by emilemathieu on
https://github.com/oxcsml/geomstats/blob/master/geomstats/_backend/jax/autodiff.py
"""

import functools
import inspect

import jax
import jax.numpy as anp
from jax import grad, jacfwd
from jax import value_and_grad as _jax_value_and_grad


def detach(x):
    """Return a new tensor detached from the current graph.

    This is a placeholder in order to have consistent backend APIs.

    Parameters
    ----------
    x : array-like
        Tensor to detach.
    """
    return x


def elementwise_grad(func):
    """Wrap autograd elementwise_grad function.

    Parameters
    ----------
    func : callable
        Function for which the element-wise grad is computed.
    """

    def _elementwise_grad(*args, **kwargs):
        def _summed_func(*inner_args):
            return anp.sum(func(*inner_args, **kwargs))

        return grad(_summed_func)(*args)

    return _elementwise_grad


def _shape_tuple(value):
    """Return a plain Python shape tuple for array-like values."""
    return tuple(int(dim) for dim in getattr(value, "shape", ()))


def _apply_custom_gradient(cotangent, grads, argument):
    """Apply a user-supplied Jacobian/VJP factor to an upstream cotangent."""
    cotangent = anp.asarray(cotangent)
    grads = anp.asarray(grads)
    argument_shape = _shape_tuple(argument)
    cotangent_shape = _shape_tuple(cotangent)

    if argument_shape and grads.shape == argument_shape:
        return cotangent * grads

    if grads.shape == cotangent_shape + argument_shape:
        if cotangent.ndim == 0:
            return cotangent * grads
        return anp.tensordot(cotangent, grads, axes=cotangent.ndim)

    return cotangent * grads


def custom_gradient(*grad_funcs):
    """Decorate a function to define its custom gradient(s).

    Parameters
    ----------
    *grad_funcs : callables
        Custom gradient functions.
    """
    if len(grad_funcs) > 3:
        raise NotImplementedError(
            "custom_gradient is not yet implemented for more than 3 gradients."
        )
    if len(grad_funcs) == 0:
        raise NotImplementedError("custom_gradient requires at least one gradient.")

    def decorator(func):
        signature = inspect.signature(func)
        parameters = tuple(signature.parameters.values())

        def bind_values(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            return tuple(bound.arguments[param.name] for param in parameters)

        def call_with_bound_values(callable_, values):
            positional_args = []
            keyword_args = {}
            for param, value in zip(parameters, values, strict=True):
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    positional_args.append(value)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    positional_args.extend(value)
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    keyword_args[param.name] = value
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    keyword_args.update(value)
            return callable_(*positional_args, **keyword_args)

        @jax.custom_vjp
        def custom_vjp_call(*values):
            return call_with_bound_values(func, values)

        def forward(*values):
            return call_with_bound_values(func, values), values

        def backward(values, cotangent):
            gradients = []
            for arg_index in range(len(values)):
                if arg_index < len(grad_funcs):
                    grads = call_with_bound_values(grad_funcs[arg_index], values)
                    gradients.append(
                        _apply_custom_gradient(cotangent, grads, values[arg_index])
                    )
                else:
                    gradients.append(None)
            return tuple(gradients)

        custom_vjp_call.defvjp(forward, backward)

        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            return custom_vjp_call(*bind_values(*args, **kwargs))

        return wrapped_function

    return decorator


def jacobian(func):
    """Wrap autograd jacobian function."""
    return jacfwd(func)


def value_and_grad(func, argnums=0, point_ndims=1, to_numpy=False):
    """Wrap JAX ``value_and_grad`` with the shared autodiff backend contract.

    The autograd and PyTorch backends expose ``argnums`` and accept keyword
    arguments when evaluating the wrapped function.  JAX already supports those
    semantics natively; this wrapper forwards them and preserves the historical
    ``to_numpy`` argument for callers that used the JAX-only signature.
    """
    if isinstance(argnums, bool):
        to_numpy = bool(argnums)
        argnums = 0
    del point_ndims  # JAX value_and_grad is scalar-output only.

    def aux_value_and_grad(*args, **kwargs):
        def func_with_kwargs(*inner_args):
            return func(*inner_args, **kwargs)

        value, grads = _jax_value_and_grad(func_with_kwargs, argnums=argnums)(*args)
        if to_numpy:
            value = jax.device_get(value)
            grads = jax.tree_util.tree_map(jax.device_get, grads)
        return value, grads

    return aux_value_and_grad


unsupported_functions = [
    "hessian",
    "hessian_vec",
    "jacobian_vec",
    "jacobian_and_hessian",
    "value_jacobian_and_hessian",
    "value_and_jacobian",
]


def _raise_unsupported(*args, **kwargs):
    raise NotImplementedError("This function is not supported in this JAX backend.")


for func_name in unsupported_functions:
    globals()[func_name] = _raise_unsupported
