"""
Wrapper around jax functions to be consistent with backends.
Based on autodiff.py by emilemathieu on
https://github.com/oxcsml/geomstats/blob/master/geomstats/_backend/jax/autodiff.py
"""

import jax.numpy as anp
from autograd.extend import defvjp, primitive  # TODO: replace
from jax import grad, jacfwd
from jax import value_and_grad as _value_and_grad


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


def custom_gradient(*grad_funcs):
    """Decorate a function to define its custom gradient(s).

    Parameters
    ----------
    *grad_funcs : callables
        Custom gradient functions.
    """

    def decorator(func):
        wrapped_function = primitive(func)

        def wrapped_grad_func(i, ans, *args, **kwargs):
            grads = grad_funcs[i](*args, **kwargs)
            if isinstance(grads, float):
                return lambda g: g * grads
            if grads.ndim == 2:
                return lambda g: g[..., None] * grads
            if grads.ndim == 3:
                return lambda g: g[..., None, None] * grads
            return lambda g: g * grads

        if len(grad_funcs) == 1:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
            )
        elif len(grad_funcs) == 2:

            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(1, ans, *args, **kwargs),
            )
        elif len(grad_funcs) == 3:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(1, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(2, ans, *args, **kwargs),
            )
        else:
            raise NotImplementedError(
                "custom_gradient is not yet implemented " "for more than 3 gradients."
            )

        return wrapped_function

    return decorator


def jacobian(func):
    """Wrap autograd jacobian function."""
    return jacfwd(func)


def value_and_grad(func, to_numpy=False):
    """Wrap autograd value_and_grad function."""

    def aux_value_and_grad(*args):
        n_args = len(args)
        value = func(*args)

        all_grads = []
        for i in range(n_args):

            def func_of_ith(*args):
                reorg_args = args[1 : i + 1] + (args[0],) + args[i + 1 :]
                return func(*reorg_args)

            new_args = (args[i],) + args[:i] + args[i + 1 :]
            _, grad_i = _value_and_grad(func_of_ith)(*new_args)
            all_grads.append(grad_i)

        if n_args == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)

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
