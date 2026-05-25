"""Execution backends.

Lead authors: Johan Mathe and Niklas Koep.
"""

import importlib
import importlib.abc
import importlib.machinery
import logging
import os
import sys
import types
from functools import wraps

import pyrecest._backend._common as common


def get_backend_name():
    return os.environ.get("PYRECEST_BACKEND", "numpy")


BACKEND_NAME = get_backend_name()


BACKEND_ATTRIBUTES = {
    "": [
        # Types
        "int32",
        "int64",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "uint8",
        # Functions
        "abs",
        "all",
        "allclose",
        "amax",
        "amin",
        "angle",
        "any",
        "arange",
        "arccos",
        "arccosh",
        "arcsin",
        "arctan2",
        "arctanh",
        "argmax",
        "argmin",
        "array",
        "array_from_sparse",
        "asarray",
        "as_dtype",
        "assignment",
        "assignment_by_sum",
        "atol",
        "broadcast_arrays",
        "broadcast_to",
        "cast",
        "ceil",
        "clip",
        "comb",
        "concatenate",
        "conj",
        "convert_to_wider_dtype",
        "copy",
        "cos",
        "cosh",
        "cross",
        "cumprod",
        "cumsum",
        "diag_indices",
        "diagonal",
        "divide",
        "dot",
        "einsum",
        "empty",
        "empty_like",
        "equal",
        "erf",
        "exp",
        "expand_dims",
        "eye",
        "flatten",
        "flip",
        "floor",
        "from_numpy",
        "gamma",
        "get_default_dtype",
        "get_default_cdtype",
        "get_slice",
        "greater",
        "has_autodiff",
        "hsplit",
        "hstack",
        "imag",
        "isclose",
        "isnan",
        "isscalar",
        "is_array",
        "is_complex",
        "is_floating",
        "is_bool",
        "kron",
        "less",
        "less_equal",
        "linspace",
        "log",
        "logical_and",
        "logical_or",
        "mat_from_diag_triu_tril",
        "matmul",
        "matvec",
        "maximum",
        "mean",
        "meshgrid",
        "minimum",
        "mod",
        "moveaxis",
        "ndim",
        "one_hot",
        "ones",
        "ones_like",
        "outer",
        "pad",
        "pi",
        "polygamma",
        "power",
        "prod",
        "quantile",
        "ravel_tril_indices",
        "real",
        "repeat",
        "reshape",
        "rtol",
        "scatter_add",
        "searchsorted",
        "set_default_dtype",
        "set_diag",
        "shape",
        "size",
        "sign",
        "sin",
        "sinh",
        "split",
        "sqrt",
        "squeeze",
        "sort",
        "stack",
        "std",
        "sum",
        "take",
        "tan",
        "tanh",
        "tile",
        "to_numpy",
        "to_ndarray",
        "trace",
        "transpose",
        "tril",
        "triu",
        "tril_indices",
        "triu_indices",
        "tril_to_vec",
        "triu_to_vec",
        "vec_to_diag",
        "unique",
        "vectorize",
        "vstack",
        "where",
        "zeros",
        "zeros_like",
        "trapezoid",  # Changed from trapz to trapezoid from scipy.integrate
        # The ones below are for pyrecest
        "diag",
        "diff",
        "apply_along_axis",
        "nonzero",
        "column_stack",
        "conj",
        "atleast_1d",
        "atleast_2d",
        "dstack",
        "full",
        "isreal",
        "triu",
        "kron",
        "angle",
        "arctan",
        "cov",
        "count_nonzero",
        "full_like",
        "isinf",
        "isfinite",
        "deg2rad",
        "rad2deg",
        "argsort",
        "max",
        "min",
        "roll",
        "dstack",
        "vmap",
        "gammaln",
        "round",
        "array_equal",
        # For Riemannian score-based SDE
        "log1p",
    ],
    "autodiff": [
        "custom_gradient",
        "hessian",
        "hessian_vec",
        "jacobian",
        "jacobian_vec",
        "jacobian_and_hessian",
        "value_and_grad",
        "value_and_jacobian",
        "value_jacobian_and_hessian",
    ],
    "linalg": [
        "cholesky",
        "det",
        "eig",
        "eigh",
        "eigvalsh",
        "expm",
        "fractional_matrix_power",
        "inv",
        "is_single_matrix_pd",
        "logm",
        "matrix_power",
        "norm",
        "qr",
        "quadratic_assignment",
        "polar",
        "solve",
        "solve_sylvester",
        "sqrtm",
        "svd",
        "matrix_rank",
        "block_diag",  # For PyRecEst
        "pinv",
    ],
    "random": [
        "choice",
        "normal",
        "multinomial",
        "multivariate_normal",
        # TODO (nkoep): Remove 'rand' and replace it by 'uniform'. Much like
        #              'randn' is a convenience wrapper (which we don't use)
        #              for 'normal', 'rand' only wraps 'uniform'.
        "rand",
        "randint",
        "seed",
        "uniform",
        # For PyRecEst
        "get_state",
        "set_state",
    ],
    "fft": [  # For PyRecEst
        "rfft",
        "irfft",
        "fftshift",
        "ifftshift",
        "fftn",
        "ifftn",
    ],
    "spatial": [  # For PyRecEst
        "Rotation",
    ],
    "signal": [  # For PyRecEst
        "fftconvolve",
    ],
}

OPTIONAL_BACKEND_ATTRIBUTES = {
    "random": [
        "create_random_state",
    ],
}


def _deduplicated_attributes(attributes):
    """Return ``attributes`` with duplicates removed while preserving order."""
    return list(dict.fromkeys(attributes))


for _module_name, _attributes in BACKEND_ATTRIBUTES.items():
    BACKEND_ATTRIBUTES[_module_name] = _deduplicated_attributes(_attributes)


def _quantile_with_numpy_axis(quantile_func, asarray_func):
    """Return a NumPy-compatible quantile wrapper for stricter backends."""

    @wraps(quantile_func)
    def quantile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=False,
        *,
        interpolation=None,
    ):
        del overwrite_input
        if interpolation is not None:
            method = interpolation

        kwargs = {"dim": axis, "keepdim": keepdims, "interpolation": method}
        if out is not None:
            kwargs["out"] = out
        return quantile_func(asarray_func(a), asarray_func(q), **kwargs)

    return quantile


def _meshgrid_with_arraylike_axes(meshgrid_func, asarray_func, atleast_1d_func):
    """Return a NumPy-compatible meshgrid wrapper for stricter backends."""

    @wraps(meshgrid_func)
    def meshgrid(*axes, **kwargs):
        coerced_axes = [atleast_1d_func(asarray_func(axis)) for axis in axes]
        return meshgrid_func(*coerced_axes, **kwargs)

    return meshgrid


def _mean_with_numpy_signature(mean_func, asarray_func, reshape_func):
    """Return a NumPy-compatible mean wrapper for stricter backends."""

    @wraps(mean_func)
    def mean(a, axis=None, dtype=None, out=None, keepdims=False):
        a = asarray_func(a, dtype=dtype) if dtype is not None else asarray_func(a)
        if axis is None:
            result = mean_func(a)
            if keepdims:
                result = reshape_func(result, (1,) * a.ndim)
        else:
            result = mean_func(a, dim=axis, keepdim=keepdims)

        if out is not None:
            out[...] = result
            return out
        return result

    return mean


def _normalize_reduction_axes(axis, ndim):
    """Return normalized reduction axes for NumPy-style ``axis`` arguments."""
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    normalized_axes = tuple(
        axis_index + ndim if axis_index < 0 else axis_index for axis_index in axes
    )
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError("duplicate value in 'axis'")

    for original_axis, normalized_axis in zip(axes, normalized_axes):
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise IndexError(
                f"axis {original_axis} is out of bounds for array of dimension {ndim}"
            )

    return normalized_axes


def _reduced_keepdims_shape(shape, axes):
    """Return the shape produced by a keepdims reduction over ``axes``."""
    return tuple(1 if dim in axes else dim_size for dim, dim_size in enumerate(shape))


def _reduction_with_numpy_keepdims(
    reduction_func,
    asarray_func,
    reshape_func,
    *,
    cast_func=None,
):
    """Return a reduction wrapper accepting NumPy's ``keepdims`` keyword."""

    @wraps(reduction_func)
    def reduction(a, axis=None, dtype=None, out=None, keepdims=False):
        a = asarray_func(a)
        if dtype is not None:
            if cast_func is None:
                raise TypeError("dtype is not supported for this reduction")
            a = cast_func(a, dtype=dtype)

        axes = (
            tuple(range(a.ndim))
            if axis is None
            else _normalize_reduction_axes(axis, a.ndim)
        )
        result = a if axis is not None and not axes else reduction_func(a, axis=axis)
        if keepdims:
            result = reshape_func(result, _reduced_keepdims_shape(tuple(a.shape), axes))

        if out is not None:
            out[...] = result
            return out
        return result

    return reduction


def _sum_with_numpy_signature(sum_func, asarray_func, reshape_func):
    """Return a NumPy-compatible sum wrapper for stricter backends."""

    @wraps(sum_func)
    def sum(a, axis=None, dtype=None, out=None, keepdims=False):
        a = asarray_func(a)
        if axis is None:
            result = sum_func(a, dtype=dtype)
            if keepdims:
                result = reshape_func(result, (1,) * a.ndim)
        else:
            result = sum_func(a, axis=axis, keepdims=keepdims, dtype=dtype)

        if out is not None:
            out[...] = result
            return out
        return result

    return sum


def _std_with_numpy_input(
    std_func,
    asarray_func,
    cast_func,
    get_default_dtype_func,
    is_complex_func,
    is_floating_func,
):
    """Return a NumPy-compatible std wrapper for stricter backends."""

    @wraps(std_func)
    def std(
        a,
        axis=None,
        dtype=None,
        out=None,
        ddof=0,
        keepdims=False,
        *,
        correction=0,
    ):
        a = asarray_func(a)
        if dtype is not None:
            a = cast_func(a, dtype=dtype)
        elif not is_floating_func(a) and not is_complex_func(a):
            a = cast_func(a, dtype=get_default_dtype_func())
        kwargs = {
            "axis": axis,
            "dtype": None,
            "out": out,
            "ddof": ddof,
            "keepdims": keepdims,
        }
        if correction != 0:
            kwargs["correction"] = correction
        return std_func(a, **kwargs)

    return std


def _arg_reduction_with_numpy_signature(arg_func, asarray_func, reshape_func):
    """Return a NumPy-compatible argmin/argmax wrapper for stricter backends."""

    @wraps(arg_func)
    def arg_reduction(
        a,
        axis=None,
        out=None,
        keepdims=False,
        *,
        dim=None,
        keepdim=None,
    ):
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError(f"{arg_func.__name__}() got both 'axis' and 'dim'")
            axis = dim
        if keepdim is not None:
            if keepdims is not False and keepdims != keepdim:
                raise TypeError(
                    f"{arg_func.__name__}() got both 'keepdims' and 'keepdim'"
                )
            keepdims = keepdim

        a = asarray_func(a)
        try:
            import torch as _torch
        except ModuleNotFoundError:  # pragma: no cover - only relevant for PyTorch
            pass
        else:
            if _torch.is_tensor(a) and a.dtype == _torch.bool:
                a = a.to(dtype=_torch.uint8)

        if axis is None:
            result = arg_func(a)
            if keepdims:
                result = reshape_func(result, (1,) * a.ndim)
        else:
            result = arg_func(a, dim=axis, keepdim=keepdims)

        if out is not None:
            out[...] = result
            return out
        return result

    return arg_reduction


def _is_empty_assignment_index(indices):
    """Return whether ``indices`` selects no elements for assignment helpers."""
    if isinstance(indices, list):
        return len(indices) == 0
    if isinstance(indices, tuple):
        return False

    ndim = getattr(indices, "ndim", None)
    shape = getattr(indices, "shape", None)
    return ndim is not None and ndim > 0 and shape is not None and shape[0] == 0


def _assignment_with_empty_indices_noop(assignment_func, copy_func):
    """Return an assignment wrapper that treats empty indices as a no-op."""

    @wraps(assignment_func)
    def assignment(x, values, indices, axis=0):
        if _is_empty_assignment_index(indices):
            return copy_func(x)
        return assignment_func(x, values, indices, axis=axis)

    return assignment


class BackendImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """
    Meta path finder and loader for dynamically creating backend modules.

    Implements the modern PEP 451 import protocol (create_module / exec_module).

    Responsible for intercepting imports of 'pyrecest.backend' and redirecting
    them to dynamically constructed backend implementations (e.g. numpy, torch).
    """

    def __init__(self, path: str):
        self._path = path

    @staticmethod
    def _import_backend(backend_name: str):
        try:
            return importlib.import_module(f"pyrecest._backend.{backend_name}")
        except ModuleNotFoundError as e:
            raise RuntimeError(f"Unknown backend '{backend_name}'") from e

    def _create_backend_module(self, backend_name: str):
        backend = self._import_backend(backend_name)

        new_module = types.ModuleType(self._path)
        new_module.__file__ = getattr(backend, "__file__", None)

        # expose chosen backend
        new_module.__backend_name__ = backend_name
        new_module.BACKEND_NAME = backend_name
        new_module.get_backend_name = staticmethod(lambda: backend_name)

        for module_name, attributes in BACKEND_ATTRIBUTES.items():
            if module_name:
                try:
                    submodule = getattr(backend, module_name)
                except AttributeError:
                    raise RuntimeError(
                        f"Backend '{backend_name}' exposes no '{module_name}' module"
                    ) from None
                new_submodule = types.ModuleType(f"{self._path}.{module_name}")
                new_submodule.__file__ = getattr(submodule, "__file__", None)
                setattr(new_module, module_name, new_submodule)
            else:
                submodule = backend
                new_submodule = new_module

            for attribute_name in attributes:
                try:
                    submodule_ = submodule
                    if module_name == "" and not hasattr(submodule, attribute_name):
                        submodule_ = common
                    attribute = getattr(submodule_, attribute_name)
                except AttributeError:
                    if module_name:
                        raise RuntimeError(
                            f"Module '{module_name}' of backend '{backend_name}' does not define the required attribute '{attribute_name}'."
                        ) from None
                    else:
                        raise RuntimeError(
                            f"Backend '{backend_name}' does not define the required attribute '{attribute_name}'."
                        ) from None
                else:
                    if (
                        module_name == ""
                        and attribute_name == "mean"
                        and backend_name == "pytorch"
                    ):
                        attribute = _mean_with_numpy_signature(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "reshape"),
                        )
                    if (
                        module_name == ""
                        and attribute_name
                        in {"all", "amax", "amin", "any", "max", "min"}
                        and backend_name == "pytorch"
                    ):
                        attribute = _reduction_with_numpy_keepdims(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "reshape"),
                        )
                    if (
                        module_name == ""
                        and attribute_name == "sum"
                        and backend_name == "pytorch"
                    ):
                        attribute = _sum_with_numpy_signature(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "reshape"),
                        )
                    if (
                        module_name == ""
                        and attribute_name == "prod"
                        and backend_name == "pytorch"
                    ):
                        attribute = _reduction_with_numpy_keepdims(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "reshape"),
                            cast_func=getattr(backend, "cast"),
                        )
                    if (
                        module_name == ""
                        and attribute_name == "std"
                        and backend_name in {"pytorch", "jax"}
                    ):
                        get_default_dtype = (
                            (lambda: getattr(backend, "asarray")(0.0).dtype)
                            if backend_name == "jax"
                            else getattr(backend, "get_default_dtype")
                        )
                        attribute = _std_with_numpy_input(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "cast"),
                            get_default_dtype,
                            getattr(backend, "is_complex"),
                            getattr(backend, "is_floating"),
                        )
                    if (
                        module_name == ""
                        and attribute_name in {"argmax", "argmin"}
                        and backend_name == "pytorch"
                    ):
                        attribute = _arg_reduction_with_numpy_signature(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "reshape"),
                        )
                    if (
                        module_name == ""
                        and attribute_name == "meshgrid"
                        and backend_name in {"jax", "pytorch"}
                    ):
                        attribute = _meshgrid_with_arraylike_axes(
                            attribute,
                            getattr(backend, "asarray"),
                            getattr(backend, "atleast_1d"),
                        )
                    if (
                        module_name == ""
                        and attribute_name == "quantile"
                        and backend_name == "pytorch"
                    ):
                        attribute = _quantile_with_numpy_axis(
                            attribute,
                            getattr(backend, "asarray"),
                        )
                    if module_name == "" and attribute_name in {
                        "assignment",
                        "assignment_by_sum",
                    }:
                        attribute = _assignment_with_empty_indices_noop(
                            attribute,
                            getattr(backend, "copy"),
                        )
                    setattr(new_submodule, attribute_name, attribute)

            for attribute_name in OPTIONAL_BACKEND_ATTRIBUTES.get(module_name, []):
                if hasattr(submodule, attribute_name):
                    setattr(
                        new_submodule,
                        attribute_name,
                        getattr(submodule, attribute_name),
                    )

        return new_module

    def find_spec(self, fullname, path=None, target=None):
        """Find a module spec for the dynamically created backend."""
        if fullname != self._path:
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec):
        """Create the module object but don’t execute it yet."""
        module = self._create_backend_module(BACKEND_NAME)
        module.__loader__ = self
        module.__spec__ = spec
        return module

    def exec_module(self, module):
        """Execute the module (initialize attributes, types, etc.)."""
        if hasattr(module, "set_default_dtype"):
            module.set_default_dtype("float64")
        logging.info(f"Using {BACKEND_NAME} backend")


TARGET = "pyrecest.backend"
if not any(
    isinstance(f, BackendImporter) and getattr(f, "_path", None) == TARGET
    for f in sys.meta_path
):
    # put it in front so it intercepts 'pyrecest.backend'
    sys.meta_path.insert(0, BackendImporter(TARGET))
