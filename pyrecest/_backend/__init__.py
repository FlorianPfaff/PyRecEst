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
        "deg2rad",
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
        "log1p"
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
                            f"Module '{module_name}' of backend '{backend_name}' "
                            f"does not define the required attribute '{attribute_name}'."
                        ) from None
                    else:
                        raise RuntimeError(
                            f"Backend '{backend_name}' does not define the required "
                            f"attribute '{attribute_name}'."
                        ) from None
                else:
                    setattr(new_submodule, attribute_name, attribute)

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
if not any(isinstance(f, BackendImporter) and getattr(f, "_path", None) == TARGET
           for f in sys.meta_path):
    # put it in front so it intercepts 'pyrecest.backend'
    sys.meta_path.insert(0, BackendImporter(TARGET))