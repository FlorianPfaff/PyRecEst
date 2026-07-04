import autograd.numpy as _np
from pyrecest._backend._dtype_utils import (
    _dyn_update_dtype,
    _modify_func_default_dtype,
    get_default_cdtype,
    get_default_dtype,
)

from .._shared_numpy._common import (
    _add_default_dtype_by_casting,
    _allow_complex_dtype,
    _box_binary_scalar,
    _box_unary_scalar,
    _cast_fout_to_input_dtype,
    _cast_out_to_input_dtype,
    _get_wider_dtype,
    _is_boolean,
    _is_iterable,
    as_dtype,
    atol,
    cast,
    convert_to_wider_dtype,
    is_array,
    is_bool,
    is_complex,
    is_floating,
    rtol,
    set_default_dtype,
    to_ndarray,
)

zeros = _dyn_update_dtype(target=_np.zeros)
eye = _dyn_update_dtype(target=_np.eye)


def _array_default_dtype_for(result):
    if is_floating(result):
        return get_default_dtype()
    if is_complex(result):
        return get_default_cdtype()
    return None


def array(object, dtype=None, *args, **kwargs):
    result = _np.array(object, dtype=dtype, *args, **kwargs)
    if dtype is None:
        default_dtype = _array_default_dtype_for(result)
        if default_dtype is not None and result.dtype != default_dtype:
            return cast(result, default_dtype)
    return result
