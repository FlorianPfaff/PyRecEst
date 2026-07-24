"""Runtime patch for PyTorch quantiles with empty batch dimensions."""

from __future__ import annotations

_QUANTILE_METHOD_CONFLICT_MESSAGE = (
    "quantile() cannot specify both 'method' and 'interpolation'"
)


def patch_pytorch_quantile_empty_batch_contract() -> None:
    """Preserve NumPy quantile shapes unsupported by native PyTorch."""

    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.pytorch as raw_pytorch  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch may be unavailable
        return

    active_pytorch_backend = getattr(backend, "__backend_name__", None) == "pytorch"
    original_quantile = getattr(raw_pytorch, "quantile", None)
    if original_quantile is None:
        return
    if getattr(original_quantile, "_pyrecest_empty_batch_contract", False):
        if active_pytorch_backend:
            backend.quantile = original_quantile
        return

    def quantile(
        a,
        q,
        axis=None,
        out=None,
        overwrite_input=False,
        method="linear",
        keepdims=False,
        *,
        dim=None,
        keepdim=None,
        interpolation=None,
    ):
        effective_axis = axis
        if dim is not None:
            if axis is not None and axis != dim:
                raise TypeError("quantile() got both 'axis' and 'dim'")
            effective_axis = dim

        effective_keepdims = keepdims
        if keepdim is not None:
            if keepdims is not False and keepdims != keepdim:
                raise TypeError("quantile() got both 'keepdims' and 'keepdim'")
            effective_keepdims = keepdim

        if interpolation is not None and method != "linear":
            raise TypeError(_QUANTILE_METHOD_CONFLICT_MESSAGE)
        effective_method = method if interpolation is None else interpolation
        values = raw_pytorch.array(a)
        q_shape = raw_pytorch._quantile_q_shape(q)
        is_integral_axis = isinstance(
            effective_axis, (int, np.integer)
        ) and not isinstance(effective_axis, (bool, np.bool_))
        if is_integral_axis and values.numel() == 0:
            normalized_axis = int(effective_axis)
            if normalized_axis < 0:
                normalized_axis += values.ndim
            if (
                0 <= normalized_axis < values.ndim
                and values.shape[normalized_axis] > 0
                and not raw_pytorch.is_complex(values)
            ):
                if not raw_pytorch.is_floating(values):
                    values = raw_pytorch.cast(
                        values, dtype=raw_pytorch.get_default_dtype()
                    )
                q_arg = raw_pytorch._quantile_q(q, values)
                if len(q_shape) > 1:
                    q_arg = q_arg.reshape(-1)
                validation_values = torch.zeros(
                    values.shape[normalized_axis],
                    dtype=values.dtype,
                    device=values.device,
                )
                torch.quantile(
                    validation_values,
                    q_arg,
                    dim=0,
                    interpolation=effective_method,
                )
                result = torch.sum(
                    values,
                    dim=normalized_axis,
                    keepdim=effective_keepdims,
                )
                if q_shape:
                    result = torch.broadcast_to(result, q_shape + tuple(result.shape))
                if out is not None:
                    out.copy_(result)
                    return out
                return result

        if len(q_shape) > 1:
            quantile_values = values
            if not raw_pytorch.is_floating(
                quantile_values
            ) and not raw_pytorch.is_complex(quantile_values):
                quantile_values = raw_pytorch.cast(
                    quantile_values,
                    dtype=raw_pytorch.get_default_dtype(),
                )
            q_arg = raw_pytorch._quantile_q(q, quantile_values).reshape(-1)
            result = original_quantile(
                a,
                q_arg,
                axis=axis,
                out=None,
                overwrite_input=overwrite_input,
                method=method,
                keepdims=keepdims,
                dim=dim,
                keepdim=keepdim,
                interpolation=interpolation,
            )
            result = result.reshape(q_shape + tuple(result.shape[1:]))
            if out is not None:
                out.copy_(result)
                return out
            return result

        return original_quantile(
            a,
            q,
            axis=axis,
            out=out,
            overwrite_input=overwrite_input,
            method=method,
            keepdims=keepdims,
            dim=dim,
            keepdim=keepdim,
            interpolation=interpolation,
        )

    quantile.__name__ = getattr(original_quantile, "__name__", "quantile")
    quantile.__doc__ = getattr(original_quantile, "__doc__", None)
    quantile._pyrecest_empty_batch_contract = True
    raw_pytorch.quantile = quantile
    if active_pytorch_backend:
        backend.quantile = quantile
