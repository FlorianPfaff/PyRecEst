"""Torch based random backend."""

from numbers import Integral as _Integral

import torch as _torch
from torch import get_rng_state as get_state  # For PyRecEst
from torch import randint
from torch import set_rng_state as set_state  # For PyRecEst
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import _allow_complex_dtype, _modify_func_default_dtype


def _choice_size(size):
    if size is None:
        return None, 1
    if not hasattr(size, "__iter__"):
        size = (size,)
    size = tuple(int(dim) for dim in size)
    return size, int(_torch.prod(_torch.tensor(size)).item())


def _integer_population_size(a):
    if isinstance(a, _Integral):
        return int(a)
    if (
        _torch.is_tensor(a)
        and a.ndim == 0
        and not _torch.is_floating_point(a)
        and not _torch.is_complex(a)
    ):
        return int(a.item())
    return None


def _choice_indices(population_size, size, num_samples, replace, p, device):
    if population_size <= 0:
        if num_samples == 0:
            return _torch.empty(size or (0,), dtype=_torch.long, device=device)
        raise ValueError("a must be greater than 0 unless no samples are taken")

    if p is not None:
        if not replace and num_samples > population_size:
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'."
            )

        p = _torch.as_tensor(p, dtype=_torch.float32, device=device)
        if p.ndim != 1 or p.shape[0] != population_size:
            raise ValueError(
                "p must be 1-dimensional with one entry per population item"
            )

        p_sum = p.sum()
        if not bool(_torch.isfinite(p_sum)) or bool(p_sum <= 0):
            raise ValueError("probabilities do not sum to a positive value")
        p = p / p_sum
        indices = _torch.multinomial(p, num_samples=num_samples, replacement=replace)
        if size is None:
            return indices[0]
        return indices.reshape(size)

    if replace:
        return _torch.randint(0, population_size, size or (), device=device)

    if num_samples > population_size:
        raise ValueError(
            "Cannot take a larger sample than population when 'replace=False'."
        )

    indices = _torch.randperm(population_size, device=device)[:num_samples]
    if size is None:
        return indices[0]
    return indices.reshape(size)


def _take_choice(a, indices, axis):
    axis = axis % a.ndim
    if indices.ndim == 0:
        return a.select(axis, int(indices.item()))

    flattened_indices = indices.reshape(-1)
    selected = _torch.index_select(a, dim=axis, index=flattened_indices)
    return selected.reshape((*a.shape[:axis], *indices.shape, *a.shape[axis + 1 :]))


def choice(a, size=None, replace=True, p=None, axis=0, shuffle=True):
    del shuffle

    size, num_samples = _choice_size(size)
    population_size = _integer_population_size(a)
    if population_size is not None:
        device = p.device if _torch.is_tensor(p) else None
        return _choice_indices(population_size, size, num_samples, replace, p, device)

    if not _torch.is_tensor(a):
        a = _torch.as_tensor(a)
    if a.ndim == 0:
        raise ValueError(
            "a must be a positive integer or an array with at least one dimension"
        )

    axis = axis % a.ndim
    indices = _choice_indices(a.shape[axis], size, num_samples, replace, p, a.device)
    return _take_choice(a, indices, axis)


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


def rand(size=None, dtype=None):
    if size is None:
        size = ()
    elif not hasattr(size, "__iter__"):
        size = (size,)
    return _torch.rand(size, dtype=dtype)


def multinomial(n, pvals):
    pvals = pvals / pvals.sum()
    return _torch.multinomial(pvals, n, replacement=True).bincount(minlength=len(pvals))


@_allow_complex_dtype
def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    elif not hasattr(size, "__iter__"):
        size = (size,)
    return _torch.normal(mean=loc, std=scale, size=size)


def uniform(low=0.0, high=1.0, size=None, dtype=None):
    device = None
    if _torch.is_tensor(low):
        device = low.device
    elif _torch.is_tensor(high):
        device = high.device

    low = _torch.as_tensor(low, dtype=dtype, device=device)
    high = _torch.as_tensor(high, dtype=dtype, device=device)
    if bool(_torch.any(low >= high)):
        raise ValueError("Upper bound must be higher than lower bound")
    if size is None:
        size = ()
    elif not hasattr(size, "__iter__"):
        size = (size,)
    return (high - low) * _torch.rand(size, dtype=dtype, device=device) + low


@_modify_func_default_dtype(copy=False, kw_only=True)
@_allow_complex_dtype
def multivariate_normal(mean, cov, size=None):
    if size is None:
        size = ()
    elif not hasattr(size, "__iter__"):
        size = (size,)
    return _MultivariateNormal(mean, cov).sample(size)
