"""Torch based random backend."""

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


def _choice_population(a, p=None):
    """Normalize NumPy-compatible ``choice`` populations to a tensor."""
    device = p.device if _torch.is_tensor(p) else None
    if isinstance(a, int):
        if a <= 0:
            raise ValueError("a must be greater than 0 unless no samples are taken")
        return _torch.arange(a, device=device)
    if _torch.is_tensor(a):
        return a
    return _torch.as_tensor(a, device=device)


def choice(a, size=None, replace=True, p=None):
    a = _choice_population(a, p=p)
    size, num_samples = _choice_size(size)
    if p is not None:
        if not replace and num_samples > len(a):
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'."
            )

        p = _torch.as_tensor(p, dtype=_torch.float32, device=a.device)
        p = p / p.sum()  # Normalize probabilities
        indices = _torch.multinomial(p, num_samples=num_samples, replacement=replace)
        if size is None:
            indices = indices[0]
        else:
            indices = indices.reshape(size)
    elif replace:
        indices = _torch.randint(0, len(a), size or (), device=a.device)
    else:
        if num_samples > len(a):
            raise ValueError(
                "Cannot take a larger sample than population when 'replace=False'."
            )
        indices = _torch.randperm(len(a), device=a.device)[:num_samples]
        if size is None:
            indices = indices[0]
        else:
            indices = indices.reshape(size)

    return a[indices]


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
