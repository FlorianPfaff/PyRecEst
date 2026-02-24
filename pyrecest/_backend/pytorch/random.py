"""Torch based random backend."""

import torch as _torch
from torch import rand, randint
from torch import get_rng_state as get_state  # For PyRecEst
from torch import set_rng_state as set_state  # For PyRecEst
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import _allow_complex_dtype, _modify_func_default_dtype


def choice(a, size=None, replace=True, p=None):
    assert _torch.is_tensor(a), "a must be a tensor"
    if p is not None:
        assert _torch.is_tensor(p), "p must be a tensor"
        if not replace:
            raise ValueError("Sampling without replacement is not supported with PyTorch when probabilities are given.")
        
        p = _torch.tensor(p, dtype=_torch.float32)
        p = p / p.sum()  # Normalize probabilities
        indices = _torch.multinomial(p, num_samples=_torch.prod(_torch.tensor(size)), replacement=True)
    else:
        indices = _torch.randint(0, len(a), size)
    
    return a[indices]


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


def multinomial(n, pvals):
    pvals = pvals / pvals.sum()
    return _torch.multinomial(pvals, n, replacement=True).bincount(minlength=len(pvals))


@_allow_complex_dtype
def normal(loc=0.0, scale=1.0, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _torch.normal(mean=loc, std=scale, size=size)


def uniform(low=0.0, high=1.0, size=(1,), dtype=None):
    if not hasattr(size, "__iter__"):
        size = (size,)
    if low >= high:
        raise ValueError("Upper bound must be higher than lower bound")
    return (high - low) * rand(*size, dtype=dtype) + low


@_modify_func_default_dtype(copy=False, kw_only=True)
@_allow_complex_dtype
def multivariate_normal(mean, cov, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _MultivariateNormal(mean, cov).sample(size)
