import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest.backend import empty, int32, int64, log, random, squeeze


class AbstractManifoldSpecificDistribution(ABC):
    """
    Abstract base class for distributions catering to specific manifolds.
    Should be inerhited by (abstract) classes limited to specific manifolds.
    """

    def __init__(self, dim: int):
        self._dim = dim

    @abstractmethod
    def get_manifold_size(self) -> float:
        pass

    def get_ln_manifold_size(self):
        return log(self.get_manifold_size())

    def convert_to(self, target_type, /, *, return_info: bool = False, **kwargs):
        """Convert or approximate this distribution as ``target_type``.

        This is a convenience wrapper around
        :func:`pyrecest.distributions.convert_distribution`. ``target_type`` may
        be either a concrete distribution class or a registered conversion
        alias such as ``"particles"`` or ``"gaussian"``.

        Parameters
        ----------
        target_type
            Concrete target representation class or conversion alias.
        return_info
            If true, return a ``ConversionResult`` containing metadata.
        **kwargs
            Conversion parameters required by the target representation.
        """
        from .conversion import convert_distribution

        return convert_distribution(
            self, target_type, return_info=return_info, **kwargs
        )

    def approximate_as(self, target_type, /, *, return_info: bool = False, **kwargs):
        """Alias for :meth:`convert_to` emphasizing approximate conversions."""
        return self.convert_to(target_type, return_info=return_info, **kwargs)

    @property
    def dim(self) -> int:
        """Get dimension of the manifold."""
        return self._dim

    @dim.setter
    def dim(self, value: int):
        """Set dimension of the manifold. Must be a positive integer or None."""
        if value <= 0:
            raise ValueError("dim must be a positive integer or None.")

        self._dim = value

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def pdf(self, xs):
        pass

    def ln_pdf(self, xs):
        return log(self.pdf(xs))

    @abstractmethod
    def mean(self):
        """
        Convenient access to a reasonable "mean" for different manifolds.

        :return: The mean of the distribution.
        :rtype:
        """

    def set_mode(self, _):
        """
        Set the mode of the distribution
        """
        raise NotImplementedError("set_mode is not implemented for this distribution")

    # Need to use Union instead of | to support torch.dtype
    def sample(self, n: Union[int, int32, int64]):
        """Obtain n samples from the distribution."""
        return self.sample_metropolis_hastings(n)

    # jscpd:ignore-start
    # pylint: disable=too-many-positional-arguments,too-many-locals
    def sample_metropolis_hastings(
        self,
        n: Union[int, int32, int64],
        burn_in: Union[int, int32, int64] = 10,
        skipping: Union[int, int32, int64] = 5,
        proposal: Callable | None = None,
        start_point=None,
    ):
        # jscpd:ignore-end
        """Metropolis Hastings sampling algorithm."""
        if pyrecest.backend.__backend_name__ == "jax":
            # Get a key from your global JAX random state *outside* of lax.scan
            import jax as _jax  # pylint: disable=import-error

            key = random.get_state()
            key, key_for_mh = _jax.random.split(key)
            # Optionally update global state for future calls
            random.set_state(key)

            if proposal is None or start_point is None:
                raise NotImplementedError(
                    "Default proposals and starting points should be set in inheriting classes."
                )
            _assert_proposal_supports_key(proposal)

            samples, _ = sample_metropolis_hastings_jax(
                key=key_for_mh,
                log_pdf=self.ln_pdf,
                proposal=proposal,  # must be (key, x) -> x_prop for JAX
                start_point=start_point,
                n=int(n),
                burn_in=int(burn_in),
                skipping=int(skipping),
            )
            # You could optionally stash `key_out` somewhere if you want chain continuation.
            return squeeze(samples)

        # Non-JAX backends -> your old NumPy/Torch code
        if proposal is None or start_point is None:
            raise NotImplementedError(
                "Default proposals and starting points should be set in inheriting classes."
            )

        total_samples = burn_in + n * skipping
        s = empty((total_samples, self.input_dim))
        x = start_point
        i = 0
        pdfx = self.pdf(x)

        while i < total_samples:
            x_new = proposal(x)
            assert (
                x_new.shape == x.shape
            ), "Proposal must return a vector of same shape as input"
            pdfx_new = self.pdf(x_new)
            a = pdfx_new / pdfx
            if a.item() > 1 or a.item() > random.rand(1):
                x = x_new
                pdfx = pdfx_new
            # Record every chain step; rejected proposals keep the current state.
            s[i, :] = x.squeeze()
            i += 1

        relevant_samples = s[burn_in::skipping, :]
        return squeeze(relevant_samples)


# pylint: disable=too-many-positional-arguments,too-many-locals,too-many-arguments
def sample_metropolis_hastings_jax(
    key,
    log_pdf,  # function: x -> log p(x)
    proposal,  # function: (key, x) -> x_prop
    start_point,
    n: int,
    burn_in: int = 10,
    skipping: int = 5,
):
    """
    Metropolis-Hastings sampler in JAX using a plain Python loop.

    Uses a Python loop (rather than lax.scan) so that log_pdf may call
    non-JAX-traceable code (e.g. scipy).

    key:        jax.random.PRNGKey
    log_pdf:    callable x -> log p(x)
    proposal:   callable (key, x) -> x_proposed
    start_point: initial state (array)
    n:          number of samples to return (after burn-in and thinning)
    """
    import jax.numpy as _jnp  # pylint: disable=import-error
    from jax import random as _random  # pylint: disable=import-error

    start_point = _jnp.asarray(start_point)
    total_steps = burn_in + n * skipping
    chain = []

    x = start_point

    def _to_scalar(val):
        """Convert a JAX array of any shape to a Python float."""
        return float(_jnp.asarray(val).ravel()[0])

    log_px = _to_scalar(log_pdf(x))

    for _ in range(total_steps):
        key, key_prop, key_u = _random.split(key, 3)

        # Propose new state
        x_prop = proposal(key_prop, x)
        log_px_prop = _to_scalar(log_pdf(x_prop))

        # Metropolis acceptance
        log_alpha = log_px_prop - log_px
        log_u = _to_scalar(_jnp.log(_random.uniform(key_u, shape=())))

        if log_u < min(0.0, log_alpha):
            x = x_prop
            log_px = log_px_prop

        chain.append(x)

    chain_array = _jnp.stack(chain, axis=0)
    samples = chain_array[burn_in::skipping]
    return samples, key


def _assert_proposal_supports_key(proposal: Callable):
    """
    Check that `proposal` can be called as proposal(key, x).

    Raises a TypeError with a helpful message if this is not the case.
    """
    # Unwrap jitted / partial / decorated functions if possible
    func = proposal
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # Can't introspect (e.g. builtins); fall back to a generic error
        raise TypeError(
            "For the JAX backend, `proposal` must accept (key, x) as arguments, "
            "but its signature could not be inspected."
        ) from None

    params = list(sig.parameters.values())

    # Count positional(-or-keyword) parameters
    num_positional = sum(
        p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )
    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

    if has_var_positional or num_positional >= 2:
        # Looks compatible with (key, x)
        return

    raise TypeError(
        "For the JAX backend, `proposal` must accept `(key, x)` as arguments.\n"
        f"Got signature: {sig}\n"
        "Hint: change your proposal from `def proposal(x): ...` to\n"
        "`def proposal(key, x): ...` and use `jax.random` with the passed key."
    )
