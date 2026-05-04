from abc import ABC


class AbstractDistributionType(ABC):
    """
    This class represents an abstract base for specific types of distributions,
    regardless of their domain (uniform, mixture, custom, etc.)
    """

    def convert_to(self, target_type, /, *, return_info: bool = False, **kwargs):
        """Convert or approximate this distribution as ``target_type``.

        This is a convenience wrapper around
        :func:`pyrecest.distributions.convert_distribution`. ``target_type`` may
        be either a concrete distribution class or a registered conversion alias.

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
