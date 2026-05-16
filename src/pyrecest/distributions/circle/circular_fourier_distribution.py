import warnings
from typing import Union

import matplotlib.pyplot as plt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arange,
    array,
    concatenate,
    conj,
    cos,
    exp,
    fft,
    hstack,
    imag,
    int32,
    int64,
    linspace,
    pi,
    real,
    sin,
    sqrt,
    sum,
)

from .abstract_circular_distribution import AbstractCircularDistribution
from .circular_dirac_distribution import CircularDiracDistribution


class CircularFourierDistribution(AbstractCircularDistribution):
    """Circular distribution represented by a Fourier series.

    References
    ----------
    Pfaff, F., Kurz, G., & Hanebeck, U. D. (2015). Multimodal Circular
    Filtering Using Fourier Series. Proceedings of the 18th International
    Conference on Information Fusion.

    Pfaff, F., Kurz, G., & Hanebeck, U. D. (2016). Nonlinear Prediction for
    Circular Filtering Using Fourier Series. Proceedings of the 19th
    International Conference on Information Fusion.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        transformation: str = "sqrt",
        c=None,
        a=None,
        b=None,
        n: int | None = None,
        multiplied_by_n: bool = True,
    ):
        AbstractCircularDistribution.__init__(self)
        assert (a is None) == (b is None)
        assert (a is None) != (c is None)
        if c is not None:  # Assumed as result from rfft
            assert c.ndim == 1
            self.c = c
            self.a = None
            self.b = None
            if n is None:
                warnings.warn(
                    "It is not clear for complex ones since they may include another coefficient or not (imaginary part of the last coefficient). Assuming it is relevant."
                )
                self.n = 2 * c.shape[0] - 1
            else:
                self.n = n
        elif a is not None and b is not None:
            assert a.ndim == 1
            assert b.ndim == 1
            self.a = a
            self.b = b
            self.c = None
            self.n = a.shape[0] + b.shape[0]
            assert self.n == n or n is None
        else:
            raise ValueError("Need to provide either c or a and b.")

        self.multiplied_by_n = multiplied_by_n
        self.transformation = transformation

    def __sub__(
        self, other: "CircularFourierDistribution"
    ) -> "CircularFourierDistribution":
        # If transformed will not yield minus of pdfs!
        assert (
            (self.a is not None and other.a is not None)
            and (self.b is not None and other.b is not None)
        ) or (
            self.c is not None and other.c is not None
        ), "Either both instances should have `a` and `b` defined, or both should have `c` defined."

        assert self.transformation == other.transformation
        assert self.n == other.n
        assert self.multiplied_by_n == other.multiplied_by_n
        if self.a is not None and self.b is not None:
            aNew = self.a - other.a
            bNew = self.b - other.b
            fdNew = CircularFourierDistribution(
                a=aNew,
                b=bNew,
                transformation=self.transformation,
                multiplied_by_n=self.multiplied_by_n,
            )
        else:
            cNew = self.c - other.c
            fdNew = CircularFourierDistribution(
                c=cNew,
                n=self.n,
                transformation=self.transformation,
                multiplied_by_n=self.multiplied_by_n,
            )
        return fdNew

    def pdf(self, xs):
        assert xs.ndim <= 2, "xs should have at most 2 dimensions."
        xs = xs.reshape(-1, 1)
        a, b = self.get_a_b()

        k_range = arange(1, a.shape[0], dtype=xs.dtype)
        p = a[0] / 2.0 + sum(
            a[1:].reshape(1, -1) * cos(xs * k_range)
            + b.reshape(1, -1) * sin(xs * k_range),
            axis=1,
        )
        if self.multiplied_by_n:
            p = p / self.n
        if self.transformation == "sqrt":
            p = p**2
        elif self.transformation == "identity":
            pass
        else:
            raise NotImplementedError("Transformation not supported.")
        return p

    def normalize(self) -> "CircularFourierDistribution":
        integral_value = self.integrate()

        if self.a is not None and self.b is not None:
            if self.transformation == "identity":
                scale_factor = 1.0 / integral_value
            elif self.transformation == "sqrt":
                scale_factor = 1.0 / sqrt(integral_value)
            else:
                raise NotImplementedError("Transformation not supported.")

            a_new = self.a * scale_factor
            b_new = self.b * scale_factor
            fd_normalized = CircularFourierDistribution(
                a=a_new,
                b=b_new,
                transformation=self.transformation,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )

        elif self.c is not None:
            if self.transformation == "identity":
                scale_factor = 1.0 / integral_value
            elif self.transformation == "sqrt":
                scale_factor = 1.0 / sqrt(integral_value)
            else:
                raise NotImplementedError("Transformation not supported.")

            c_new = self.c * scale_factor
            fd_normalized = CircularFourierDistribution(
                c=c_new,
                transformation=self.transformation,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )

        else:
            raise ValueError("Need either a and b or c.")

        return fd_normalized

    # pylint: disable=too-many-branches
    def integrate(self, integration_boundaries=None) -> float:
        assert (
            integration_boundaries is None
        ), "Currently, only supported for entire domain."
        if self.a is not None and self.b is not None:
            a: array = self.a
            b: array = self.b
            if self.multiplied_by_n:
                a = a * (1.0 / self.n)
                b = b * (1.0 / self.n)

            if self.transformation == "identity":
                a0_non_rooted = a[0]
            elif self.transformation == "sqrt":
                from_a0 = a[0] ** 2 * 0.5
                from_a1_to_end_and_b = sum(a[1:] ** 2) + sum(b**2)
                a0_non_rooted = from_a0 + from_a1_to_end_and_b
            else:
                raise NotImplementedError("Transformation not supported.")
            integral = a0_non_rooted * pi
        elif self.c is not None:
            if self.transformation == "identity":
                if self.multiplied_by_n:
                    c0 = real(self.c[0]) * (1.0 / self.n)
                else:
                    c0 = real(self.c[0])
                integral = 2.0 * pi * c0
            elif self.transformation == "sqrt":
                if self.multiplied_by_n:
                    c = self.c * (1 / self.n)
                else:
                    c = self.c
                from_c0 = (real(c[0])) ** 2
                from_c1_to_end = sum((real(c[1:])) ** 2) + sum((imag(c[1:])) ** 2)

                a0_non_rooted = 2.0 * from_c0 + 4.0 * from_c1_to_end
                integral = a0_non_rooted * pi
            else:
                raise NotImplementedError("Transformation not supported.")
        else:
            raise ValueError("Need either a and b or c.")
        return integral

    def plot_grid(self):
        grid_values = fft.irfft(self.get_c(), self.n)
        xs = linspace(0, 2 * pi, grid_values.shape[0], endpoint=False)
        vals = grid_values.squeeze()

        if self.transformation == "sqrt":
            p = vals**2
        elif self.transformation == "log":
            p = exp(vals)
        elif self.transformation == "identity":
            p = vals
        else:
            raise NotImplementedError("Transformation not supported.")

        plt.plot(xs, p, "r+")
        plt.show()

    def plot(self, resolution=128, **kwargs):
        xs = linspace(0.0, 2.0 * pi, resolution)

        if self.a is not None:
            xs = xs.astype(self.a.dtype)
        else:
            xs = xs.astype(real(self.c).dtype)

        pdf_vals = self.pdf(xs)

        p = plt.plot(xs, pdf_vals, **kwargs)
        plt.show()

        return p

    def get_a_b(self):
        if self.a is not None:
            a = self.a
            b = self.b
        elif self.c is not None:
            a = 2.0 * real(self.c)
            b = -2.0 * imag(self.c[1:])
        else:
            raise ValueError("Need either a and b or c.")
        assert (
            self.n is None or (a.shape[0] + b.shape[0]) == self.n
        )  # Other case not implemented yet!
        return a, b

    def get_c(self):
        if self.a is not None:
            c = hstack((self.a[0:1], self.a[1:] - 1j * self.b)) * 0.5
        elif self.c is not None:
            c = self.c
        else:
            raise ValueError("Need either a and b or c.")

        return c

    def to_real_fd(self):
        if self.a is not None:
            fd = self
        elif self.c is not None:
            a, b = self.get_a_b()
            fd = CircularFourierDistribution(
                transformation=self.transformation,
                a=a,
                b=b,
                n=self.n,
                multiplied_by_n=self.multiplied_by_n,
            )
        else:
            raise ValueError("Need either a and b or c.")

        return fd

    def get_full_c(self):
        assert self.c is not None
        neg_c = conj(self.c[-1:0:-1])  # Create array for negative-frequency components
        full_c = concatenate([neg_c, self.c])  # Concatenate arrays to get full spectrum
        return full_c

    @staticmethod
    def from_distribution(
        distribution: AbstractCircularDistribution,
        n: Union[int, int32, int64],
        transformation: str = "sqrt",
        store_values_multiplied_by_n: bool = True,
    ) -> "CircularFourierDistribution":
        if isinstance(distribution, CircularDiracDistribution):
            if transformation != "identity":
                warnings.warn(
                    "CircularDiracDistribution is represented in Fourier form "
                    "with identity transformation.",
                    RuntimeWarning,
                )
                transformation = "identity"
            coeffs = array(
                [
                    conj(distribution.trigonometric_moment(k).squeeze()) / (2.0 * pi)
                    for k in range(int(n) // 2 + 1)
                ]
            )
            fd = CircularFourierDistribution(
                c=coeffs,
                n=int(n),
                transformation=transformation,
                multiplied_by_n=False,
            )
            if store_values_multiplied_by_n:
                warnings.warn("Scaling up for WD (this is not recommended).")
                fd.c = fd.c * fd.n
                fd.multiplied_by_n = True
        else:
            xs = arange(
                0.0, 2.0 * pi, 2.0 * pi / n
            )  # Like linspace without endpoint but with compatbiility for pytorch
            fvals = distribution.pdf(xs)
            if transformation == "identity":
                pass
            elif transformation == "sqrt":
                fvals = sqrt(fvals)
            else:
                raise NotImplementedError("Transformation not supported.")
            fd = CircularFourierDistribution.from_function_values(
                fvals, transformation, store_values_multiplied_by_n
            )

        return fd

    @staticmethod
    def from_function_values(
        fvals,
        transformation: str = "sqrt",
        store_values_multiplied_by_n: bool = True,
    ) -> "CircularFourierDistribution":
        c = fft.rfft(fvals)
        if not store_values_multiplied_by_n:
            c = c * (1.0 / fvals.shape[0])

        fd = CircularFourierDistribution(
            c=c,
            transformation=transformation,
            n=fvals.shape[0],
            multiplied_by_n=store_values_multiplied_by_n,
        )

        return fd
