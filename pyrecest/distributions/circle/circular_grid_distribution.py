from .abstract_circular_distribution import AbstractCircularDistribution
from .circular_fourier_distribution import CircularFourierDistribution

from pyrecest.backend import array, pi, where, sin, linspace, ceil, floor, arange, sqrt, sum, maximum, mod, round, isclose, \
import numpy as np
import matplotlib.pyplot as plt
import warnings

class CircularGridDistribution(AbstractCircularDistribution, HypertoroidalGridDistribution):
    """
    Density representation using function values on a grid with Fourier interpolation.
    """

    def __init__(self, gridValues, enforcePdfNonnegative=True):
        gridValues = array(gridValues)
        # Check if gridValues is already a distribution (in which case use fromDistribution)
        if isinstance(gridValues, AbstractCircularDistribution):
            raise ValueError("You gave a distribution as the first argument. "
                             "To convert distributions to a distribution in grid representation, use fromDistribution.")
        # Call parent constructor (HypertoroidalGridDistribution)
        HypertoroidalGridDistribution.__init__(self, None, gridValues, enforcePdfNonnegative)
        self.gridValues = gridValues
        self.enforcePdfNonnegative = enforcePdfNonnegative

    def pdf(self, xs, useSinc=False, sincRepetitions=5):
        xs = array(xs)
        if useSinc:
            if sincRepetitions % 2 != 1:
                raise ValueError("sincRepetitions must be an odd integer.")
            N = len(self.gridValues)
            step_size = 2 * pi / N

            # Define MATLAB-style sinc: sinc(x) = sin(x)/x with sinc(0)=1
            def matlab_sinc(x):
                return where(x == 0, 1.0, sin(x) / x)

            # Create the range vector: from -floor(sincRepetitions/2)*N to ceil(sincRepetitions/2)*N - 1
            lower = int(floor(sincRepetitions / 2) * N)
            upper = int(ceil(sincRepetitions / 2) * N)
            r = arange(-lower, upper)

            # Compute the sinc values. (xs/step_size) becomes a column vector.
            sincVals = matlab_sinc((xs / step_size)[:, None] - r[None, :])
            # Tile the grid values; note that MATLAB’s repmat(gridValues', [1, sincRepetitions])
            if self.enforcePdfNonnegative:
                coeffs = np.tile(sqrt(self.gridValues), sincRepetitions)
                p = sum(coeffs * sincVals, axis=1) ** 2
            else:
                coeffs = np.tile(self.gridValues, sincRepetitions)
                p = sum(coeffs * sincVals, axis=1)
            return p
        else:
            N = len(self.gridValues)
            noCoeffs = N
            if N % 2 == 0:
                noCoeffs += 1  # extra coefficient as in the MATLAB code
            transform = 'sqrt' if self.enforcePdfNonnegative else 'identity'
            fd = CircularFourierDistribution.fromFunctionValues(self.gridValues, noCoeffs, transform)
            return fd.pdf(xs)

    def pdfOnGrid(self, noOfDesiredGridpoints):
        N = len(self.gridValues)
        xGrid = np.linspace(0, 2*np.pi, N, endpoint=False)
        step = N / noOfDesiredGridpoints
        if not float(step).is_integer():
            raise ValueError("Number of function values must be a multiple of noOfDesiredGridpoints")
        step = int(step)
        vals = self.gridValues[::step]
        return vals, xGrid

    def trigonometricMoment(self, n):
        N = len(self.gridValues)
        noCoeffs = N
        if N % 2 == 0:
            noCoeffs += 1
            # In MATLAB a warning is suppressed here; in Python you might use warnings.filterwarnings.
        fd = CircularFourierDistribution.fromFunctionValues(self.gridValues, noCoeffs, 'identity')
        return fd.trigonometricMoment(n)

    def plot(self, *args, **kwargs):
        N = len(self.gridValues)
        gridPoints = np.linspace(0, 2*np.pi, N, endpoint=False)
        # Optionally, call a parent plot if available.
        super().plot(*args, **kwargs)
        # Then also plot the grid points as markers.
        plt.plot(gridPoints, self.gridValues, 'x')
        plt.xlabel("Angle")
        plt.ylabel("Grid Values")
        plt.show()
        # Returning the list of line handles is optional in Python.
        return

    def value(self, xa):
        """
        Evaluate the density at points when interpreted as a probability mass function.
        This implementation assumes that the grid points are used as indicators.
        """
        xa = np.array(xa)
        grid = self.getGrid()
        # Find exact matches (using np.isclose for floating-point comparisons)
        result = np.zeros_like(xa)
        for i, angle in enumerate(xa):
            match = np.where(np.isclose(grid, angle))[0]
            result[i] = self.gridValues[match[0]] if match.size > 0 else 0
        return result

    def getGrid(self):
        N = len(self.gridValues)
        return np.linspace(0, 2*np.pi, N, endpoint=False)

    def getGridPoint(self, indices=None):
        """
        If indices is None, returns the entire grid.
        Otherwise, returns grid points corresponding to (indices - 1)*2*pi/N.
        (MATLAB indices are 1-based; here we mimic that conversion.)
        """
        N = len(self.gridValues)
        if indices is None:
            return self.getGrid()
        else:
            indices = np.array(indices)
            return (indices - 1) * (2 * np.pi / N)

    def convolve(self, f2):
        if self.enforcePdfNonnegative != f2.enforcePdfNonnegative:
            raise ValueError("Mismatch in enforcePdfNonnegative between the two distributions.")
        if len(self.gridValues) != len(f2.gridValues):
            raise ValueError("Grid sizes must be identical for convolution.")
        N = len(self.gridValues)
        # Circular convolution using FFTs
        fft1 = np.fft.fft(self.gridValues)
        fft2 = np.fft.fft(f2.gridValues)
        convResult = np.real(np.fft.ifft(fft1 * fft2)) * (2 * np.pi / N)
        convResult[convResult < 0] = 0  # Remove small negative values due to numerical error
        return CircularGridDistribution(convResult, self.enforcePdfNonnegative)

    def truncate(self, noOfGridpoints):
        N = len(self.gridValues)
        if noOfGridpoints - 1 <= 0:
            raise ValueError("Number of coefficients must be an integer greater than zero")
        step = N / noOfGridpoints
        if float(step).is_integer():
            new_vals = self.gridValues[::int(step)]
            return CircularGridDistribution(new_vals, self.enforcePdfNonnegative)
        elif N < noOfGridpoints:
            warnings.warn("Less coefficients than desired, interpolate using Fourier while ensuring nonnegativity.")
            self.enforcePdfNonnegative = True
            return CircularGridDistribution.fromDistribution(self, noOfGridpoints, self.enforcePdfNonnegative)
        else:
            warnings.warn("Cannot downsample directly. Transforming to Fourier to interpolate.")
            return CircularGridDistribution.fromDistribution(self, noOfGridpoints, self.enforcePdfNonnegative)

    def normalize(self, tol=1e-2, warnUnnorm=True):
        # Call normalization from the HypertoroidalGridDistribution
        return super().normalize(tol=tol, warnUnnorm=warnUnnorm)

    def shift(self, angle):
        if not np.isscalar(angle):
            raise ValueError("Angle must be a scalar.")
        fd = CircularFourierDistribution.fromFunctionValues(self.gridValues, len(self.gridValues), 'identity')
        fd = fd.shift(angle)
        new_vals, _ = fd.pdfOnGrid(len(self.gridValues))
        return CircularGridDistribution(new_vals, self.enforcePdfNonnegative)

    def getClosestPoint(self, xa):
        xa = np.array(xa)
        N = len(self.gridValues)
        # MATLAB: indices = mod(round(xa/(2*pi/N)), N) + 1 (1-indexed)
        # In Python, we compute zero-indexed indices.
        indices = (np.round(xa / (2 * np.pi / N)) % N).astype(int)
        points = indices * (2 * np.pi / N)
        return points, indices

    # --- Static methods ---
    @staticmethod
    def fromDistribution(distribution, noOfGridpoints, enforcePdfNonnegative=True):
        if isinstance(distribution, CicularFourierDistribution):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fdToConv = distribution.truncate(noOfGridpoints)
            # The MATLAB code uses ifftshift and then ifft; here we mimic that:
            c_shifted = np.fft.ifftshift(fdToConv.c)
            valsOnGrid = np.real(np.fft.ifft(c_shifted)) * (len(fdToConv.a) + len(fdToConv.b))
            if fdToConv.transformation == 'identity':
                if np.any(valsOnGrid < 0):
                    warnings.warn("This is an inaccurate transformation because negative values occurred. They are increased to 0.")
                    valsOnGrid = np.maximum(valsOnGrid, 0)
                if enforcePdfNonnegative:
                    warnings.warn("Interpolation differences may lead to inaccuracies; consider setting enforcePdfNonnegative to False.")
            elif fdToConv.transformation == 'sqrt':
                if np.any(valsOnGrid < 0) and enforcePdfNonnegative:
                    warnings.warn("Negative values occurred in the sqrt transformation. Consider converting to identity first.")
                elif (not enforcePdfNonnegative) and noOfGridpoints < (2 * (len(distribution.a) + len(distribution.b)) - 1):
                    warnings.warn("Interpolation differences may lead to inaccuracies with too few coefficients.")
                valsOnGrid = valsOnGrid ** 2
            else:
                raise ValueError("Transformation unsupported")
            return CircularGridDistribution(valsOnGrid, enforcePdfNonnegative)
        else:
            return CircularGridDistribution.fromFunction(lambda x: distribution.pdf(x),
                                                  noOfGridpoints,
                                                  enforcePdfNonnegative)

    @staticmethod
    def fromFunction(fun, noOfCoefficients, enforcePdfNonnegative=True):
        gridPoints = np.linspace(0, 2*np.pi, noOfCoefficients, endpoint=False)
        gridValues = np.array(fun(gridPoints))
        return CircularGridDistribution(gridValues, enforcePdfNonnegative)

    @staticmethod
    def fromFunctionValues(fvals, noOfGridpoints, enforcePdfNonnegative=True):
        fvals = np.array(fvals)
        step = len(fvals) / noOfGridpoints
        if not float(step).is_integer():
            raise ValueError("Number of function values has to be a multiple of noOfGridpoints")
        step = int(step)
        fvals_reduced = fvals[::step]
        return CircularGridDistribution(fvals_reduced, enforcePdfNonnegative)