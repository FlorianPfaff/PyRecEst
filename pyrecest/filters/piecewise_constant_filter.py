import numpy as np
from pyrecest.distributions.circle.piecewise_constant_distribution import PieceWiseConstantDistribution
from pyrecest.distributions import AbstractCircularDistribution
from scipy.integrate import nquad, quad
from .abstract_circular_filter import AbstractCircularFilter


class PiecewiseConstantFilter(AbstractCircularFilter):
    def __init__(self, n):
        # Constructor
        # Parameters:
        #   n (scalar)
        #       number of discretization steps
        assert np.isscalar(n)
        self.set_state(PieceWiseConstantDistribution(np.ones(n) / n))

    def set_state(self, new_state):
        assert isinstance(new_state, AbstractCircularDistribution)
        if isinstance(new_state, PieceWiseConstantDistribution):
            self.pwc = new_state
        else:
            self.pwc = PieceWiseConstantDistribution(PieceWiseConstantDistribution.calculate_parameters_numerically(new_state.pdf, len(self.pwc.w)))

    def predict(self, A):
        # Prediction step based on transition matrix
        # Parameters:
        #   A (L x L matrix)
        #       system matrix
        w = self.pwc.w
        w = np.matmul(A, w.T)
        w = w.T / np.sum(w)
        self.pwc = PieceWiseConstantDistribution(w)

    def update(self, H, z):
        # Measurement update based on measurement matrix
        # Parameters:
        #   H (Lw x L matrix)
        #       measurement matrix 
        #   z (scalar)
        #       measurement 
        # get the right column from H
        assert H.shape[1] == len(self.pwc.w)
        assert np.isscalar(z)
        Lw = H.shape[0]
        row = int(np.floor(1 + z / (2 * np.pi) * Lw))
        w = self.pwc.w
        w = H[row, :] * w
        self.pwc = PieceWiseConstantDistribution(w)

    def update_likelihood(self, likelihood, z):
        # Updates assuming nonlinear measurement model given by a
        # likelihood function likelihood(z, x) = f(z | x), where z is the
        # measurement. The function can be created using the
        # LikelihoodFactory.
        #
        # Parameters:
        #   likelihood (function handle)
        #       function from Z x [0, 2pi) to [0, infinity), where Z is
        #       the measurement space containing z
        #   z (arbitrary)
        #       measurement
        L = len(self.pwc.w)
        tmp = np.zeros(L)
        for i in range(L):
            tmp[i] = quad(lambda x: likelihood(z, x), PieceWiseConstantDistribution.left_border(i, L), PieceWiseConstantDistribution.right_border(i, L))[0]
        self.pwc = PieceWiseConstantDistribution(tmp * self.pwc.w)

    def get_estimate(self):
        return self.pwc


    
    @staticmethod
    def calculateSystemMatrixNumerically(L, a, noiseDistribution):
        # Obtains system matrix by 2d numerical integration from system
        # function
        # Parameters:
        #   L (scalar)
        #       number of discretization intervals
        #   a (function handle)
        #       system function x_k+1 = a(x_k,w_k), needs to be
        #       vectorized
        #   noiseDistribution (AbstractCircularDistribution)
        #       noise (assumed to be defined on [0,2pi)
        # Returns;
        #   A (L x L matrix)
        #       system matrix
        assert isinstance(L, int)
        assert isinstance(noiseDistribution, AbstractCircularDistribution)
        assert callable(a)

        A = np.zeros((L, L))
        for i in range(L):
            l1 = PieceWiseConstantDistribution.leftBorder(i, L)
            r1 = PieceWiseConstantDistribution.rightBorder(i, L)
            for j in range(L):
                l2 = PieceWiseConstantDistribution.leftBorder(j, L)
                r2 = PieceWiseConstantDistribution.rightBorder(j, L)
                indicator = lambda x: np.where((l2 < x) & (x < r2), 1, 0)

                def integrand(x, w):
                    return noiseDistribution.pdf(w) * indicator(a(x, w))

                A[j, i] = nquad(integrand, [[l1, r1], [0, 2 * np.pi]])[0] * L / 2 / np.pi

        return A
    
    
    @staticmethod
    def calculateMeasurementMatrixNumerically(L, Lmeas, h, noiseDistribution):
        # Obtains system matrix by 2d numerical integration from
        # measurement function
        # Parameters:
        #   L (scalar)
        #       number of discretization intervals for state
        #   Lmeas (scalar)
        #       number of discretization intervals for measurement
        #   h (function handle)
        #       system function z_k = h(x_k,v_k), needs to be
        #       vectorized
        #   noiseDistribution (AbstractCircularDistribution)
        #       noise (assumed to be defined on [0,2pi)
        # Returns;
        #   H (Lmeas x L matrix)
        #       measurement matrix
        assert isinstance(L, int)
        assert isinstance(Lmeas, int)
        assert isinstance(noiseDistribution, AbstractCircularDistribution)
        assert callable(h)

        H = np.zeros((Lmeas, L))
        for i in range(Lmeas):
            l1 = PieceWiseConstantDistribution.leftBorder(i, Lmeas)
            r1 = PieceWiseConstantDistribution.rightBorder(i, Lmeas)
            for j in range(L):
                l2 = PieceWiseConstantDistribution.leftBorder(j, L)
                r2 = PieceWiseConstantDistribution.rightBorder(j, L)
                indicator = lambda x: np.where((l1 < x) & (x < r1), 1, 0)

                def integrand(x, v):
                    return noiseDistribution.pdf(v) * indicator(h(x, v))

                H[i, j] = nquad(integrand, [[l2, r2], [0, 2 * np.pi]])[0] * L / 2 / np.pi

        return H
