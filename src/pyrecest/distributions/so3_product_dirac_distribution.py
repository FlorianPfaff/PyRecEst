"""Dirac distributions on Cartesian products of SO(3)."""

# pylint: disable=no-name-in-module,no-member,arguments-renamed
from math import prod

from pyrecest.backend import (
    abs,
    all,
    arange,
    arccos,
    array,
    asarray,
    clip,
    linalg,
    ndim,
    pi,
    random,
    reshape,
    spatial,
    stack,
    sum,
    where,
)

from .cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from .hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)


class SO3ProductDiracDistribution(HyperhemisphereCartProdDiracDistribution):
    """Weighted Dirac distribution on SO(3)^K.

    Rotations are represented as scalar-last unit quaternions ``(x, y, z, w)``.
    The constructor also accepts flattened particles of shape ``(n, 4 * K)``.
    Internally, particles are stored as ``(n, K, 4)`` and canonicalized to the
    upper quaternion hemisphere.
    """

    def __init__(self, d, w=None, num_rotations=None):
        quaternions, inferred_num_rotations = self._as_particle_array(
            d, num_rotations=num_rotations
        )
        self.num_rotations = inferred_num_rotations
        super().__init__(
            quaternions,
            w=w,
            dim_hemisphere=3,
            n_hemispheres=inferred_num_rotations,
            store_flat=False,
        )

    def get_manifold_size(self):
        return pi ** (2 * self.num_rotations)

    @staticmethod
    def _as_particle_array(d, num_rotations=None):
        quaternions = array(d)

        if quaternions.ndim == 2:
            if num_rotations is not None and quaternions.shape == (num_rotations, 4):
                inferred_num_rotations = num_rotations
                quaternions = reshape(quaternions, (1, num_rotations, 4))
            else:
                if quaternions.shape[-1] % 4 != 0:
                    raise ValueError(
                        "Flattened SO(3)^K Dirac locations need 4 entries per "
                        "rotation."
                    )
                inferred_num_rotations = quaternions.shape[-1] // 4
                if (
                    num_rotations is not None
                    and inferred_num_rotations != num_rotations
                ):
                    raise ValueError("num_rotations does not match input shape.")
                quaternions = SO3ProductDiracDistribution._as_component_array(
                    quaternions,
                    dim_hemisphere=3,
                    n_hemispheres=inferred_num_rotations,
                )
        elif quaternions.ndim == 3:
            if quaternions.shape[-1] != 4:
                raise ValueError("SO(3) quaternions must have four entries.")
            inferred_num_rotations = quaternions.shape[1]
            if num_rotations is not None and inferred_num_rotations != num_rotations:
                raise ValueError("num_rotations does not match input shape.")
        else:
            raise ValueError(
                "SO(3)^K Dirac locations must have shape (n, K, 4), "
                "(n, 4 * K), or (K, 4) with num_rotations=K."
            )

        return (
            SO3ProductDiracDistribution._canonicalize_quaternions(
                SO3ProductDiracDistribution._normalize_quaternions(quaternions)
            ),
            inferred_num_rotations,
        )

    @staticmethod
    def _normalize_quaternions(quaternions):
        norms = linalg.norm(quaternions, axis=-1)
        if not all(norms > 0.0):
            raise ValueError("SO(3) quaternions must be nonzero.")
        return quaternions / reshape(norms, tuple(norms.shape) + (1,))

    @staticmethod
    def _canonicalize_quaternions(quaternions):
        return where(quaternions[..., -1:] < 0.0, -quaternions, quaternions)

    @staticmethod
    def _require_rotation_method(method_name):
        if not hasattr(spatial.Rotation, method_name):
            raise NotImplementedError(
                f"Rotation.{method_name} is not supported by the active backend."
            )

    @classmethod
    def from_rotation_matrices(cls, rotation_matrices, w=None):
        """Create an SO(3)^K Dirac distribution from rotation matrices."""
        cls._require_rotation_method("from_matrix")
        rotation_matrices = asarray(rotation_matrices)
        if ndim(rotation_matrices) < 2 or rotation_matrices.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrices must have shape (..., 3, 3).")

        if ndim(rotation_matrices) == 2:
            quaternions = array(
                spatial.Rotation.from_matrix(rotation_matrices).as_quat()
            )
            return cls(reshape(quaternions, (1, 1, 4)), w=w)

        if ndim(rotation_matrices) == 3:
            quaternions = array(
                spatial.Rotation.from_matrix(rotation_matrices).as_quat()
            )
            return cls(quaternions, w=w, num_rotations=quaternions.shape[0])

        num_rotations = int(rotation_matrices.shape[-3])
        leading_shape = tuple(rotation_matrices.shape[:-3])
        num_particles = int(prod(leading_shape))
        flat_matrices = reshape(
            rotation_matrices, (num_particles * num_rotations, 3, 3)
        )
        quaternions = array(spatial.Rotation.from_matrix(flat_matrices).as_quat())
        return cls(reshape(quaternions, (num_particles, num_rotations, 4)), w=w)

    def as_quaternions(self):
        """Return Dirac locations with shape ``(n, K, 4)``."""
        return self.d

    def as_flat_quaternions(self):
        """Return Dirac locations with shape ``(n, 4 * K)``."""
        return self.as_flat_array()

    def sample(self, n):
        indices = random.choice(arange(self.d.shape[0]), n, p=self.w)
        samples = self.d[indices]
        if int(n) == 1 and samples.shape == (self.num_rotations, 4):
            return reshape(samples, (1, self.num_rotations, 4))
        return samples

    def marginalize_rotation(self, rotation_index):
        """Return the single SO(3) marginal at ``rotation_index``."""
        return HyperhemisphericalDiracDistribution(
            self.component_particles(rotation_index), self.w
        )

    def marginalize_rotations(self, rotation_indices):
        """Return the SO(3)^L marginal selected by ``rotation_indices``."""
        return SO3ProductDiracDistribution(self.d[:, rotation_indices, :], self.w)

    def moment(self, rotation_index=None):
        """Return weighted quaternion second moments.

        With ``rotation_index=None``, the result has shape ``(K, 4, 4)``.
        Otherwise it returns the ``(4, 4)`` moment of one SO(3) component.
        """
        if rotation_index is not None:
            return self._moment_for_rotation(rotation_index)
        return super().moment()

    def _moment_for_rotation(self, rotation_index):
        return self._moment_for_component(rotation_index)

    def mean_quaternion(self, rotation_index=None):
        """Return the weighted chordal quaternion mean."""
        if rotation_index is not None:
            return self._mean_quaternion_for_rotation(rotation_index)
        return stack(
            [self._mean_quaternion_for_rotation(i) for i in range(self.num_rotations)],
            0,
        )

    def _mean_quaternion_for_rotation(self, rotation_index):
        return self._mean_axis_for_component(rotation_index)

    def mean(self):
        """Return the weighted chordal quaternion mean for each component."""
        return self.mean_quaternion()

    @staticmethod
    def geodesic_distance(rotation_a, rotation_b):
        """Return SO(3) geodesic distances between scalar-last quaternions."""
        quaternion_a = SO3ProductDiracDistribution._normalize_quaternions(
            array(rotation_a)
        )
        quaternion_b = SO3ProductDiracDistribution._normalize_quaternions(
            array(rotation_b)
        )
        dot_products = sum(quaternion_a * quaternion_b, axis=-1)
        return 2.0 * arccos(clip(abs(dot_products), -1.0, 1.0))

    def distance_to(self, rotations, reduce=True):
        """Return component-wise or summed geodesic distances to ``rotations``."""
        rotations, _ = self._as_particle_array(
            rotations, num_rotations=self.num_rotations
        )

        if rotations.shape[0] == 1:
            distances = self.geodesic_distance(self.d, rotations[0])
        elif rotations.shape[0] == self.d.shape[0]:
            distances = self.geodesic_distance(self.d, rotations)
        else:
            raise ValueError(
                "rotations must contain one product point or one point per Dirac."
            )

        if reduce:
            return sum(distances, axis=-1)
        return distances

    def angular_error_mean(self, rotations):
        """Return the weighted mean summed geodesic distance to ``rotations``."""
        return sum(self.w * self.distance_to(rotations, reduce=True))

    @staticmethod
    def as_rotation_matrices(quaternions):
        """Convert scalar-last unit quaternions to rotation matrices."""
        quaternions = SO3ProductDiracDistribution._normalize_quaternions(
            array(quaternions)
        )
        x = quaternions[..., 0]
        y = quaternions[..., 1]
        z = quaternions[..., 2]
        w = quaternions[..., 3]

        row_0 = stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            -1,
        )
        row_1 = stack(
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            -1,
        )
        row_2 = stack(
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            -1,
        )

        return stack((row_0, row_1, row_2), -2)

    def mean_rotation_matrices(self):
        """Return rotation matrices of the component-wise quaternion means."""
        return self.as_rotation_matrices(self.mean_quaternion())

    def is_valid(self, tolerance=1e-6):
        norms = linalg.norm(self.d, axis=-1)
        valid_norms = all(abs(norms - 1.0) <= tolerance)
        canonical = all(self.d[..., -1] >= -tolerance)
        return bool(valid_norms) and bool(canonical)
