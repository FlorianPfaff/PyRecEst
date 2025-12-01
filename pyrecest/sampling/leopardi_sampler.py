"""Implementation of the equal area partition of the sphere S^dim into N regions by Paul Leopardi. See
"A partition of the unit sphere into regions of equal area and small diameter", Paul Leopardi, 2006.
The code follows Leopardi's Matlab implementation of the algorithm. The coverage of the "lower" halves of
the sphere is different than in his original code. Note that the order of hyperspherical coordinates
output by get_partition_points_polar follow Leopardi's convention and not that used in pyrecest.
For obtaning Cartesian coordinates, see LeopardiSampler in hyperspherical_sampler.py.
"""

import copy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    arange,
    array,
    flip,
    int32,
    linspace,
    max,
    ones,
    pi,
    reshape,
    round,
    sin,
    vstack,
    zeros,
    zeros_like,
)
from scipy.optimize import root_scalar
from scipy.special import betainc  # pylint: disable=E0611

from ..distributions import (
    AbstractHypersphereSubsetDistribution,
)


def get_cap_area(dim, colatitude):
    """
    Compute the surface area of a spherical cap on S^dim, given its colatitude.

    Parameters:
    - dim (int): Dimension of the sphere.
    - colatitude: Colatitude angle in radians, in [0, pi].

    Returns:
    - area: Area of the spherical cap.
    """
    if dim == 1:
        return 2 * colatitude
    if dim == 2:
        return 4 * pi * sin(colatitude / 2) ** 2

    area_sphere = (
        AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(dim)
    )
    x = sin(colatitude / 2) ** 2
    a = dim / 2
    b = dim / 2
    area = area_sphere * betainc(a, b, x)
    return area


def get_cap_colatitude(dim, area):
    """
    Compute the colatitude of a spherical cap on S^dim, given its area.

    Parameters:
    - dim : Dimension of the sphere.
    - area: Area of the spherical cap.

    Returns:
    - colatitude: Colatitude angle in radians, in [0, pi].
    """
    colatitude = zeros_like(area)

    area_sphere = (
        AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(dim)
    )

    def f(s, dim, area):
        result = get_cap_area(dim, s) - area
        if result < 0 and abs(result) < 1e-12:
            # Adjust for numerical errors near zero
            result = 0
        return result

    if area >= area_sphere:
        colatitude = pi
    elif area <= 0:
        colatitude = 0
    else:
        if area > area_sphere / 2:
            flipped = True
            area = area_sphere - area
        else:
            flipped = False
        res = root_scalar(f, args=(dim, area), bracket=[0, pi / 2], method="bisect")
        s = res.root
        if flipped:
            colatitude = pi - s
        else:
            colatitude = s
    return colatitude


def get_region_area(dim, N):
    """
    Compute the area of one region in an equal area partition of S^dim into N regions.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Number of regions.

    Returns:
    - area: Area of one region.
    """
    return array(
        AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(dim) / N
    )


def get_polar_cap_colatitude(dim, N):
    """
    Compute the colatitude of the polar cap for the EQ partitioning.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Number of regions.

    Returns:
    - c_polar: Colatitude angle in radians.
    """
    if N == 1:
        return array(pi)

    if N == 2:
        return array(pi / 2)

    area = get_region_area(dim, N)
    return get_cap_colatitude(dim, area)


def get_ideal_collar_angle(dim, N):
    """
    Compute the ideal collar angle for EQ partitioning.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Number of regions.

    Returns:
    - angle: Ideal collar angle in radians.
    """
    area = get_region_area(dim, N)
    angle = area ** (1 / dim)
    return angle


def get_number_of_collars(N, c_polar, ideal_angle):
    """
    Determine the number of collars between the polar caps.

    Parameters:
    - N: Number of regions.
    - c_polar: Colatitude of polar cap.
    - ideal_angle: Ideal collar angle.

    Returns:
    - n_collars: Number of collars.
    """
    if not ((N > 2) & (ideal_angle > 0)):
        return 0

    return max(array((1, round((pi - 2 * c_polar) / ideal_angle)), dtype=int32))


def get_ideal_region_counts(dim, N, c_polar, n_collars):
    """
    Get the ideal (real) number of regions in each zone (collars and caps).

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.
    - c_polar: Colatitude of the polar cap.
    - n_collars (int): Number of collars.

    Returns:
    - r_regions (ndarray): Ideal number of regions in each zone.
    """
    r_regions = zeros(n_collars + 2)
    r_regions[0] = 1  # North polar cap
    ideal_region_area = get_region_area(dim, N)
    if n_collars > 0:
        a_fitting = (pi - 2 * c_polar) / n_collars
        for collar_n in range(1, n_collars + 1):
            a_top = c_polar + (collar_n - 1) * a_fitting
            a_bot = c_polar + collar_n * a_fitting
            collar_area = get_cap_area(dim, a_bot) - get_cap_area(dim, a_top)
            r_regions[collar_n] = collar_area / ideal_region_area
    r_regions[-1] = 1  # South polar cap
    return r_regions


def round_region_counts(region_counts):
    """
    Round region counts to integers summing to N, minimizing total discrepancy.

    Parameters:
    - N (int): Total number of regions.
    - region_counts (array_like): Ideal (real) region counts.

    Returns:
    - n_regions (ndarray): Rounded region counts summing to N.
    """
    n_regions = zeros_like(region_counts, dtype=int)
    discrepancy = 0
    for zone_n in range(len(region_counts)):
        n_regions[zone_n] = int(round(region_counts[zone_n] + discrepancy))
        discrepancy += region_counts[zone_n] - n_regions[zone_n]
    return n_regions


def get_cap_colatitudes(dim, N, c_polar, n_regions):
    """
    Compute the colatitudes of caps that enclose cumulative sums of regions.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.
    - c_polar (float): Colatitude of the polar cap.
    - n_regions (array_like): Number of regions in each zone.

    Returns:
    - c_caps (ndarray): Colatitudes of caps.
    """
    n_collars = len(n_regions) - 2
    c_caps = zeros(n_collars + 2)
    c_caps[0] = c_polar
    ideal_region_area = get_region_area(dim, N)
    subtotal_n_regions = n_regions[0]
    for collar_n in range(1, n_collars + 1):
        subtotal_n_regions += n_regions[collar_n]
        area = subtotal_n_regions * ideal_region_area
        c_caps[collar_n] = get_cap_colatitude(dim, area)
    c_caps[-1] = pi
    return c_caps


def get_equal_area_caps(dim, N, symmetric: bool = False):
    """
    Partition the sphere into nested spherical caps and get the number
    of regions in each zone.

    Parameters
    ----------
    dim : int
        Dimension of the sphere.
    N : int
        Total number of regions.
    symmetric : bool, optional
        If True, enforce an approximately even number of collars so that
        the partition is symmetric w.r.t. the equatorial hyperplane.
        If False, use the standard Leopardi choice of collars.

    Returns
    -------
    cap_colatitudes : ndarray
        Colatitudes of caps (in increasing order).
    n_regions : ndarray
        Number of regions in each zone.
    """
    if dim == 1:
        sector = arange(1, N + 1)
        cap_colatitudes = sector * 2 * pi / N
        n_regions = ones(N, dtype=int)
        return cap_colatitudes, n_regions

    if N == 1:
        cap_colatitudes = array([pi])
        n_regions = array([1])
        return cap_colatitudes, n_regions

    c_polar = get_polar_cap_colatitude(dim, N)
    ideal_angle = get_ideal_collar_angle(dim, N)

    if symmetric and (N > 2) and (ideal_angle > 0):
        # Symmetric choice: even number of collars so equator is a cap boundary
        ratio_half = 0.5 * (pi - 2 * c_polar) / ideal_angle
        n_half = max(array((0.5, round(ratio_half))))
        n_collars = int(2 * n_half)
    else:
        # Standard Leopardi choice
        n_collars = get_number_of_collars(N, c_polar, ideal_angle)

    region_counts = get_ideal_region_counts(dim, N, c_polar, n_collars)
    n_regions = round_region_counts(region_counts)
    cap_colatitudes = get_cap_colatitudes(dim, N, c_polar, n_regions)
    return cap_colatitudes, n_regions


# pylint: disable=R0914
def get_partition_points_polar(dim, N, extra_offset=False):
    """
    Get the center points of the regions in the EQ partition, in spherical polar coordinates.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.
    - extra_offset (bool): Whether to use extra offsets (experimental).

    Returns:
    - points_s (ndarray): Spherical polar coordinates of the points (shape: [dim, N]).
    """
    if N == 1:
        return zeros((dim, 1))

    if dim == 1:
        points_s = linspace(0, 2 * pi, N, endpoint=False) + pi / N
        points_s = reshape(points_s, (1, N))
        return points_s

    cap_colatitudes, n_regions = get_equal_area_caps(dim, N, symmetric=False)
    n_collars = len(n_regions) - 2  # Excluding the two polar caps
    points_s = zeros((dim, N))
    point_n = 0
    # North polar cap center point
    points_s[:, point_n] = zeros(dim)
    point_n += 1
    # For each collar
    for collar_n in range(n_collars):
        a_top = cap_colatitudes[collar_n]
        a_bot = cap_colatitudes[collar_n + 1]
        n_in_collar = n_regions[collar_n + 1]
        # Recursively partition the (dim-1)-sphere
        points_1 = get_partition_points_polar(dim - 1, n_in_collar, extra_offset)
        num_points = points_1.shape[1]
        # For each point, construct a new point in dim dimensions
        for i in range(num_points):
            point = zeros(dim)
            point[:-1] = points_1[:, i]
            point[-1] = (a_top + a_bot) / 2
            points_s[:, point_n] = point
            point_n += 1
    # South polar cap center point
    points_s[:, point_n] = zeros(dim)
    points_s[-1, point_n] = pi
    point_n += 1
    return points_s


def get_partition_points_polar_north(
    dim: int,
    N: int,
    extra_offset: bool = False,
):
    """
    Symmetric variant of the Leopardi equal-area partition on S^2.

    Parameters
    ----------
    dim : int
        Dimension of the sphere S^dim. Currently only dim=2 is supported.
    N : int
        Total number of regions on the full sphere. Must be even.

    Returns
    -------
    points_s : ndarray
        If delete_half is False: shape (dim, N).
        If delete_half is True:  shape (dim, N/2).
        Columns are hyperspherical coordinates of region centres.
        For dim=2 these are (azimuth, colatitude).
    """

    if N % 2 != 0:
        raise ValueError("Number of points N must be even for symmetric partitions.")
    if N < 2:
        raise ValueError("N must be at least 2.")

    # Build a symmetric equal-area partition in terms of caps/collars,
    # then only take the northern half as the "fundamental domain".
    cap_colatitudes, n_regions = get_equal_area_caps(dim, N, symmetric=True)
    n_collars = len(n_regions) - 2  # excluding the two polar caps

    N_half = N // 2
    points_half = zeros((dim, N_half), dtype=float)

    # North pole
    points_half[:, 0] = 0.0
    point_idx = 1

    # Only collars fully contained in the northern half
    n_collars_half = n_collars // 2
    for collar_n in range(n_collars_half):
        a_top = cap_colatitudes[collar_n]
        a_bot = cap_colatitudes[collar_n + 1]
        n_in_collar = int(n_regions[collar_n + 1])  # skip the north cap entry

        if n_in_collar == 0:
            continue

        # Partition the S^1 factor in this collar
        sub_points = get_partition_points_polar(
            dim - 1, n_in_collar, extra_offset=extra_offset
        )
        colat = 0.5 * (a_top + a_bot)

        for i in range(sub_points.shape[1]):
            if point_idx >= N_half:
                break  # safety
            points_half[:-1, point_idx] = sub_points[:, i]
            points_half[-1, point_idx] = colat
            point_idx += 1

    return points_half


def get_partition_points_cartesian(
    dim: int,
    N: int,
    delete_half: bool = False,
    symmetry_type: str = "asymm",
    extra_offset: bool = False,
):
    """
    Cartesian EQ point set with optional symmetry.

    Parameters
    ----------
    dim : int
        Dimension of the sphere S^dim (embedded in R^{dim+1}).
    N : int
        Number of points on the *full* sphere.
        Must be even for symmetry_type != 'asymm'.
    delete_half : bool, optional
        For symmetric types: if True, only return the 'northern' half
        (N/2 points). For asymm, this is not allowed.
    symmetry_type : {'asymm', 'plane', 'antipodal'}
        - 'asymm'   : standard Leopardi EQ partition (no enforced symmetry).
        - 'plane'   : plane-symmetric w.r.t. the equatorial hyperplane
                      (last Cartesian coordinate flips sign).
        - 'antipodal': point-symmetric w.r.t. the origin (± pairs).
    extra_offset : bool, optional
        Passed through to the polar partitioners (currently unused).

    Returns
    -------
    points_x : ndarray, shape (N_eff, dim+1)
        Cartesian coordinates of the centre points, where
        N_eff = N      if delete_half is False,
        N_eff = N / 2  if delete_half is True and symmetry_type != 'asymm'.
    """
    symmetry_type = symmetry_type.lower()
    if symmetry_type not in ("asymm", "plane", "antipodal"):
        raise ValueError("symmetry_type must be 'asymm', 'plane', or 'antipodal'.")

    # --- Asymmetric case: standard Leopardi mapping via 'colatitude' ---
    if symmetry_type == "asymm":
        if delete_half:
            raise ValueError(
                "delete_half=True is not supported for symmetry_type='asymm'."
            )

        # get_partition_points_polar returns angles in Leopardi order:
        #   for dim=2: (azimuth, colatitude)
        pts_s_leop = get_partition_points_polar(
            dim,
            N,
            extra_offset=extra_offset,
        )

        # Convert to hyperspherical 'colatitude' order expected by hypersph_to_cart:
        #   for dim=2: (colatitude, azimuth)
        pts_s_hyp = flip(pts_s_leop, axis=0).T  # shape (N, dim)

        grid_eucl = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
            pts_s_hyp,
            mode="colatitude",
        )
        return grid_eucl

    # --- Symmetric cases: 'plane' and 'antipodal' ---
    if N % 2 != 0:
        raise ValueError("For symmetric partitions, N must be even.")

    # 1) Build the *northern* half of a symmetric Leopardi partition in polar coords.
    #    delete_half=True means we only get the north half (including north pole),
    #    with no collars crossing the equator (handled by the symmetric caps).
    pts_s_half_leop = get_partition_points_polar_north(
        dim,
        N,
        extra_offset=extra_offset,
    )

    # 2) Convert that half to Cartesian using the same 'colatitude' path
    #    as in the asymmetric case (Leopardi → hyperspherical colatitude).
    pts_s_half_hyp = flip(pts_s_half_leop, axis=0).T  # (N/2, dim)

    north = AbstractHypersphereSubsetDistribution.hypersph_to_cart(
        pts_s_half_hyp,
        mode="colatitude",
    )

    N_half = north.shape[0]

    if delete_half:
        # For symmetric modes, delete_half=True means: only return the northern half.
        return north

    # 3) Enforce symmetry *in Cartesian space*.
    if symmetry_type == "plane":
        # Plane reflection w.r.t. equatorial hyperplane: last coord flips sign
        south = copy.deepcopy(north)
        # First dim becomes last time when using SymmetricLeopardiSampler
        south[:, 0] *= -1.0
    elif symmetry_type == "antipodal":
        # Antipodal symmetry: ± pairs
        south = -north
    else:
        # Should not happen due to check at top
        raise ValueError("Invalid symmetry_type encountered.")

    # 4) Stack north and south to get N points on S^dim
    if 2 * N_half != N:
        raise RuntimeError(
            f"Inconsistent hemisphere size: N={N}, N_half={N_half} (expected N_half={N//2})."
        )

    points_x = vstack((north, south))

    return points_x
