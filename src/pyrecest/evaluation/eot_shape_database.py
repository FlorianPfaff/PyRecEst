# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, linspace, pi, random, sin, to_numpy
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union


def _uniform_scalar(low: float = 0.0, high: float = 1.0) -> float:
    return float(to_numpy(random.uniform(size=()) * (high - low) + low))


class PolygonWithSampling(Polygon):  # pylint: disable=abstract-method
    __slots__ = Polygon.__slots__

    def __new__(cls, shell=None, holes=None):  # nosec
        polygon = super().__new__(cls, shell=shell, holes=holes)  # nosec  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        polygon.__class__ = cls
        return polygon

    def sample_on_boundary(self, num_points: int):
        points = []
        perimeter = self.length

        if isinstance(self.boundary, LineString):
            lines = [self.boundary]
        elif isinstance(self.boundary, MultiLineString):
            lines = list(self.boundary.geoms)
        else:
            raise ValueError("Unsupported boundary type")

        for _ in range(num_points):
            # Generate a random distance along the perimeter
            distance = _uniform_scalar(0.0, perimeter)

            # Traverse the edges to place the point
            for line in lines:
                if distance < line.length:
                    points.append(line.interpolate(distance))
                    break
                distance -= line.length

        return array([(point.x, point.y) for point in points])

    def sample_within(self, num_points: int):
        min_x, min_y, max_x, max_y = self.bounds
        points = []

        for _ in range(num_points):
            random_point = Point(
                [
                    _uniform_scalar(min_x, max_x),
                    _uniform_scalar(min_y, max_y),
                ]
            )
            while not random_point.within(self):
                random_point = Point(
                    [
                        _uniform_scalar(min_x, max_x),
                        _uniform_scalar(min_y, max_y),
                    ]
                )

            points.append(random_point)

        return array([(point.x, point.y) for point in points])


class StarShapedPolygon(PolygonWithSampling):  # pylint: disable=abstract-method
    __slots__ = Polygon.__slots__
    # Inheriting __new__ of PolygonWithSampling

    def is_convex(self):
        return self.area == self.convex_hull.area

    def compute_kernel(self):
        if self.is_convex():
            return self

        # Compute the kernel of a star-convex polygon
        # Create a visibility polygon for each vertex and intersect them all
        vertices = list(self.exterior.coords)[
            :-1
        ]  # omit the last one because it's the same as the first
        visibility_polygons = []

        for vertex in vertices:
            vis_poly = self.compute_visibility_polygon(Point(vertex))
            visibility_polygons.append(vis_poly)

        # Intersect all visibility polygons to get the kernel
        kernel = visibility_polygons[0]
        for vis_poly in visibility_polygons[1:]:
            kernel = kernel.intersection(vis_poly)

        return kernel

    def compute_visibility_polygon(self, point):
        # Compute the visibility polygon for a given point inside a polygon

        # List to store segments of the visibility polygon
        segments = []

        # Get vertices of the polygon
        vertices = list(self.exterior.coords)[
            :-1
        ]  # omit the last one because it's the same as the first

        for vertex in vertices:
            # Check if the vertex is visible from the given point
            line_of_sight = LineString([point, vertex])
            if self.contains(line_of_sight):
                segments.append(line_of_sight)

            # Check for intersections along the direction to the vertex
            direction = array(vertex) - array(point.coords[0])
            far_away_point = Point(to_numpy(array(vertex) + 1000 * direction))
            ray = LineString([point, far_away_point])

            # Find intersection points with the polygon boundary
            intersections = ray.intersection(self.boundary)

            if intersections.geom_type == "Point":
                segments.append(LineString([point, intersections]))
            elif intersections.geom_type == "MultiPoint":
                # If there are multiple intersection points, choose the closest one
                closest_point = min(intersections.geoms, key=point.distance)
                segments.append(LineString([point, closest_point]))

        # Create the visibility polygon by taking the union of all visible segments
        visibility_polygon = unary_union(segments).convex_hull

        return visibility_polygon


class Star(StarShapedPolygon):  # pylint: disable=abstract-method
    __slots__ = Polygon.__slots__

    def __new__(cls, radius=1, arms=5, arm_width=0.3, center=(0, 0)):
        arm_angle = 2 * pi / arms
        points = []
        for i in range(arms):
            base_angle = i * arm_angle
            inner_angle = base_angle + arm_angle / 2
            # External point
            points.append(
                (
                    center[0] + radius * cos(base_angle),
                    center[1] + radius * sin(base_angle),
                )
            )
            # Internal point
            points.append(
                (
                    center[0] + arm_width * cos(inner_angle),
                    center[1] + arm_width * sin(inner_angle),
                )
            )
        # Close the loop
        points.append(points[0])

        polygon = super().__new__(cls, shell=points, holes=None)  # nosec
        polygon.__class__ = cls
        return polygon


class Cross(StarShapedPolygon):  # pylint: disable=abstract-method
    __slots__ = Polygon.__slots__

    # pylint: disable=signature-differs,too-many-positional-arguments
    def __new__(cls, height_1, height_2, width_1, width_2, centroid=None):
        # Assertions to check conditions
        assert width_1 > height_2, "width_1 has to be larger than height_2"
        assert width_2 > height_1, "width_2 has to be larger than height_1"

        # Use a default centroid if none is provided
        if centroid is None:
            centroid = array([0.0, 0.0])
        else:
            centroid = array(centroid)

        # Calculate half dimensions for clarity
        half_height_1 = height_1 / 2
        half_height_2 = height_2 / 2
        half_width_1 = width_1 / 2
        half_width_2 = width_2 / 2

        # Define polygon points
        polygon_points = array(
            [
                [half_width_1, half_height_1],
                [half_height_2, half_height_1],
                [half_height_2, half_width_2],
                [-half_height_2, half_width_2],
                [-half_height_2, half_height_1],
                [-half_width_1, half_height_1],
                [-half_width_1, -half_height_1],
                [-half_height_2, -half_height_1],
                [-half_height_2, -half_width_2],
                [half_height_2, -half_width_2],
                [half_height_2, -half_height_1],
                [half_width_1, -half_height_1],
            ]
        )

        # Adjust points by centroid
        polygon_points = polygon_points + centroid

        # Create polygon instance
        polygon = super().__new__(
            cls, shell=to_numpy(polygon_points), holes=None
        )  # nosec
        polygon.__class__ = cls

        return polygon


class StarFish(StarShapedPolygon):  # pylint: disable=abstract-method
    __slots__ = Polygon.__slots__

    # pylint: disable=signature-differs
    def __new__(cls, scaling_factor=1):
        theta = linspace(0, 2 * pi, 1000)
        r = 5 + 1.5 * sin(6 * theta)

        x = r * cos(theta) * scaling_factor
        y = r * sin(theta) * scaling_factor

        # Create polygon instance
        polygon = super().__new__(
            cls, shell=zip(to_numpy(x), to_numpy(y)), holes=None
        )  # nosec
        polygon.__class__ = cls

        return polygon
