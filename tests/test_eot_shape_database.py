import unittest

from pyrecest.evaluation.eot_shape_database import (
    Cross,
    PolygonWithSampling,
    Star,
    StarFish,
    StarShapedPolygon,
)
from shapely.geometry import LineString, Point, Polygon
from shapely.plotting import plot_polygon


class TestStarShapedPolygon(unittest.TestCase):
    def setUp(self):
        self.square_star_convex_poly = StarShapedPolygon(
            [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]
        )

    def test_area(self):
        square_poly = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])
        self.assertEqual(self.square_star_convex_poly.area, square_poly.area)

    def test_is_convex(self):
        self.assertTrue(self.square_star_convex_poly.is_convex())

    def test_compute_kernel_convex(self):
        self.assertEqual(
            self.square_star_convex_poly.compute_kernel().area,
            self.square_star_convex_poly.area,
        )

    def test_sample_on_boundary_with_hole(self):
        polygon_with_hole = PolygonWithSampling(
            [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)],
            holes=[
                [
                    (-0.25, -0.25),
                    (-0.25, 0.25),
                    (0.25, 0.25),
                    (0.25, -0.25),
                ]
            ],
        )

        self.assertEqual(polygon_with_hole.boundary.geom_type, "MultiLineString")
        samples = polygon_with_hole.sample_on_boundary(5)

        self.assertEqual(samples.shape, (5, 2))

    def test_zero_point_samples_keep_coordinate_dimension(self):
        boundary_samples = self.square_star_convex_poly.sample_on_boundary(0)
        within_samples = self.square_star_convex_poly.sample_within(0)

        self.assertEqual(boundary_samples.shape, (0, 2))
        self.assertEqual(within_samples.shape, (0, 2))

    def test_degenerate_polygon_rejects_positive_sample_requests(self):
        degenerate_polygon = PolygonWithSampling()

        with self.assertRaisesRegex(ValueError, "positive finite perimeter"):
            degenerate_polygon.sample_on_boundary(1)
        with self.assertRaisesRegex(ValueError, "positive finite area"):
            degenerate_polygon.sample_within(1)

    def test_degenerate_polygon_still_allows_zero_samples(self):
        degenerate_polygon = PolygonWithSampling()

        self.assertEqual(degenerate_polygon.sample_on_boundary(0).shape, (0, 2))
        self.assertEqual(degenerate_polygon.sample_within(0).shape, (0, 2))


class TestStar(unittest.TestCase):
    def test_compute_kernel_star_convex(self):
        star_full = Star(0.5)
        star_kernel = star_full.compute_kernel()
        # Kernel no more than 50% of the area of the original polygon
        self.assertLess(star_kernel.area, 0.5 * star_full.area)


class TestCross(unittest.TestCase):
    def setUp(self) -> None:
        self.cross_full = Cross(2.0, 1.0, 2.0, 3.0)
        self.cross_kernel = self.cross_full.compute_kernel()

    def test_area(self):
        self.assertEqual(self.cross_full.area, 5)

    def test_rejects_invalid_arm_dimensions(self):
        with self.assertRaisesRegex(ValueError, "width_1"):
            Cross(2.0, 1.0, 1.0, 3.0)
        with self.assertRaisesRegex(ValueError, "width_2"):
            Cross(2.0, 1.0, 2.0, 2.0)

    def test_compute_kernel_cross_convex(self):
        # Determining the kernel of a cross-shaped polygon is trivial
        self.assertEqual(self.cross_kernel.area, 2)

    def test_plotting(self):
        plot_polygon(self.cross_full)
        plot_polygon(self.cross_kernel, color="red")


class TestStarfish(unittest.TestCase):
    def setUp(self):
        self.starfish = StarFish()
        # self.starfish_kernel = self.starfish.compute_kernel()

    def test_circle_containment(self):
        # Create a circle with a radius of 2
        circle = Point(0, 0).buffer(2)
        self.assertTrue(
            self.starfish.contains(circle), "The polygon does not contain the circle."
        )

    def test_line_containment(self):
        # Create a line from (-2,-4) to (2,4)
        line = LineString([(-2, -4), (2, 4)])
        self.assertTrue(
            self.starfish.contains(line), "The polygon does not contain the line."
        )

    def test_plotting(self):
        plot_polygon(self.starfish)


if __name__ == "__main__":
    unittest.main()
