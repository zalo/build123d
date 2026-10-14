"""Tests for the kernel facade that the topology layer talks to."""

import unittest

from math import pi

from build123d import Box, Cylinder, Location, Plane, Solid, Vector
from build123d.kernel import (
    BoolOp,
    DraftFailure,
    JoinKind,
    OcctKernel,
    TopoKind,
    WireStatus,
    kernel,
)


class TestKernelFacade(unittest.TestCase):
    def test_kinds(self):
        box = Solid.make_box(1, 2, 3)
        self.assertEqual(kernel.kind(box.wrapped), TopoKind.SOLID)
        self.assertEqual(kernel.kind(Box(1, 2, 3).wrapped), TopoKind.COMPOUND)
        self.assertEqual(kernel.kind(box.faces()[0].wrapped), TopoKind.FACE)
        self.assertTrue(kernel.is_kind(box.wrapped, TopoKind.SOLID))
        self.assertFalse(kernel.is_kind(None, TopoKind.SOLID))
        self.assertEqual(TopoKind.from_label("Face"), TopoKind.FACE)
        self.assertEqual(TopoKind.FACE.label, "Face")
        with self.assertRaises(ValueError):
            kernel.kind(None)

    def test_sub_shapes_and_uses(self):
        cylinder = Cylinder(1, 2)
        curved = [f for f in cylinder.faces() if f.geom_type.name == "CYLINDER"][0]
        distinct = kernel.sub_shapes(curved.wrapped, TopoKind.EDGE)
        uses = kernel.sub_shape_uses(curved.wrapped, TopoKind.EDGE)
        self.assertEqual(len(distinct), 3)  # two circles and the seam
        self.assertEqual(len(uses), 4)  # the seam is used twice
        self.assertEqual(
            len(kernel.sub_shapes(Box(1, 1, 1).wrapped, TopoKind.VERTEX)), 8
        )

    def test_identity(self):
        box = Box(1, 1, 1)
        face = box.faces()[0].wrapped
        again = box.faces()[0].wrapped
        self.assertTrue(kernel.is_same(face, again))
        self.assertTrue(kernel.is_partner(face, again))
        self.assertEqual(kernel.hash_key(face), kernel.hash_key(again))
        self.assertFalse(kernel.is_same(face, box.faces()[1].wrapped))
        copy = kernel.copy_deep(box.wrapped)
        self.assertFalse(kernel.is_same(copy, box.wrapped))
        self.assertAlmostEqual(kernel.volume(copy), 1.0)

    def test_measures(self):
        box = Box(1, 2, 3)
        self.assertAlmostEqual(kernel.volume(box.wrapped), 6.0)
        self.assertAlmostEqual(kernel.area(box.wrapped), 22.0)
        self.assertAlmostEqual(
            kernel.length(box.edges()[0].wrapped), box.edges()[0].length
        )
        self.assertEqual(
            Vector(kernel.center_of_mass(box.wrapped, TopoKind.SOLID)), Vector(0, 0, 0)
        )
        self.assertTrue(kernel.is_valid(box.wrapped))
        self.assertTrue(kernel.is_valid(kernel.fix(box.wrapped)))

    def test_backend_name(self):
        self.assertIsInstance(kernel, OcctKernel)
        self.assertEqual(kernel.name, "occt")


if __name__ == "__main__":
    unittest.main()


class TestKernelFacadeOperations(unittest.TestCase):
    def test_bounding_box(self):
        box = Solid.make_box(1, 2, 3)
        corners = kernel.bounding_box(box.wrapped)
        self.assertIsNotNone(corners)
        xmin, ymin, zmin, xmax, ymax, zmax = corners
        self.assertAlmostEqual(xmax - xmin, 1.0, 5)
        self.assertAlmostEqual(ymax - ymin, 2.0, 5)
        self.assertAlmostEqual(zmax - zmin, 3.0, 5)
        quick = kernel.bounding_box(box.wrapped, optimal=False)
        self.assertIsNotNone(quick)

    def test_boolean_and_history(self):
        from build123d.kernel import BoolOp

        a = Solid.make_box(2, 2, 2).wrapped
        b = Solid.make_box(1, 1, 1).wrapped
        cut, record = kernel.boolean(BoolOp.CUT, [a], [b])
        self.assertAlmostEqual(kernel.volume(cut), 7.0, 5)
        self.assertIsNotNone(record)
        fused, _ = kernel.boolean(BoolOp.FUSE, [a], [b], fuzzy=1e-6)
        self.assertAlmostEqual(kernel.volume(fused), 8.0, 5)
        common, _ = kernel.boolean(BoolOp.COMMON, [a], [b])
        self.assertAlmostEqual(kernel.volume(common), 1.0, 5)
        split, _ = kernel.boolean(BoolOp.SPLIT, [a], [b])
        self.assertEqual(len(kernel.sub_shapes(split, TopoKind.SOLID)), 2)

    def test_section_and_unify(self):
        a = Solid.make_box(2, 2, 2).wrapped
        b = Solid.make_box(2, 2, 2, Plane((1, 0, 0))).wrapped
        curves, _ = kernel.section(a, b)
        self.assertGreaterEqual(len(kernel.sub_shapes(curves, TopoKind.EDGE)), 4)
        from build123d.kernel import BoolOp

        fused, _ = kernel.boolean(BoolOp.FUSE, [a], [b])
        unified, _ = kernel.unify_same_domain(fused)
        self.assertEqual(len(kernel.sub_shapes(unified, TopoKind.FACE)), 6)

    def test_placement(self):
        box = Solid.make_box(1, 1, 1)
        identity = kernel.identity_location()
        self.assertTrue(kernel.location_of(box.wrapped).IsIdentity())
        shift = Location((1, 2, 3)).wrapped
        moved = kernel.moved(box.wrapped, shift)
        self.assertTrue(kernel.is_partner(moved, box.wrapped))
        self.assertFalse(kernel.is_same(moved, box.wrapped))
        back = kernel.relocation_between(moved, box.wrapped)
        self.assertTrue(kernel.is_same(kernel.moved(moved, back), box.wrapped))
        placed = kernel.located(moved, identity)
        self.assertTrue(kernel.is_same(placed, box.wrapped))
        kernel.set_location(placed, shift)
        self.assertTrue(kernel.is_same(placed, moved))


class TestKernelFacadeGeometry(unittest.TestCase):
    def test_curve_view(self):
        from build123d import Cylinder

        cylinder = Cylinder(1, 2)
        circle = [e for e in cylinder.edges() if e.geom_type.name == "CIRCLE"][0]
        view = kernel.curve_of(circle.wrapped)
        self.assertEqual(view.geom_type, "CIRCLE")
        self.assertAlmostEqual(view.length(), 2 * 3.141592653589793, 6)
        center, axis, radius = view.circle()
        self.assertAlmostEqual(radius, 1.0, 6)
        self.assertAlmostEqual(abs(axis[2]), 1.0, 6)
        mid = view.param_at_length(view.length() / 2)
        self.assertAlmostEqual(
            sum((a - b) ** 2 for a, b in zip(view.value(mid), center)) ** 0.5, 1.0, 6
        )
        self.assertTrue(view.is_periodic())
        self.assertGreater(len(view.positions_by_deflection(0.01)), 8)
        self.assertAlmostEqual(
            view.closest_param(view.value(view.first)), view.first, 6
        )

    def test_surface_view(self):
        from build123d import Cylinder

        cylinder = Cylinder(1, 2)
        curved = [f for f in cylinder.faces() if f.geom_type.name == "CYLINDER"][0]
        view = kernel.surface_of(curved.wrapped)
        self.assertEqual(view.geom_type, "CYLINDER")
        self.assertAlmostEqual(view.radius(), 1.0, 6)
        self.assertTrue(view.is_u_periodic())
        point, normal = view.normal(0.0, 0.0)
        self.assertAlmostEqual(abs(normal[0]), 1.0, 6)
        self.assertIsNone(view.plane())
        flat = [f for f in cylinder.faces() if f.geom_type.name == "PLANE"][0]
        self.assertIsNotNone(kernel.surface_of(flat.wrapped).plane())

    def test_construction(self):
        from build123d import Plane
        from build123d.kernel import JoinKind, SweepTransition

        frame = Plane.XY.to_gp_ax2()
        box = kernel.make_box(frame, 1, 2, 3)
        self.assertAlmostEqual(kernel.volume(box), 6.0, 6)
        self.assertAlmostEqual(
            kernel.volume(kernel.make_cylinder(frame, 1, 1, 2 * 3.141592653589793)),
            3.141592653589793,
            5,
        )
        face = kernel.sub_shapes(box, TopoKind.FACE)[0]
        prism = kernel.extrude(face, (1.0, 0.0, 0.0))
        self.assertEqual(kernel.kind(prism), TopoKind.SOLID)
        self.assertAlmostEqual(kernel.volume(prism), 6.0, 6)
        filleted, record = kernel.fillet(
            box, kernel.sub_shapes(box, TopoKind.EDGE)[:1], 0.25
        )
        self.assertEqual(len(kernel.sub_shapes(filleted, TopoKind.FACE)), 7)
        self.assertIsNotNone(record)
        shelled = kernel.thick_solid(box, [face], -0.1, 1e-4, JoinKind.ARC)
        self.assertLess(kernel.volume(shelled), 6.0)
        self.assertIsInstance(SweepTransition.ROUND, SweepTransition)

    def test_tessellation(self):
        box = Solid.make_box(1, 1, 1)
        kernel.mesh(box.wrapped, 0.1, 0.1)
        face = kernel.sub_shapes(box.wrapped, TopoKind.FACE)[0]
        triangulation = kernel.face_triangulation(face)
        self.assertIsNotNone(triangulation)
        points, triangles = triangulation
        self.assertEqual(len(points), 4)
        self.assertEqual(len(triangles), 2)


class TestKernelFacadeTraversalAndBuilders(unittest.TestCase):
    def test_children_ancestors_and_vertices(self):
        box = Solid.make_box(1, 2, 3).wrapped
        self.assertEqual(len(kernel.direct_children(box)), 1)  # one shell
        edge_faces = kernel.ancestors(box, TopoKind.EDGE, TopoKind.FACE)
        self.assertEqual(len(edge_faces), 12)
        edge = edge_faces.keys()[0]
        self.assertEqual(len(edge_faces.get(edge)), 2)
        self.assertIn(kernel.reversed(edge), edge_faces)  # orientation is ignored
        first, last = kernel.edge_vertices(edge)
        self.assertEqual(kernel.kind(first), TopoKind.VERTEX)
        self.assertNotEqual(kernel.vertex_point(first), kernel.vertex_point(last))
        self.assertTrue(kernel.is_forward(edge) or kernel.is_reversed(edge))
        self.assertFalse(kernel.same_orientation(edge, kernel.reversed(edge)))

    def test_vertex_wire_face_builders(self):
        a, b, c = kernel.make_vertex((0, 0, 0)), (1, 0, 0), (1, 1, 0)
        self.assertEqual(kernel.vertex_point(a), (0.0, 0.0, 0.0))
        edges = [
            kernel.make_line((0, 0, 0), b),
            kernel.make_line(b, c),
            kernel.make_line(c, (0, 0, 0)),
        ]
        wire, status = kernel.make_wire(edges)
        self.assertEqual(status, WireStatus.OK)
        self.assertEqual(len(kernel.wire_edges(wire)), 3)
        self.assertTrue(kernel.is_closed(wire))
        _, status = kernel.make_wire([edges[0], kernel.make_line((5, 5, 5), (6, 5, 5))])
        self.assertEqual(status, WireStatus.DISCONNECTED)
        placed_wire, placed = kernel.make_wire_placed(edges)
        self.assertEqual(len(placed), 3)
        face = kernel.make_planar_face(placed_wire, [])
        self.assertAlmostEqual(kernel.area(face), 0.5)
        self.assertEqual(kernel.kind(kernel.outer_wire(face)), TopoKind.WIRE)
        self.assertEqual(
            kernel.kind(kernel.make_compound([face, wire])), TopoKind.COMPOUND
        )
        self.assertEqual(kernel.kind(kernel.make_shell([face])), TopoKind.SHELL)
        sewn, _ = kernel.sew(
            kernel.sub_shapes(Solid.make_box(1, 1, 1).wrapped, TopoKind.FACE)
        )
        self.assertEqual(kernel.kind(sewn), TopoKind.SHELL)
        self.assertAlmostEqual(kernel.volume(kernel.solid_from_shell(sewn)), 1.0)
        self.assertEqual(
            len(kernel.make_polygon([(0, 0, 0), b, c, (0, 0, 0)]).__class__.__name__),
            11,
        )

    def test_distances_and_classification(self):
        box = Solid.make_box(1, 1, 1).wrapped
        far = kernel.make_vertex((3, 0.5, 0.5))
        self.assertAlmostEqual(kernel.distance(box, far), 2.0)
        separation, on_box, on_vertex = kernel.closest_points(box, far)
        self.assertAlmostEqual(separation, 2.0)
        self.assertAlmostEqual(on_box[0], 1.0)
        self.assertEqual(on_vertex, (3.0, 0.5, 0.5))
        separation, pairs = kernel.extrema(box, kernel.make_vertex((0.5, 0.5, 1)))
        self.assertAlmostEqual(separation, 0.0)
        self.assertGreaterEqual(len(pairs), 1)
        self.assertTrue(kernel.point_in_solid(box, (0.5, 0.5, 0.5), 1e-6))
        self.assertFalse(kernel.point_in_solid(box, (2, 2, 2), 1e-6))
        face = kernel.sub_shapes(box, TopoKind.FACE)[0]
        surface = kernel.surface_of(face)
        u_min, u_max, v_min, v_max = surface.uv_bounds()
        middle = ((u_min + u_max) / 2, (v_min + v_max) / 2)
        self.assertTrue(kernel.point_on_face(face, surface.value(*middle), 1e-6))
        self.assertTrue(kernel.uv_in_face(face, middle, 1e-6))
        self.assertFalse(kernel.uv_in_face(face, (5.0, 5.0), 1e-6))
        self.assertEqual(
            len(kernel.line_intersections(box, (0.5, 0.5, -1), (0, 0, 1), 1e-6)), 2
        )

    def test_curves(self):
        frame = Plane.XY.to_gp_ax2()
        arc = kernel.make_circle_arc(frame, 2.0, 0.0, pi / 2, True)
        view = kernel.curve_of(arc)
        self.assertEqual(view.geom_type, "CIRCLE")
        self.assertAlmostEqual(view.circle()[2], 2.0)
        self.assertAlmostEqual(view.length(), pi)
        self.assertAlmostEqual(
            kernel.curve_of(kernel.make_circle(frame, 1.0)).length(), 2 * pi
        )
        ellipse = kernel.curve_of(kernel.make_ellipse(frame, 3.0, 2.0))
        self.assertEqual(ellipse.ellipse()[2:], (3.0, 2.0))
        bezier = kernel.curve_of(kernel.make_bezier([(0, 0, 0), (1, 1, 0), (2, 0, 0)]))
        self.assertEqual(bezier.bezier()[0], 2)
        spline = kernel.make_spline_interpolated(
            [(0, 0, 0), (1, 1, 0), (2, 0, 0)], periodic=False, tolerance=1e-6
        )
        self.assertEqual(kernel.curve_of(spline).geom_type, "BSPLINE")
        degree, periodic, poles, _, knots, mults = kernel.curve_of(spline).bspline()
        self.assertEqual((degree >= 2, periodic), (True, False))
        self.assertEqual(len(knots), len(mults))
        rebuilt = kernel.make_bspline(poles, knots, mults, degree, periodic)
        self.assertAlmostEqual(
            kernel.curve_of(rebuilt).length(), kernel.curve_of(spline).length()
        )
        helix = kernel.make_helix((0, 0, 0), (0, 0, 1), 1.0, 1.0, 2.0, 0.0, False)
        self.assertAlmostEqual(kernel.curve_of(helix).value(0.0)[0], 1.0)
        trimmed = kernel.trim_edge(arc, 0.0, pi / 4)
        self.assertAlmostEqual(kernel.curve_of(trimmed).length(), pi / 2)
        self.assertAlmostEqual(
            kernel.curve_of(kernel.trim_edge_by_length(arc, 0.0, 1.0)).length(), 1.0
        )
        line = kernel.make_line((0, 0, 0), (1, 0, 0))
        self.assertEqual(kernel.edge_range(line), (0.0, 1.0))
        self.assertEqual(
            kernel.vertex_param_on_edge(kernel.edge_vertices(line)[1], line), 1.0
        )
        self.assertEqual(
            kernel.find_plane(
                kernel.make_polygon([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)])
            )[1],
            (0.0, 0.0, 1.0),
        )
        self.assertEqual(
            kernel.planar_intersections(
                kernel.make_line((0, 0, 0), (2, 2, 0)),
                (0, 1),
                kernel.make_line((0, 2, 0), (2, 0, 0)),
                (0, 1),
                Plane.XY.wrapped,
                1e-6,
            ),
            [(1.0, 1.0)],
        )

    def test_operations(self):
        box = Solid.make_box(1, 1, 1).wrapped
        top = max(
            kernel.sub_shapes(box, TopoKind.FACE),
            key=lambda f: kernel.center_of_mass(f, TopoKind.FACE)[2],
        )
        side = min(
            kernel.sub_shapes(box, TopoKind.FACE),
            key=lambda f: kernel.center_of_mass(f, TopoKind.FACE)[0],
        )
        drafted, _ = kernel.draft(box, [(side, 0.1)], (0, 0, 1), Plane.XY.wrapped)
        self.assertNotAlmostEqual(kernel.volume(drafted), 1.0)
        with self.assertRaises(DraftFailure):
            kernel.draft(box, [(top, 0.1)], (0, 0, 1), Plane.XY.wrapped)
        prism, generated, _ = kernel.extrude_generated(top, (0, 0, 1))
        self.assertAlmostEqual(kernel.volume(prism), 1.0)
        self.assertEqual(len(generated), 4)
        swept = kernel.sweep_multi(
            [kernel.outer_wire(top)],
            kernel.make_wire([kernel.make_line((0.5, 0.5, 1), (0.5, 0.5, 3))])[0],
            is_frenet=False,
            make_solid=True,
        )
        self.assertAlmostEqual(kernel.volume(swept), 2.0)
        offset = kernel.offset_shape(
            top,
            0.5,
            1e-5,
            join=JoinKind.INTERSECTION,
            intersection=True,
            thicken=True,
            remove_internal_edges=True,
        )
        self.assertAlmostEqual(kernel.volume(offset), 0.5)
        lefts, rights = kernel.split_by_edges(
            top, [kernel.make_line((0.5, 0, 1), (0.5, 1, 1))]
        )
        self.assertEqual((len(lefts), len(rights)), (1, 1))
        transformed, _ = kernel.transformed(
            box, Location((1, 0, 0)).wrapped.Transformation()
        )
        self.assertAlmostEqual(
            kernel.center_of_mass(transformed, TopoKind.SOLID)[0], 1.5
        )
        self.assertEqual(len(kernel.inertia_matrix(box)), 3)
        self.assertEqual(len(kernel.principal_inertia(box)), 3)
        self.assertFalse(kernel.has_inconsistent_orientation(box))
        self.assertEqual(
            kernel.surface_axis(
                kernel.sub_shapes(Cylinder(1, 2).wrapped, TopoKind.FACE)[0]
            )[0],
            "CYLINDER",
        )
        filling = kernel.filling()
        for edge in kernel.sub_shapes(top, TopoKind.EDGE):
            filling.add_edge(edge, "C0")
        self.assertAlmostEqual(kernel.area(filling.build()), 1.0, places=3)
        split, _ = kernel.boolean(
            BoolOp.SPLIT, [box], [kernel.face_of_plane(Plane.XY.offset(0.5).wrapped)]
        )
        self.assertEqual(len(kernel.sub_shapes(split, TopoKind.SOLID)), 2)
