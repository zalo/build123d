"""Tests for the kernel facade that the topology layer talks to."""

import unittest

from build123d import Box, Cylinder, Location, Plane, Solid, Vector
from build123d.kernel import TopoKind, OcctKernel, kernel


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
