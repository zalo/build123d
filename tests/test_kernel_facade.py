"""Tests for the kernel facade that the topology layer talks to."""

import unittest

from build123d import Box, Cylinder, Solid, Vector
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
