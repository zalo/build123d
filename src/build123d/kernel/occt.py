"""
build123d kernel facade

name: occt.py
by:   Gumyr
date: October 12, 2026

desc:
    The OpenCascade implementation of :class:`~build123d.kernel.base.Kernel`
    over the OCP bindings. Handles are ``TopoDS_Shape`` objects. These are
    the routines that used to live as private helpers in
    ``topology/shape_core.py``; they are kept as thin functions so that the
    hot paths (entity enumeration, hashing, equality) do not pay for an
    extra layer of Python objects.

license:

    Copyright 2026 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from __future__ import annotations

from collections.abc import Callable, Hashable

import OCP.TopAbs as ta
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Shape

from build123d.kernel.base import Kernel, TopoKind

_KIND_TO_TOPABS = {
    TopoKind.VERTEX: ta.TopAbs_VERTEX,
    TopoKind.EDGE: ta.TopAbs_EDGE,
    TopoKind.WIRE: ta.TopAbs_WIRE,
    TopoKind.FACE: ta.TopAbs_FACE,
    TopoKind.SHELL: ta.TopAbs_SHELL,
    TopoKind.SOLID: ta.TopAbs_SOLID,
    TopoKind.COMPSOLID: ta.TopAbs_COMPSOLID,
    TopoKind.COMPOUND: ta.TopAbs_COMPOUND,
}
_TOPABS_TO_KIND = {value: key for key, value in _KIND_TO_TOPABS.items()}

_DOWNCAST = {
    ta.TopAbs_VERTEX: TopoDS.Vertex,
    ta.TopAbs_EDGE: TopoDS.Edge,
    ta.TopAbs_WIRE: TopoDS.Wire,
    ta.TopAbs_FACE: TopoDS.Face,
    ta.TopAbs_SHELL: TopoDS.Shell,
    ta.TopAbs_SOLID: TopoDS.Solid,
    ta.TopAbs_COMPSOLID: TopoDS.CompSolid,
    ta.TopAbs_COMPOUND: TopoDS.Compound,
}

_MASS_PROPERTIES: dict[TopoKind, Callable[[TopoDS_Shape, GProp_GProps], None]] = {
    TopoKind.VERTEX: BRepGProp.LinearProperties_s,
    TopoKind.EDGE: BRepGProp.LinearProperties_s,
    TopoKind.WIRE: BRepGProp.LinearProperties_s,
    TopoKind.FACE: BRepGProp.SurfaceProperties_s,
    TopoKind.SHELL: BRepGProp.SurfaceProperties_s,
    TopoKind.SOLID: BRepGProp.VolumeProperties_s,
    TopoKind.COMPSOLID: BRepGProp.VolumeProperties_s,
    TopoKind.COMPOUND: BRepGProp.VolumeProperties_s,
}


class OcctKernel(Kernel):
    """OpenCascade through OCP."""

    name = "occt"

    # ---- handles and identity ------------------------------------------------

    def kind(self, shape: TopoDS_Shape) -> TopoKind:
        if shape is None or shape.IsNull():
            raise ValueError("Null TopoDS_Shape object")
        return _TOPABS_TO_KIND[shape.ShapeType()]

    def is_null(self, shape: TopoDS_Shape | None) -> bool:
        return shape is None or shape.IsNull()

    def downcast(self, shape: TopoDS_Shape) -> TopoDS_Shape:
        if shape is None or shape.IsNull():
            raise ValueError("Null TopoDS_Shape object")
        return _DOWNCAST[shape.ShapeType()](shape)

    def is_same(self, a: TopoDS_Shape, b: TopoDS_Shape) -> bool:
        return a.IsSame(b)

    def is_equal(self, a: TopoDS_Shape, b: TopoDS_Shape) -> bool:
        return a.IsEqual(b)

    def is_partner(self, a: TopoDS_Shape, b: TopoDS_Shape) -> bool:
        return a.IsPartner(b)

    def hash_key(self, shape: TopoDS_Shape) -> Hashable:
        return hash(shape)

    def copy_deep(self, shape: TopoDS_Shape) -> TopoDS_Shape:
        return self.downcast(BRepBuilderAPI_Copy(shape).Shape())

    # ---- traversal ----------------------------------------------------------

    def sub_shapes(self, shape: TopoDS_Shape, kind: TopoKind) -> list[TopoDS_Shape]:
        found: dict[int, TopoDS_Shape] = {}  # a dict so repeats collapse
        explorer = TopExp_Explorer(shape, _KIND_TO_TOPABS[kind])
        while explorer.More():
            item = explorer.Current()
            found[hash(item)] = item  # a seam edge is visited once per use
            explorer.Next()
        return list(found.values())

    def sub_shape_uses(self, shape: TopoDS_Shape, kind: TopoKind) -> list[TopoDS_Shape]:
        found: list[TopoDS_Shape] = []
        explorer = TopExp_Explorer(shape, _KIND_TO_TOPABS[kind])
        while explorer.More():
            found.append(explorer.Current())
            explorer.Next()
        return found

    # ---- validity and repair ------------------------------------------------

    def is_valid(self, shape: TopoDS_Shape) -> bool:
        checker = BRepCheck_Analyzer(shape)
        checker.SetParallel(True)
        return checker.IsValid()

    def fix(self, shape: TopoDS_Shape) -> TopoDS_Shape:
        shape_fix = ShapeFix_Shape(shape)
        shape_fix.Perform()
        return self.downcast(shape_fix.Shape())

    # ---- measures -----------------------------------------------------------

    def area(self, shape: TopoDS_Shape) -> float:
        properties = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, properties)
        return properties.Mass()

    def volume(self, shape: TopoDS_Shape) -> float:
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, properties)
        return properties.Mass()

    def length(self, shape: TopoDS_Shape) -> float:
        properties = GProp_GProps()
        BRepGProp.LinearProperties_s(shape, properties)
        return properties.Mass()

    def center_of_mass(
        self, shape: TopoDS_Shape, kind: TopoKind
    ) -> tuple[float, float, float]:
        properties = GProp_GProps()
        _MASS_PROPERTIES[kind](shape, properties)
        center = properties.CentreOfMass()
        return (center.X(), center.Y(), center.Z())


def kind_to_topabs(kind: TopoKind) -> ta.TopAbs_ShapeEnum:
    """The OCCT enum for ``kind`` (for modules that still drive OCP directly)."""
    return _KIND_TO_TOPABS[kind]
