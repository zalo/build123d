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

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import OCP.TopAbs as ta
from OCP.Bnd import Bnd_Box
from OCP.BOPAlgo import BOPAlgo_GlueEnum
from OCP.BRepAlgoAPI import (
    BRepAlgoAPI_BuilderAlgo,
    BRepAlgoAPI_Common,
    BRepAlgoAPI_Cut,
    BRepAlgoAPI_Fuse,
    BRepAlgoAPI_Section,
    BRepAlgoAPI_Splitter,
)
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepTools import BRepTools
from OCP.collections import List_TopoDS_Shape
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from build123d.kernel.base import BoolOp, Kernel, TopoKind

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

    def bounding_box(
        self, shape: TopoDS_Shape, optimal: bool = True
    ) -> tuple[float, float, float, float, float, float] | None:
        box = Bnd_Box()
        BRepTools.Clean_s(shape)  # a stale mesh would widen the box
        if optimal:
            BRepBndLib.AddOptimal_s(shape, box)
        else:
            BRepBndLib.Add_s(shape, box, True)
        if box.IsVoid():
            return None
        # the OCP stubs do not declare the per-axis getters
        return (
            box.GetXMin(),  # type: ignore[attr-defined]
            box.GetYMin(),  # type: ignore[attr-defined]
            box.GetZMin(),  # type: ignore[attr-defined]
            box.GetXMax(),  # type: ignore[attr-defined]
            box.GetYMax(),  # type: ignore[attr-defined]
            box.GetZMax(),  # type: ignore[attr-defined]
        )

    # ---- operations ---------------------------------------------------------

    def boolean(
        self,
        op: BoolOp,
        args: Sequence[TopoDS_Shape],
        tools: Sequence[TopoDS_Shape],
        *,
        glue: bool = False,
        fuzzy: float | None = None,
    ) -> tuple[TopoDS_Shape, Any]:
        operation: Any = _BOOLEAN_ALGORITHMS[op]()  # Set* live on the subclasses
        if glue:
            operation.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
        if fuzzy:
            operation.SetFuzzyValue(fuzzy)
        operation.SetArguments(_shape_list(args))
        operation.SetTools(_shape_list(tools))
        operation.SetRunParallel(True)
        operation.Build()
        return self.downcast(operation.Shape()), operation.History()

    def section(self, a: TopoDS_Shape, b: TopoDS_Shape) -> tuple[TopoDS_Shape, Any]:
        section = BRepAlgoAPI_Section(a, b)
        section.SetRunParallel(True)
        section.Approximation(True)
        section.ComputePCurveOn1(True)
        section.ComputePCurveOn2(True)
        section.Build()
        return section.Shape(), section.History()

    def unify_same_domain(self, shape: TopoDS_Shape) -> tuple[TopoDS_Shape, Any]:
        upgrader = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
        upgrader.AllowInternalEdges(False)
        upgrader.Build()
        return self.downcast(upgrader.Shape()), upgrader.History()

    # ---- placement ----------------------------------------------------------

    def identity_location(self) -> TopLoc_Location:
        return TopLoc_Location()

    def location_of(self, shape: TopoDS_Shape) -> TopLoc_Location:
        return shape.Location()

    def set_location(self, shape: TopoDS_Shape, location: TopLoc_Location) -> None:
        shape.Location(location)

    def located(self, shape: TopoDS_Shape, location: TopLoc_Location) -> TopoDS_Shape:
        return self.downcast(shape.Located(location))

    def moved(self, shape: TopoDS_Shape, location: TopLoc_Location) -> TopoDS_Shape:
        return self.downcast(shape.Moved(location))

    def relocation_between(
        self, source: TopoDS_Shape, target: TopoDS_Shape
    ) -> TopLoc_Location:
        return target.Location().Multiplied(source.Location().Inverted())


_BOOLEAN_ALGORITHMS: dict[BoolOp, Callable[[], BRepAlgoAPI_BuilderAlgo]] = {
    BoolOp.FUSE: BRepAlgoAPI_Fuse,
    BoolOp.CUT: BRepAlgoAPI_Cut,
    BoolOp.COMMON: BRepAlgoAPI_Common,
    BoolOp.SECTION: BRepAlgoAPI_Section,
    BoolOp.SPLIT: BRepAlgoAPI_Splitter,
}


def _shape_list(shapes: Sequence[TopoDS_Shape]) -> List_TopoDS_Shape:
    """A kernel list holding ``shapes`` in order."""
    result = List_TopoDS_Shape()
    for shape in shapes:
        result.Append(shape)
    return result


def kind_to_topabs(kind: TopoKind) -> ta.TopAbs_ShapeEnum:
    """The OCCT enum for ``kind`` (for modules that still drive OCP directly)."""
    return _KIND_TO_TOPABS[kind]
