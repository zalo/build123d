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

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import (
    BRepAdaptor_CompCurve,
    BRepAdaptor_Curve,
    BRepAdaptor_Surface,
)
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_RightCorner,
    BRepBuilderAPI_RoundCorner,
    BRepBuilderAPI_Transformed,
)
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp_Face
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepOffsetAPI import (
    BRepOffsetAPI_MakePipeShell,
    BRepOffsetAPI_MakeThickSolid,
    BRepOffsetAPI_ThruSections,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCone,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakePrism,
    BRepPrimAPI_MakeRevol,
    BRepPrimAPI_MakeSphere,
    BRepPrimAPI_MakeTorus,
    BRepPrimAPI_MakeWedge,
)
from OCP.GCPnts import (
    GCPnts_AbscissaPoint,
    GCPnts_QuasiUniformDeflection,
    GCPnts_UniformDeflection,
)
import OCP.GeomAbs as ga
from OCP.GeomAbs import GeomAbs_C2, GeomAbs_C3, GeomAbs_CN, GeomAbs_JoinType
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_ProjectPointOnSurf
from OCP.GeomLib import GeomLib_IsPlanarSurface
from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Vec

from build123d.kernel.base import (
    BoolOp,
    CurveView,
    JoinKind,
    Kernel,
    Point,
    SurfaceView,
    SweepTransition,
    TopoKind,
)

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


_PLANAR_TOLERANCE = 1e-6  # build123d's TOLERANCE; the facade must not import geometry

_CURVE_TYPE_LABELS = {
    ga.GeomAbs_Line: "LINE",
    ga.GeomAbs_Circle: "CIRCLE",
    ga.GeomAbs_Ellipse: "ELLIPSE",
    ga.GeomAbs_Hyperbola: "HYPERBOLA",
    ga.GeomAbs_Parabola: "PARABOLA",
    ga.GeomAbs_BezierCurve: "BEZIER",
    ga.GeomAbs_BSplineCurve: "BSPLINE",
    ga.GeomAbs_OffsetCurve: "OFFSET",
    ga.GeomAbs_OtherCurve: "OTHER",
}
_SURFACE_TYPE_LABELS = {
    ga.GeomAbs_Plane: "PLANE",
    ga.GeomAbs_Cylinder: "CYLINDER",
    ga.GeomAbs_Cone: "CONE",
    ga.GeomAbs_Sphere: "SPHERE",
    ga.GeomAbs_Torus: "TORUS",
    ga.GeomAbs_BezierSurface: "BEZIER",
    ga.GeomAbs_BSplineSurface: "BSPLINE",
    ga.GeomAbs_SurfaceOfRevolution: "REVOLUTION",
    ga.GeomAbs_SurfaceOfExtrusion: "EXTRUSION",
    ga.GeomAbs_OffsetSurface: "OFFSET",
    ga.GeomAbs_OtherSurface: "OTHER",
}
_JOIN = {
    JoinKind.ARC: GeomAbs_JoinType.GeomAbs_Arc,
    JoinKind.INTERSECTION: GeomAbs_JoinType.GeomAbs_Intersection,
}
_TRANSITION = {
    SweepTransition.TRANSFORMED: BRepBuilderAPI_Transformed,
    SweepTransition.ROUND: BRepBuilderAPI_RoundCorner,
    SweepTransition.RIGHT: BRepBuilderAPI_RightCorner,
}


def _pt(p) -> Point:
    return (p.X(), p.Y(), p.Z())


class OcctCurveView(CurveView):
    """``CurveView`` over a ``BRepAdaptor_Curve`` (edge) or ``BRepAdaptor_CompCurve`` (wire)."""

    def __init__(self, shape: TopoDS_Shape):
        if shape.ShapeType() == ta.TopAbs_WIRE:
            self._shape: TopoDS_Shape = TopoDS.Wire(shape)
            self.wrapped: BRepAdaptor_Curve | BRepAdaptor_CompCurve = (
                BRepAdaptor_CompCurve(self._shape)
            )
        else:
            self._shape = TopoDS.Edge(shape)
            self.wrapped = BRepAdaptor_Curve(self._shape)
        self.geom_type = _CURVE_TYPE_LABELS[self.wrapped.GetType()]
        self.first = self.wrapped.FirstParameter()
        self.last = self.wrapped.LastParameter()

    def is_periodic(self) -> bool:
        return self.wrapped.IsPeriodic()

    def period(self) -> float:
        return self.wrapped.Period()

    def value(self, t: float) -> Point:
        return _pt(self.wrapped.Value(t))

    def d1(self, t: float) -> tuple[Point, Point]:
        point, tangent = gp_Pnt(), gp_Vec()
        self.wrapped.D1(t, point, tangent)
        return _pt(point), _pt(tangent)

    def length(self, t0: float | None = None, t1: float | None = None) -> float:
        if t0 is None and t1 is None:
            return GCPnts_AbscissaPoint.Length_s(self.wrapped)
        return GCPnts_AbscissaPoint.Length_s(
            self.wrapped,
            self.first if t0 is None else t0,
            self.last if t1 is None else t1,
        )

    def param_at_length(self, length: float, t0: float | None = None) -> float:
        return GCPnts_AbscissaPoint(
            self.wrapped, length, self.first if t0 is None else t0
        ).Parameter()

    def closest_param(self, point: Point) -> float:
        curve = BRep_Tool.Curve_s(TopoDS.Edge(self._shape), float(), float())
        projector = GeomAPI_ProjectPointOnCurve(gp_Pnt(*point), curve)
        return projector.LowerDistanceParameter()

    def positions_by_deflection(self, deflection: float) -> list[Point]:
        if self.wrapped.Continuity() in (GeomAbs_C2, GeomAbs_C3, GeomAbs_CN):
            discretizer: GCPnts_UniformDeflection | GCPnts_QuasiUniformDeflection = (
                GCPnts_UniformDeflection()
            )
        else:
            discretizer = GCPnts_QuasiUniformDeflection()
        discretizer.Initialize(self.wrapped, deflection, self.first, self.last)
        if not discretizer.IsDone() or discretizer.NbPoints() == 0:
            raise RuntimeError("Deflection calculation failed")
        return [
            _pt(self.wrapped.Value(discretizer.Parameter(i + 1)))
            for i in range(discretizer.NbPoints())
        ]

    def circle(self) -> tuple[Point, Point, float]:
        circle = self.wrapped.Circle()
        return (
            _pt(circle.Position().Location()),
            _pt(circle.Axis().Direction()),
            circle.Radius(),
        )

    def ellipse(self) -> tuple[Point, Point, float, float]:
        ellipse = self.wrapped.Ellipse()
        return (
            _pt(ellipse.Position().Location()),
            _pt(ellipse.Axis().Direction()),
            ellipse.MajorRadius(),
            ellipse.MinorRadius(),
        )


class OcctSurfaceView(SurfaceView):
    """``SurfaceView`` over a face: ``BRepAdaptor_Surface`` for evaluation,
    ``BRepGProp_Face`` for oriented normals, ``Geom_Surface`` for projection."""

    def __init__(self, face: TopoDS_Shape):
        self._face = TopoDS.Face(face)
        self.adaptor = BRepAdaptor_Surface(self._face)
        self.geom_type = _SURFACE_TYPE_LABELS[self.adaptor.GetType()]

    def uv_bounds(self) -> tuple[float, float, float, float]:
        return BRepTools.UVBounds_s(self._face)

    def is_u_periodic(self) -> bool:
        return self.adaptor.IsUPeriodic()

    def is_v_periodic(self) -> bool:
        return self.adaptor.IsVPeriodic()

    def u_period(self) -> float:
        return self.adaptor.UPeriod()

    def v_period(self) -> float:
        return self.adaptor.VPeriod()

    def value(self, u: float, v: float) -> Point:
        return _pt(self.adaptor.Value(u, v))

    def d1(self, u: float, v: float) -> tuple[Point, Point, Point]:
        point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        BRep_Tool.Surface_s(self._face).D1(u, v, point, du, dv)
        return _pt(point), _pt(du), _pt(dv)

    def dn(self, u: float, v: float, u_order: int, v_order: int) -> Point:
        return _pt(self.adaptor.DN(u, v, u_order, v_order))

    def normal(self, u: float, v: float) -> tuple[Point, Point]:
        point, normal = gp_Pnt(), gp_Vec()
        BRepGProp_Face(self._face).Normal(u, v, point, normal)
        return _pt(point), _pt(normal)

    def closest_uv(self, point: Point) -> tuple[float, float]:
        projector = GeomAPI_ProjectPointOnSurf(
            gp_Pnt(*point), BRep_Tool.Surface_s(self._face)
        )
        return projector.LowerDistanceParameters()

    def plane(self) -> tuple[Point, Point, Point] | None:
        searcher = GeomLib_IsPlanarSurface(
            BRep_Tool.Surface_s(self._face), _PLANAR_TOLERANCE
        )
        if not searcher.IsPlanar():
            return None
        pln = searcher.Plan()
        if not pln.Position().Direct():  # a left-handed frame was returned
            pln = gp_Pln(gp_Ax3(pln.Position().Ax2()))
        return (
            _pt(pln.Location()),
            _pt(pln.Axis().Direction()),
            _pt(pln.XAxis().Direction()),
        )

    def radius(self) -> float:
        if self.geom_type == "CYLINDER":
            return self.adaptor.Cylinder().Radius()
        if self.geom_type == "SPHERE":
            return self.adaptor.Sphere().Radius()
        raise ValueError(f"a {self.geom_type} surface has no radius")

    def semi_angle(self) -> float:
        return self.adaptor.Cone().SemiAngle()

    def torus_radii(self) -> tuple[float, float]:
        torus = self.adaptor.Torus()
        return torus.MajorRadius(), torus.MinorRadius()


class _OcctGeometry:
    """The KernelGeometry half of OcctKernel; OcctKernel inherits it."""

    downcast: Callable[[TopoDS_Shape], TopoDS_Shape]  # provided by OcctKernel

    # ---- views --------------------------------------------------------------

    def curve_of(self, shape: TopoDS_Shape) -> OcctCurveView:
        return OcctCurveView(shape)

    def surface_of(self, face: TopoDS_Shape) -> OcctSurfaceView:
        return OcctSurfaceView(face)

    # ---- construction -------------------------------------------------------

    def make_box(
        self, frame: gp_Ax2, length: float, width: float, height: float
    ) -> TopoDS_Shape:
        return TopoDS.Solid(BRepPrimAPI_MakeBox(frame, length, width, height).Shape())

    def make_cylinder(
        self, frame: gp_Ax2, radius: float, height: float, angle: float
    ) -> TopoDS_Shape:
        return TopoDS.Solid(
            BRepPrimAPI_MakeCylinder(frame, radius, height, angle).Shape()
        )

    def make_cone(
        self,
        frame: gp_Ax2,
        base_radius: float,
        top_radius: float,
        height: float,
        angle: float,
    ) -> TopoDS_Shape:
        return TopoDS.Solid(
            BRepPrimAPI_MakeCone(frame, base_radius, top_radius, height, angle).Shape()
        )

    def make_sphere(
        self, frame: gp_Ax2, radius: float, angle1: float, angle2: float, angle3: float
    ) -> TopoDS_Shape:
        return TopoDS.Solid(
            BRepPrimAPI_MakeSphere(frame, radius, angle1, angle2, angle3).Shape()
        )

    def make_torus(
        self,
        frame: gp_Ax2,
        major_radius: float,
        minor_radius: float,
        angle1: float,
        angle2: float,
        angle3: float,
    ) -> TopoDS_Shape:
        return TopoDS.Solid(
            BRepPrimAPI_MakeTorus(
                frame, major_radius, minor_radius, angle1, angle2, angle3
            ).Shape()
        )

    def make_wedge(
        self,
        frame: gp_Ax2,
        delta_x: float,
        delta_y: float,
        delta_z: float,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
    ) -> TopoDS_Shape:
        return TopoDS.Solid(
            BRepPrimAPI_MakeWedge(
                frame, delta_x, delta_y, delta_z, min_x, min_z, max_x, max_z
            ).Solid()
        )

    def extrude(self, shape: TopoDS_Shape, direction: Point) -> TopoDS_Shape:
        return self.downcast(BRepPrimAPI_MakePrism(shape, gp_Vec(*direction)).Shape())

    def revolve(self, shape: TopoDS_Shape, axis: Any, angle: float) -> TopoDS_Shape:
        return self.downcast(BRepPrimAPI_MakeRevol(shape, axis, angle, True).Shape())

    def loft(
        self, sections: Sequence[TopoDS_Shape], solid: bool, ruled: bool
    ) -> TopoDS_Shape:
        builder = BRepOffsetAPI_ThruSections(solid, ruled)
        for section in sections:
            if section.ShapeType() == ta.TopAbs_VERTEX:
                builder.AddVertex(TopoDS.Vertex(section))
            elif section.ShapeType() == ta.TopAbs_WIRE:
                builder.AddWire(TopoDS.Wire(section))
        builder.Build()
        return builder.Shape()

    def sweep(
        self,
        profile: TopoDS_Shape,
        path: TopoDS_Shape,
        *,
        is_frenet: bool,
        transition: SweepTransition,
        make_solid: bool,
        binormal: Point | None = None,
        binormal_origin: Point | None = None,
        auxiliary_spine: TopoDS_Shape | None = None,
    ) -> TopoDS_Shape:
        builder = BRepOffsetAPI_MakePipeShell(TopoDS.Wire(path))
        rotate = False
        if binormal is not None:
            frame = gp_Ax2()
            frame.SetLocation(gp_Pnt(*(binormal_origin or (0.0, 0.0, 0.0))))
            frame.SetDirection(gp_Dir(*binormal))
            builder.SetMode(frame)
            rotate = True
        elif auxiliary_spine is not None:
            builder.SetMode(TopoDS.Wire(auxiliary_spine), True)
        else:
            builder.SetMode(is_frenet)
        builder.SetTransitionMode(_TRANSITION[transition])
        builder.Add(profile, False, rotate)
        builder.Build()
        if make_solid:
            builder.MakeSolid()
        return builder.Shape()

    def fillet(
        self, shape: TopoDS_Shape, edges: Sequence[TopoDS_Shape], radius: float
    ) -> tuple[TopoDS_Shape, Any]:
        builder = BRepFilletAPI_MakeFillet(shape)
        for edge in edges:
            builder.Add(radius, TopoDS.Edge(edge))
        return builder.Shape(), builder

    def chamfer(
        self,
        shape: TopoDS_Shape,
        specs: Sequence[tuple[TopoDS_Shape, TopoDS_Shape, float, float]],
    ) -> tuple[TopoDS_Shape, Any]:
        builder = BRepFilletAPI_MakeChamfer(shape)
        for edge, face, distance1, distance2 in specs:
            builder.Add(distance1, distance2, TopoDS.Edge(edge), TopoDS.Face(face))
        return builder.Shape(), builder

    def thick_solid(
        self,
        shape: TopoDS_Shape,
        openings: Sequence[TopoDS_Shape],
        thickness: float,
        tolerance: float,
        join: JoinKind,
        remove_internal_edges: bool = False,
    ) -> TopoDS_Shape:
        builder = BRepOffsetAPI_MakeThickSolid()
        options: dict[str, Any] = {"Intersection": True, "Join": _JOIN[join]}
        if remove_internal_edges:
            options["RemoveIntEdges"] = True
        builder.MakeThickSolidByJoin(
            shape, _shape_list(openings), thickness, tolerance, **options
        )
        builder.Build()
        return builder.Shape()

    def make_solid(self, shells: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
        builder = BRepBuilderAPI_MakeSolid(TopoDS.Shell(shells[0]))
        for shell in shells[1:]:
            builder.Add(TopoDS.Shell(shell))
        return builder.Shape()

    # ---- tessellation -------------------------------------------------------

    def mesh(
        self, shape: TopoDS_Shape, tolerance: float, angular_tolerance: float
    ) -> None:
        if not BRepTools.Triangulation_s(shape, tolerance):
            BRepMesh_IncrementalMesh(shape, tolerance, True, angular_tolerance, True)

    def face_triangulation(
        self, face: TopoDS_Shape
    ) -> tuple[list[Point], list[tuple[int, int, int]]] | None:
        location = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(TopoDS.Face(face), location)
        if poly is None:
            return None
        trsf = location.Transformation()
        reverse = face.Orientation() == ta.TopAbs_REVERSED
        points = [
            _pt(poly.Node(i).Transformed(trsf)) for i in range(1, poly.NbNodes() + 1)
        ]
        triangles = [
            (
                (t.Value(1) - 1, t.Value(3) - 1, t.Value(2) - 1)
                if reverse
                else (t.Value(1) - 1, t.Value(2) - 1, t.Value(3) - 1)
            )
            for t in poly.Triangles()
        ]
        return points, triangles


class OcctKernel(_OcctGeometry, Kernel):
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
