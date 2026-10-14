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
from math import cos, pi, sqrt
from typing import Any
from typing import cast as tcast

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
from OCP.BRepTools import BRepTools, BRepTools_History
from OCP.collections import List_TopoDS_Shape
from OCP.GProp import GProp_GProps
from OCP.ShapeFix import ShapeFix_Shape
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape

from OCP.BRep import BRep_Tool
from OCP.BRep import BRep_Builder
from OCP.BRepAdaptor import (
    BRepAdaptor_CompCurve,
    BRepAdaptor_Curve,
    BRepAdaptor_Surface,
)
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_DisconnectedWire,
    BRepBuilderAPI_EmptyWire,
    BRepBuilderAPI_GTransform,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_NonManifoldWire,
    BRepBuilderAPI_RightCorner,
    BRepBuilderAPI_RoundCorner,
    BRepBuilderAPI_Sewing,
    BRepBuilderAPI_Transform,
    BRepBuilderAPI_Transformed,
)
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepLib import BRepLib_FindSurface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet2d
from OCP.BRepOffset import BRepOffset_MakeOffset
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
from OCP.BRepProj import BRepProj_Projection
from OCP.GeomFill import (
    GeomFill_CorrectedFrenet,
    GeomFill_Frenet,
    GeomFill_TrihedronLaw,
)
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.ShapeFix import ShapeFix_Wireframe
from OCP.collections import (
    HSequence_TopoDS_Shape,
    HArray2_double,
    HArray2_gp_Pnt,
    Sequence_TopoDS_Shape,
)
from OCP.BRep import BRep_TEdge
from OCP.BRepFeat import BRepFeat_MakeDPrism, BRepFeat_SplitShape
from OCP.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepOffset import BRepOffset_Skin
from OCP.BRepOffsetAPI import BRepOffsetAPI_DraftAngle, BRepOffsetAPI_MakeFilling
from OCP.BRepTools import BRepTools_ReShape
from OCP.BRepFill import BRepFill
from OCP.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
from OCP.Geom import (
    Geom_BoundedSurface,
    Geom_BezierSurface,
    Geom_OffsetSurface,
    Geom_RectangularTrimmedSurface,
)
from OCP.GeomAdaptor import GeomAdaptor_Surface
from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCP.Precision import Precision
from OCP.ShapeAnalysis import (
    ShapeAnalysis_Curve,
    ShapeAnalysis_Edge,
    ShapeAnalysis_Shell,
    ShapeAnalysis_Surface,
)
from OCP.ShapeCustom import ShapeCustom, ShapeCustom_RestrictionParameters
from OCP.ShapeFix import ShapeFix_Solid
from OCP.gp import gp_Lin, gp_XYZ
from OCP.IntTools import IntTools_FClass2d
from OCP.ShapeFix import ShapeFix_Face
from OCP.Standard import (
    Standard_ConstructionError,
    Standard_DomainError,
    Standard_Failure,
    Standard_NoSuchObject,
)
from OCP.StdFail import StdFail_NotDone
from OCP.TopExp import TopExp
from OCP.TopoDS import (
    TopoDS_Builder,
    TopoDS_Compound,
    TopoDS_Face,
    TopoDS_Iterator,
    TopoDS_Shell,
)
from OCP.collections import (
    IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher,
)
from OCP.gp import (
    gp_Circ,
    gp_Dir2d,
    gp_Elips,
    gp_GTrsf,
    gp_Hypr,
    gp_Parab,
    gp_Pnt2d,
    gp_Trsf,
    gp_Vec2d,
)
from OCP.BRepAdaptor import BRepAdaptor_Curve2d
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepExtrema import BRepExtrema_SupportType
from OCP.BRepLib import BRepLib
from OCP.BRepLProp import BRepLProp
from OCP.BRepTools import BRepTools_WireExplorer
from OCP.Extrema import Extrema_ExtPC
from OCP.GC import (
    GC_MakeArcOfCircle,
    GC_MakeArcOfEllipse,
    GC_MakeArcOfHyperbola,
    GC_MakeArcOfParabola,
)
from OCP.Geom import (
    Geom_BezierCurve,
    Geom_BSplineCurve,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Line,
    Geom_Plane,
    Geom_Surface,
    Geom_TrimmedCurve,
)
from OCP.Geom2d import Geom2d_Line, Geom2d_TrimmedCurve
from OCP.Geom2dAPI import Geom2dAPI_InterCurveCurve
from OCP.GeomAdaptor import GeomAdaptor_Curve
from OCP.GeomAPI import GeomAPI_Interpolate, GeomAPI_PointsToBSpline
from OCP.GeomConvert import (
    GeomConvert_ApproxCurve,
    GeomConvert_CompCurveToBSplineCurve,
)
from OCP.collections import (
    Array1_double,
    Array1_gp_Pnt,
    Array1_gp_Vec,
    Array1_int,
    HArray1_bool,
    HArray1_double,
    HArray1_gp_Pnt,
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
from OCP.gp import gp_Ax1, gp_Ax2, gp_Ax3, gp_Dir, gp_Pln, gp_Pnt, gp_Vec

from build123d.kernel.base import (
    AncestorMap,
    BoolOp,
    CurveView,
    DraftFailure,
    JoinKind,
    Kernel,
    Point,
    SurfaceFilling,
    SurfaceView,
    SweepTransition,
    TopoKind,
    WireStatus,
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
_CONTINUITY_FROM_LABEL = {
    "C0": ga.GeomAbs_C0,
    "G1": ga.GeomAbs_G1,
    "C1": ga.GeomAbs_C1,
    "G2": ga.GeomAbs_G2,
    "C2": ga.GeomAbs_C2,
}
_CONTINUITY_LABELS = {
    ga.GeomAbs_C0: "C0",
    ga.GeomAbs_G1: "G1",
    ga.GeomAbs_C1: "C1",
    ga.GeomAbs_G2: "G2",
    ga.GeomAbs_C2: "C2",
    ga.GeomAbs_C3: "C3",
    ga.GeomAbs_CN: "CN",
}
_JOIN = {
    JoinKind.ARC: GeomAbs_JoinType.GeomAbs_Arc,
    JoinKind.INTERSECTION: GeomAbs_JoinType.GeomAbs_Intersection,
    JoinKind.TANGENT: GeomAbs_JoinType.GeomAbs_Tangent,
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
        self.shape = self._shape
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
        if isinstance(self.wrapped, BRepAdaptor_CompCurve):
            extrema = Extrema_ExtPC(gp_Pnt(*point), self.wrapped)
            if not extrema.IsDone() or extrema.NbExt() == 0:
                raise RuntimeError("Failed to find point on curve")
            nearest = min(
                range(1, extrema.NbExt() + 1), key=lambda i: extrema.SquareDistance(i)
            )
            return extrema.Point(nearest).Parameter()
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

    def hyperbola(self) -> tuple[Point, Point, float, float]:
        hyperbola = self.wrapped.Hyperbola()
        return (
            _pt(hyperbola.Location()),
            _pt(hyperbola.Axis().Direction()),
            hyperbola.MajorRadius(),
            hyperbola.MinorRadius(),
        )

    def parabola(self) -> tuple[Point, Point, float]:
        parabola = self.wrapped.Parabola()
        return (
            _pt(parabola.Location()),
            _pt(parabola.Axis().Direction()),
            parabola.Focal(),
        )

    def bezier(self) -> tuple[int, list[Point], list[float] | None]:
        bezier = self.wrapped.Bezier()
        count = bezier.NbPoles()
        poles = [_pt(bezier.Pole(i)) for i in range(1, count + 1)]
        weights = (
            [bezier.Weight(i) for i in range(1, count + 1)]
            if bezier.IsRational()
            else None
        )
        return bezier.Degree(), poles, weights

    def bspline(
        self,
    ) -> tuple[int, bool, list[Point], list[float] | None, list[float], list[int]]:
        spline = self.wrapped.BSpline()
        pole_count, knot_count = spline.NbPoles(), spline.NbKnots()
        return (
            spline.Degree(),
            spline.IsPeriodic(),
            [_pt(spline.Pole(i)) for i in range(1, pole_count + 1)],
            (
                [spline.Weight(i) for i in range(1, pole_count + 1)]
                if spline.IsRational()
                else None
            ),
            [spline.Knot(i) for i in range(1, knot_count + 1)],
            [spline.Multiplicity(i) for i in range(1, knot_count + 1)],
        )

    def offset(self) -> tuple[float, Point, TopoDS_Shape]:
        offset_curve = self.wrapped.OffsetCurve()
        return (
            offset_curve.Offset(),
            _pt(offset_curve.Direction()),
            BRepBuilderAPI_MakeEdge(offset_curve.BasisCurve()).Edge(),
        )

    def dn(self, t: float, order: int) -> Point:
        return _pt(self.wrapped.DN(t, order))

    def is_closed(self) -> bool:
        return self.wrapped.IsClosed()


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

    def curvatures(
        self, u: float, v: float, tolerance: float
    ) -> tuple[float, float, Point] | None:
        props = BRepLProp_SLProps(self.adaptor, u, v, 2, tolerance)
        if not props.IsCurvatureDefined():
            return None
        return props.MinCurvature(), props.MaxCurvature(), _pt(props.Normal())

    def uv_of_point(self, point: Point, tolerance: float) -> tuple[float, float, float]:
        location = TopLoc_Location()
        surface = BRep_Tool.Surface_s(self._face, location)
        local = gp_Pnt(*point).Transformed(location.Transformation().Inverted())
        analysis = ShapeAnalysis_Surface(surface)
        uv = analysis.ValueOfUV(local, tolerance)
        return uv.X(), uv.Y(), analysis.Gap()

    def iso_lengths(
        self, u_min: float, u_max: float, v_min: float, v_max: float, tolerance: float
    ) -> tuple[float, float]:
        surface = BRep_Tool.Surface_s(self._face)
        v_iso = GeomAdaptor_Curve(surface.VIso((v_min + v_max) * 0.5))
        u_iso = GeomAdaptor_Curve(surface.UIso((u_min + u_max) * 0.5))
        return (
            GCPnts_AbscissaPoint.Length_s(v_iso, u_min, u_max, tolerance),
            GCPnts_AbscissaPoint.Length_s(u_iso, v_min, v_max, tolerance),
        )


class OcctSurfaceFilling(SurfaceFilling):
    """``SurfaceFilling`` over ``BRepOffsetAPI_MakeFilling``."""

    def __init__(self, **options: Any):
        self.wrapped = BRepOffsetAPI_MakeFilling(**options)

    def add_edge(
        self, edge: TopoDS_Shape, continuity: str, face: TopoDS_Shape | None = None
    ) -> None:
        order = _CONTINUITY_FROM_LABEL[continuity]
        if face is None:
            self.wrapped.Add(TopoDS.Edge(edge), order)
        else:
            self.wrapped.Add(TopoDS.Edge(edge), TopoDS.Face(face), order)

    def add_point(self, point: Point) -> None:
        self.wrapped.Add(gp_Pnt(*point))

    def build(self) -> TopoDS_Shape:
        self.wrapped.Build()
        return self.wrapped.Shape()


class _OcctGeometry:
    """The KernelGeometry half of OcctKernel; OcctKernel inherits it."""

    # provided by OcctKernel
    downcast: Callable[[TopoDS_Shape], TopoDS_Shape]
    ancestors: Callable[[TopoDS_Shape, TopoKind, TopoKind], AncestorMap]
    sub_shapes: Callable[[TopoDS_Shape, TopoKind], list[TopoDS_Shape]]

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
        curvilinear_equivalence: bool = True,
    ) -> TopoDS_Shape:
        builder = BRepOffsetAPI_MakePipeShell(TopoDS.Wire(path))
        rotate = self._set_sweep_mode(
            builder,
            is_frenet,
            binormal,
            binormal_origin,
            auxiliary_spine,
            curvilinear_equivalence,
        )
        builder.SetTransitionMode(_TRANSITION[transition])
        builder.Add(profile, False, rotate)
        builder.Build()
        if make_solid:
            builder.MakeSolid()
        return builder.Shape()

    @staticmethod
    def _set_sweep_mode(
        builder: BRepOffsetAPI_MakePipeShell,
        is_frenet: bool,
        binormal: Point | None,
        binormal_origin: Point | None,
        auxiliary_spine: TopoDS_Shape | None,
        curvilinear_equivalence: bool = True,
    ) -> bool:
        """Set how the profile turns along the path; True when it must rotate."""
        if binormal is not None:
            frame = gp_Ax2()
            frame.SetLocation(gp_Pnt(*(binormal_origin or (0.0, 0.0, 0.0))))
            frame.SetDirection(gp_Dir(*binormal))
            builder.SetMode(frame)
            return True
        if auxiliary_spine is not None:
            builder.SetMode(TopoDS.Wire(auxiliary_spine), curvilinear_equivalence)
        else:
            builder.SetMode(is_frenet)
        return False

    def sweep_multi(
        self,
        profiles: Sequence[TopoDS_Shape],
        path: TopoDS_Shape,
        *,
        is_frenet: bool,
        make_solid: bool,
        binormal: Point | None = None,
        binormal_origin: Point | None = None,
        auxiliary_spine: TopoDS_Shape | None = None,
    ) -> TopoDS_Shape:
        builder = BRepOffsetAPI_MakePipeShell(TopoDS.Wire(path))
        rotate = self._set_sweep_mode(
            builder, is_frenet, binormal, binormal_origin, auxiliary_spine
        )
        for profile in profiles:
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

    def make_vertex(self, point: Point) -> TopoDS_Shape:
        return BRepBuilderAPI_MakeVertex(gp_Pnt(*point)).Vertex()

    # ---- edges from analytic curves -----------------------------------------

    def make_line(self, start: Point, end: Point) -> TopoDS_Shape:
        return BRepBuilderAPI_MakeEdge(gp_Pnt(*start), gp_Pnt(*end)).Edge()

    def make_infinite_line(self, origin: Point, direction: Point) -> TopoDS_Shape:
        return BRepBuilderAPI_MakeEdge(
            Geom_Line(gp_Pnt(*origin), gp_Dir(*direction))
        ).Edge()

    def make_circle(self, frame: gp_Ax2, radius: float) -> TopoDS_Shape:
        return BRepBuilderAPI_MakeEdge(gp_Circ(frame, radius)).Edge()

    def make_circle_arc(
        self, frame: gp_Ax2, radius: float, start: float, end: float, sense: bool
    ) -> TopoDS_Shape:
        arc = GC_MakeArcOfCircle(gp_Circ(frame, radius), start, end, sense).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    @staticmethod
    def _ellipse(
        frame: gp_Ax2, major: float, minor: float, rotation: float
    ) -> gp_Elips:
        ellipse = gp_Elips(frame, major, minor)
        if rotation:
            ellipse = ellipse.Rotated(frame.Axis(), rotation)
        return ellipse

    def make_ellipse(
        self, frame: gp_Ax2, major: float, minor: float, rotation: float = 0.0
    ) -> TopoDS_Shape:
        return BRepBuilderAPI_MakeEdge(
            self._ellipse(frame, major, minor, rotation)
        ).Edge()

    def make_ellipse_arc(
        self,
        frame: gp_Ax2,
        major: float,
        minor: float,
        start: float,
        end: float,
        sense: bool,
        rotation: float = 0.0,
    ) -> TopoDS_Shape:
        arc = GC_MakeArcOfEllipse(
            self._ellipse(frame, major, minor, rotation), start, end, sense
        ).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    def make_parabola_arc(
        self, frame: gp_Ax2, focal_length: float, start: float, end: float, sense: bool
    ) -> TopoDS_Shape:
        arc = GC_MakeArcOfParabola(
            gp_Parab(frame, focal_length), start, end, sense
        ).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    @staticmethod
    def _hyperbola(
        frame: gp_Ax2, major: float, minor: float, rotation: float
    ) -> gp_Hypr:
        hyperbola = gp_Hypr(frame, major, minor)
        if rotation:
            hyperbola = hyperbola.Rotated(frame.Axis(), rotation)
        return hyperbola

    def make_hyperbola(
        self, frame: gp_Ax2, major: float, minor: float, rotation: float = 0.0
    ) -> TopoDS_Shape:
        return BRepBuilderAPI_MakeEdge(
            self._hyperbola(frame, major, minor, rotation)
        ).Edge()

    def make_hyperbola_arc(
        self,
        frame: gp_Ax2,
        major: float,
        minor: float,
        start: float,
        end: float,
        sense: bool,
        rotation: float = 0.0,
    ) -> TopoDS_Shape:
        arc = GC_MakeArcOfHyperbola(
            self._hyperbola(frame, major, minor, rotation), start, end, sense
        ).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    def make_tangent_arc(
        self, start: Point, tangent: Point, end: Point
    ) -> TopoDS_Shape:
        arc = GC_MakeArcOfCircle(gp_Pnt(*start), gp_Vec(*tangent), gp_Pnt(*end)).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    def make_three_point_arc(self, p1: Point, p2: Point, p3: Point) -> TopoDS_Shape:
        arc = GC_MakeArcOfCircle(gp_Pnt(*p1), gp_Pnt(*p2), gp_Pnt(*p3)).Value()
        return BRepBuilderAPI_MakeEdge(arc).Edge()

    def make_bezier(
        self, poles: Sequence[Point], weights: Sequence[float] | None = None
    ) -> TopoDS_Shape:
        if weights:
            curve = Geom_BezierCurve(_point_array(poles), _double_array(weights))
        else:
            curve = Geom_BezierCurve(_point_array(poles))
        return BRepBuilderAPI_MakeEdge(curve).Edge()

    def make_bspline(
        self,
        poles: Sequence[Point],
        knots: Sequence[float],
        multiplicities: Sequence[int],
        degree: int,
        periodic: bool,
        weights: Sequence[float] | None = None,
    ) -> TopoDS_Shape:
        multiplicity_array = Array1_int(1, len(multiplicities))
        for index, multiplicity in enumerate(multiplicities, start=1):
            multiplicity_array.SetValue(index, multiplicity)
        if weights:
            curve = Geom_BSplineCurve(
                _point_array(poles),
                _double_array(weights),
                _double_array(knots),
                multiplicity_array,
                degree,
                periodic,
            )
        else:
            curve = Geom_BSplineCurve(
                _point_array(poles),
                _double_array(knots),
                multiplicity_array,
                degree,
                periodic,
            )
        return BRepBuilderAPI_MakeEdge(curve).Edge()

    def make_spline_interpolated(
        self,
        points: Sequence[Point],
        *,
        periodic: bool,
        tolerance: float,
        parameters: Sequence[float] | None = None,
        end_tangents: tuple[Point, Point] | None = None,
        point_tangents: Sequence[Point | None] | None = None,
        scale: bool = True,
    ) -> TopoDS_Shape:
        point_array = HArray1_gp_Pnt(1, len(points))
        for index, point in enumerate(points, start=1):
            point_array.SetValue(index, gp_Pnt(*point))
        if parameters is None:
            builder = GeomAPI_Interpolate(point_array, periodic, tolerance)
        else:
            parameter_array = HArray1_double(1, len(parameters))
            for index, value in enumerate(parameters, start=1):
                parameter_array.SetValue(index, value)
            builder = GeomAPI_Interpolate(
                point_array, parameter_array, periodic, tolerance
            )
        if end_tangents is not None:
            builder.Load(gp_Vec(*end_tangents[0]), gp_Vec(*end_tangents[1]), scale)
        elif point_tangents is not None:
            tangent_array = Array1_gp_Vec(1, len(point_tangents))
            enabled_array = HArray1_bool(1, len(point_tangents))
            for index, tangent in enumerate(point_tangents, start=1):
                enabled_array.SetValue(index, tangent is not None)
                tangent_array.SetValue(index, gp_Vec(*(tangent or (0.0, 0.0, 0.0))))
            builder.Load(tangent_array, enabled_array, scale)
        builder.Perform()
        if not builder.IsDone():
            raise ValueError("B-spline interpolation failed")
        return BRepBuilderAPI_MakeEdge(builder.Curve()).Edge()

    def make_spline_fit(
        self,
        points: Sequence[Point],
        tolerance: float,
        min_degree: int,
        max_degree: int,
        smoothing: tuple[float, float, float] | None = None,
    ) -> TopoDS_Shape:
        point_array = _point_array(points)
        if smoothing:
            builder = GeomAPI_PointsToBSpline(
                point_array, *smoothing, DegMax=max_degree, Tol3D=tolerance
            )
        else:
            builder = GeomAPI_PointsToBSpline(
                point_array, DegMin=min_degree, DegMax=max_degree, Tol3D=tolerance
            )
        if not builder.IsDone():
            raise ValueError("B-spline approximation failed")
        return BRepBuilderAPI_MakeEdge(builder.Curve()).Edge()

    def make_helix(
        self,
        center: Point,
        normal: Point,
        radius: float,
        pitch: float,
        height: float,
        cone_angle: float,
        lefthand: bool,
    ) -> TopoDS_Shape:
        frame = gp_Ax3(gp_Pnt(*center), gp_Dir(*normal))
        surface: Geom_Surface
        if cone_angle == 0.0:
            surface = Geom_CylindricalSurface(frame, radius)
        else:
            surface = Geom_ConicalSurface(frame, cone_angle, radius)

        # the helix is a straight line in the surface's (u, v) parameters
        line_x = (-1 if lefthand else 1) * 2 * pi
        line_norm = sqrt(line_x * line_x + pitch * pitch)
        line_x, line_y = line_x / line_norm, pitch / line_norm
        line_length = (height / line_y) / cos(cone_angle)
        line = Geom2d_Line(gp_Pnt2d(0, 0), gp_Dir2d(line_x, line_y))
        segment = Geom2d_TrimmedCurve(line, 0, line_length, theAdjustPeriodic=True)

        edge = BRepBuilderAPI_MakeEdge(segment, surface).Edge()
        BRepLib.BuildCurves3d_s(edge, 1e-9, MaxSegment=2000)
        return edge

    # ---- edges from edges -----------------------------------------------------

    def edge_range(self, edge: TopoDS_Shape) -> tuple[float, float]:
        return BRep_Tool.Range_s(TopoDS.Edge(edge))

    def vertex_param_on_edge(self, vertex: TopoDS_Shape, edge: TopoDS_Shape) -> float:
        return BRep_Tool.Parameter_s(TopoDS.Vertex(vertex), TopoDS.Edge(edge))

    def reversed_geometry(self, edge: TopoDS_Shape) -> TopoDS_Shape:
        topods_edge = TopoDS.Edge(edge)
        first, last = BRep_Tool.Range_s(topods_edge)
        curve = BRep_Tool.Curve_s(topods_edge, first, last)
        return BRepBuilderAPI_MakeEdge(
            curve.Reversed(),
            curve.ReversedParameter(last),
            curve.ReversedParameter(first),
        ).Edge()

    def trim_edge(self, edge: TopoDS_Shape, start: float, end: float) -> TopoDS_Shape:
        curve = BRep_Tool.Curve_s(TopoDS.Edge(edge), float(), float())
        return BRepBuilderAPI_MakeEdge(curve, start, end).Edge()

    def trim_edge_by_length(
        self, edge: TopoDS_Shape, start: float, length: float
    ) -> TopoDS_Shape:
        curve = BRep_Tool.Curve_s(TopoDS.Edge(edge), float(), float())
        end = GCPnts_AbscissaPoint(GeomAdaptor_Curve(curve), length, start).Parameter()
        return BRepBuilderAPI_MakeEdge(Geom_TrimmedCurve(curve, start, end)).Edge()

    def approximate_edge(
        self, edge: TopoDS_Shape, tolerance: float, max_segments: int, max_degree: int
    ) -> TopoDS_Shape:
        adaptor = BRepAdaptor_Curve(TopoDS.Edge(edge))
        approximation = GeomConvert_ApproxCurve(
            Geom_TrimmedCurve(
                adaptor.Curve().Curve(),
                adaptor.FirstParameter(),
                adaptor.LastParameter(),
            ),
            tolerance,
            GeomAbs_C2,
            max_segments,
            max_degree,
        )
        if not approximation.HasResult():  # pragma: no cover
            raise RuntimeError("the curve could not be converted to a spline")
        return BRepBuilderAPI_MakeEdge(approximation.Curve()).Edge()

    def wire_as_bspline_edge(
        self, wire: TopoDS_Shape, tolerance: float
    ) -> TopoDS_Shape:
        builder = GeomConvert_CompCurveToBSplineCurve()
        for edge in self.wire_edges(wire):
            topods_edge = TopoDS.Edge(edge)
            if BRep_Tool.Degenerated_s(topods_edge):
                continue
            first, last = BRep_Tool.Range_s(topods_edge)
            curve = BRep_Tool.Curve_s(topods_edge, float(), float())
            if not builder.Add(Geom_TrimmedCurve(curve, first, last), tolerance):
                raise RuntimeError("Failed to build bspline.")
        edge_builder = BRepBuilderAPI_MakeEdge(builder.BSplineCurve())
        if not edge_builder.IsDone():
            raise RuntimeError("Failed to build bspline.")
        return edge_builder.Edge()

    def wire_edges(self, wire: TopoDS_Shape) -> list[TopoDS_Shape]:
        edges: list[TopoDS_Shape] = []
        explorer = BRepTools_WireExplorer(TopoDS.Wire(wire))
        while explorer.More():
            edges.append(explorer.Current())
            explorer.Next()
        return edges

    def build_curves_3d(
        self, edge: TopoDS_Shape, tolerance: float, max_segments: int | None = None
    ) -> None:
        if max_segments is None:
            BRepLib.BuildCurves3d_s(edge, tolerance)
        else:
            BRepLib.BuildCurves3d_s(edge, tolerance, MaxSegment=max_segments)

    def find_plane(self, shape: TopoDS_Shape) -> tuple[Point, Point, Point] | None:
        finder = BRepLib_FindSurface(shape, OnlyPlane=True)
        if not finder.Found():
            return None
        surface = finder.Surface()
        if not isinstance(surface, Geom_Plane):
            return None
        pln = surface.Pln()
        return (
            _pt(pln.Location()),
            _pt(pln.Axis().Direction()),
            _pt(pln.XAxis().Direction()),
        )

    def continuity_between(
        self,
        edge1: TopoDS_Shape,
        edge2: TopoDS_Shape,
        u1: float,
        u2: float,
        tolerance: float,
    ) -> str:
        continuity = BRepLProp.Continuity_s(
            BRepAdaptor_Curve(TopoDS.Edge(edge1)),
            BRepAdaptor_Curve(TopoDS.Edge(edge2)),
            u1,
            u2,
            tolerance,
            tolerance,
        )
        return _CONTINUITY_LABELS[continuity]

    def planar_intersections(
        self,
        edge: TopoDS_Shape,
        bounds: tuple[float, float],
        other: TopoDS_Shape | None,
        other_bounds: tuple[float, float] | None,
        plane: gp_Pln,
        tolerance: float,
    ) -> list[tuple[float, float]]:
        surface = BRep_Tool.Surface_s(BRepBuilderAPI_MakeFace(plane).Face())
        curve = BRep_Tool.CurveOnPlane_s(
            TopoDS.Edge(edge), surface, TopLoc_Location(), *bounds
        )
        if other is not None:
            assert other_bounds is not None
            other_curve = BRep_Tool.CurveOnPlane_s(
                TopoDS.Edge(other), surface, TopLoc_Location(), *other_bounds
            )
            intersector = Geom2dAPI_InterCurveCurve(curve, other_curve, tolerance)
        else:
            intersector = Geom2dAPI_InterCurveCurve(curve, tolerance)
        return [
            (intersector.Point(i).X(), intersector.Point(i).Y())
            for i in range(1, intersector.NbPoints() + 1)
        ]

    def nearest_edge(
        self, shape: TopoDS_Shape, point: Point
    ) -> tuple[TopoDS_Shape, float] | None:
        extrema = BRepExtrema_DistShapeShape(self.make_vertex(point), shape)
        extrema.Perform()
        if not extrema.IsDone() or extrema.NbSolution() == 0:
            return None
        support = extrema.SupportTypeShape2(1)
        if support == BRepExtrema_SupportType.BRepExtrema_IsOnEdge:
            edge = TopoDS.Edge(extrema.SupportOnShape2(1))
            return edge, extrema.ParOnEdgeS2(1)[0]
        if support == BRepExtrema_SupportType.BRepExtrema_IsVertex:
            vertex = TopoDS.Vertex(extrema.SupportOnShape2(1))
            edge = TopoDS.Edge(
                self.ancestors(shape, TopoKind.VERTEX, TopoKind.EDGE).get(vertex)[0]
            )
            return edge, BRep_Tool.Parameter_s(vertex, edge)
        return None  # pragma: no cover

    def edge_uv_on_face(
        self, edge: TopoDS_Shape, face: TopoDS_Shape, params: Sequence[float]
    ) -> list[tuple[float, float]]:
        topods_edge = TopoDS.Edge(edge)
        pcurve = BRep_Tool.CurveOnSurface_s(
            topods_edge, TopoDS.Face(face), *BRep_Tool.Range_s(topods_edge)
        )
        return [(uv.X(), uv.Y()) for uv in (pcurve.Value(t) for t in params)]

    def vertex_uv_on_face(
        self, vertex: TopoDS_Shape, face: TopoDS_Shape
    ) -> tuple[float, float]:
        uv = BRep_Tool.Parameters_s(TopoDS.Vertex(vertex), TopoDS.Face(face))
        return uv.X(), uv.Y()

    def frame_at(
        self, shape: TopoDS_Shape, param: float, corrected: bool
    ) -> tuple[Point, Point, Point, Point]:
        curve = OcctCurveView(shape).wrapped
        law: GeomFill_TrihedronLaw = (
            GeomFill_CorrectedFrenet() if corrected else GeomFill_Frenet()
        )
        law.SetCurve(curve)
        tangent, normal, binormal = gp_Vec(), gp_Vec(), gp_Vec()
        law.D0(param, tangent, normal, binormal)
        return _pt(curve.Value(param)), _pt(tangent), _pt(normal), _pt(binormal)

    def make_polygon(self, points: Sequence[Point]) -> TopoDS_Shape:
        builder = BRepBuilderAPI_MakePolygon()
        for point in points:
            builder.Add(gp_Pnt(*point))
        return builder.Wire()

    def connect_edges_to_wires(
        self, edges: Sequence[TopoDS_Shape], tolerance: float
    ) -> list[TopoDS_Shape]:
        edges_in = HSequence_TopoDS_Shape()
        wires_out = HSequence_TopoDS_Shape()
        for edge in edges:
            edges_in.Append(edge)
        ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(
            edges_in, tolerance, False, wires_out
        )
        return [
            TopoDS.Wire(wires_out.Value(i)) for i in range(1, wires_out.Length() + 1)
        ]

    def outer_wire(self, face: TopoDS_Shape) -> TopoDS_Shape:
        return BRepTools.OuterWire_s(TopoDS.Face(face))

    def fillet_2d(
        self, face: TopoDS_Shape, corners: Sequence[tuple[TopoDS_Shape, float]]
    ) -> tuple[TopoDS_Shape, Any]:
        builder = BRepFilletAPI_MakeFillet2d(TopoDS.Face(face))
        for vertex, radius in corners:
            builder.AddFillet(TopoDS.Vertex(vertex), radius)
        builder.Build()
        return builder.Shape(), builder

    def chamfer_2d(
        self,
        face: TopoDS_Shape,
        corners: Sequence[tuple[TopoDS_Shape, TopoDS_Shape, float, float]],
    ) -> tuple[TopoDS_Shape, Any]:
        builder = BRepFilletAPI_MakeFillet2d(TopoDS.Face(face))
        for edge1, edge2, distance1, distance2 in corners:
            builder.AddChamfer(
                TopoDS.Edge(edge1), TopoDS.Edge(edge2), distance1, distance2
            )
        builder.Build()
        return builder.Shape(), builder

    def fix_small_edges(
        self, wire: TopoDS_Shape, precision: float, max_tolerance: float
    ) -> TopoDS_Shape:
        fixer = ShapeFix_Wireframe(wire)
        fixer.SetPrecision(precision)
        fixer.SetMaxTolerance(max_tolerance)
        fixer.FixSmallEdges()
        fixer.FixWireGaps()
        return self.downcast(fixer.Shape())

    def project(
        self,
        shape: TopoDS_Shape,
        target: TopoDS_Shape,
        *,
        direction: Point | None = None,
        center: Point | None = None,
    ) -> list[TopoDS_Shape]:
        if direction is not None:
            projection = BRepProj_Projection(shape, target, gp_Dir(*direction))
        else:
            assert center is not None
            projection = BRepProj_Projection(shape, target, gp_Pnt(*center))
        wires: list[TopoDS_Shape] = []
        while projection.More():
            wires.append(projection.Current())
            projection.Next()
        return wires

    def offset_2d(
        self,
        wire: TopoDS_Shape,
        distance: float,
        join: JoinKind,
        plane: gp_Pln | None = None,
    ) -> TopoDS_Shape | None:
        builder = BRepOffsetAPI_MakeOffset()
        if plane is not None:
            builder.Init(BRepBuilderAPI_MakeFace(plane).Face(), _JOIN[join])
        else:
            builder.Init(_JOIN[join])
        builder.AddWire(TopoDS.Wire(wire))
        builder.Perform(distance)
        if not builder.IsDone():
            return None
        return self.downcast(builder.Shape())

    def offset_shape(
        self,
        shape: TopoDS_Shape,
        amount: float,
        tolerance: float,
        *,
        join: JoinKind = JoinKind.ARC,
        intersection: bool = False,
        thicken: bool = False,
        remove_internal_edges: bool = False,
    ) -> TopoDS_Shape:
        builder = BRepOffset_MakeOffset()
        builder.Initialize(
            shape,
            Offset=amount,
            Tol=tolerance,
            Mode=BRepOffset_Skin,
            Intersection=intersection,
            SelfInter=False,
            Join=_JOIN[join],
            Thickening=thicken,
            RemoveIntEdges=remove_internal_edges,
        )
        builder.MakeOffsetShape()
        return builder.Shape()

    def make_bezier_surface(
        self,
        points: Sequence[Sequence[Point]],
        weights: Sequence[Sequence[float]] | None = None,
    ) -> TopoDS_Shape:
        if weights:
            surface = Geom_BezierSurface(_point_grid(points), _double_grid(weights))
        else:
            surface = Geom_BezierSurface(_point_grid(points))
        return BRepBuilderAPI_MakeFace(surface, Precision.Confusion_s()).Face()

    def make_spline_surface_fit(
        self,
        points: Sequence[Sequence[Point]],
        tolerance: float,
        min_degree: int,
        max_degree: int,
        smoothing: tuple[float, float, float] | None = None,
    ) -> TopoDS_Shape:
        grid = _point_grid(points)
        if smoothing:
            builder = GeomAPI_PointsToBSplineSurface(
                grid, *smoothing, DegMax=max_degree, Tol3D=tolerance
            )
        else:
            builder = GeomAPI_PointsToBSplineSurface(
                grid, DegMin=min_degree, DegMax=max_degree, Tol3D=tolerance
            )
        if not builder.IsDone():
            raise ValueError("B-spline approximation failed")
        return BRepBuilderAPI_MakeFace(
            builder.Surface(), Precision.Confusion_s()
        ).Face()

    def filling(self, **options: Any) -> OcctSurfaceFilling:
        return OcctSurfaceFilling(**options)

    def draft_prism(
        self,
        shape: TopoDS_Shape,
        face: TopoDS_Shape,
        basis: TopoDS_Shape | None,
        taper: float,
        additive: bool,
        *,
        up_to_face: TopoDS_Shape | None = None,
        depth: float | None = None,
    ) -> TopoDS_Shape:
        feature = BRepFeat_MakeDPrism(
            shape,
            TopoDS.Face(face),
            TopoDS.Face(basis) if basis is not None else TopoDS_Face(),
            taper,
            additive,
            False,
        )
        if up_to_face is not None:
            feature.Perform(TopoDS.Face(up_to_face))
        elif depth is None:
            feature.PerformThruAll()
        else:
            feature.Perform(depth)
        return feature.Shape()

    def extrude_generated(
        self, shape: TopoDS_Shape, direction: Point
    ) -> tuple[TopoDS_Shape, AncestorMap, Any]:
        builder = BRepPrimAPI_MakePrism(shape, gp_Vec(*direction))
        prism = builder.Shape()
        entries = [
            (edge, list(builder.Generated(edge)))
            for edge in self.sub_shapes(shape, TopoKind.EDGE)
        ]
        return prism, AncestorMap(self, entries), builder  # type: ignore[arg-type]

    def draft(
        self,
        shape: TopoDS_Shape,
        specs: Sequence[tuple[TopoDS_Shape, float]],
        direction: Point,
        plane: gp_Pln,
    ) -> tuple[TopoDS_Shape, Any]:
        builder = BRepOffsetAPI_DraftAngle(shape)
        pull = gp_Dir(*direction)
        for index, (face, angle) in enumerate(specs):
            builder.Add(TopoDS.Face(face), pull, angle, plane, True)
            if not builder.AddDone():
                raise DraftFailure("add", index, builder.ProblematicShape())
        try:
            builder.Build()
            done = builder.IsDone()
        except self.failure_types as err:  # type: ignore[attr-defined]
            raise DraftFailure("build", None, builder.ProblematicShape()) from err
        if not done:
            raise DraftFailure("build", None, builder.ProblematicShape())
        return builder.Shape(), builder

    def split_by_edges(
        self, shape: TopoDS_Shape, edges: Sequence[TopoDS_Shape]
    ) -> tuple[list[TopoDS_Shape], list[TopoDS_Shape]]:
        splitter = BRepFeat_SplitShape(shape)
        edge_sequence = Sequence_TopoDS_Shape()
        for edge in edges:
            edge_sequence.Append(edge)
        splitter.Add(edge_sequence)
        splitter.Build()
        return (
            [s for s in splitter.Left() if not s.IsNull()],
            [s for s in splitter.Right() if not s.IsNull()],
        )

    def line_intersections(
        self, shape: TopoDS_Shape, origin: Point, direction: Point, tolerance: float
    ) -> list[tuple[TopoDS_Shape, Point]]:
        intersector = BRepIntCurveSurface_Inter()
        intersector.Init(shape, gp_Lin(gp_Pnt(*origin), gp_Dir(*direction)), tolerance)
        found: list[tuple[TopoDS_Shape, Point]] = []
        while intersector.More():
            found.append((intersector.Face(), _pt(intersector.Pnt())))
            intersector.Next()
        return found

    def solid_from_shell(self, shell: TopoDS_Shape) -> TopoDS_Shape:
        return ShapeFix_Solid().SolidFromShell(TopoDS.Shell(shell))

    def remove_wires(
        self, face: TopoDS_Shape, wires: Sequence[TopoDS_Shape]
    ) -> TopoDS_Shape:
        reshaper = BRepTools_ReShape()
        for wire in wires:
            reshaper.Remove(wire)
        return self.downcast(reshaper.Apply(face))

    def is_seam(self, edge: TopoDS_Shape, face: TopoDS_Shape) -> bool:
        return ShapeAnalysis_Edge().IsSeam(TopoDS.Edge(edge), TopoDS.Face(face))

    def to_splines(
        self, shape: TopoDS_Shape, degree: int, tolerance: float, nurbs: bool
    ) -> TopoDS_Shape:
        return ShapeCustom.BSplineRestriction_s(
            shape,
            tolerance,  # 3D tolerance
            tolerance,  # 2D tolerance
            degree,
            1,  # dummy value, degree is leading
            ga.GeomAbs_C0,
            ga.GeomAbs_C0,
            True,  # set degree to be leading
            not nurbs,
            ShapeCustom_RestrictionParameters(),
        )

    def edge_in_plane(
        self, edge: TopoDS_Shape, normal: Point, tolerance: float
    ) -> bool:
        t_edge = tcast(BRep_TEdge, edge.TShape())
        for curve in t_edge.Curves():
            if curve.IsCurve3D():
                return ShapeAnalysis_Curve.IsPlanar_s(
                    curve.Curve3D(), gp_XYZ(*normal), tolerance
                )
        return False

    def offset_face(
        self, face: TopoDS_Shape, amount: float, tolerance: float
    ) -> TopoDS_Shape:
        offsetter = BRepOffset_MakeOffset()
        offsetter.Initialize(TopoDS.Face(face), Offset=amount, Tol=tolerance)
        offsetter.MakeOffsetShape()
        return offsetter.Shape()

    def edge_uv_ends_on_face(
        self, edge: TopoDS_Shape, face: TopoDS_Shape
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        curve = BRepAdaptor_Curve2d(TopoDS.Edge(edge), TopoDS.Face(face))
        ends = []
        for param, sign in (
            (curve.FirstParameter(), 1.0),
            (curve.LastParameter(), -1.0),
        ):
            point, tangent = gp_Pnt2d(), gp_Vec2d()
            curve.D1(param, point, tangent)
            ends.append(
                ((point.X(), point.Y()), (sign * tangent.X(), sign * tangent.Y()))
            )
        return ends

    def vertex_point(self, vertex: TopoDS_Shape) -> Point:
        return _pt(BRep_Tool.Pnt_s(TopoDS.Vertex(vertex)))

    def make_wire(
        self, shapes: Sequence[TopoDS_Shape]
    ) -> tuple[TopoDS_Shape | None, WireStatus]:
        builder = BRepBuilderAPI_MakeWire()
        pending: list[TopoDS_Shape] = []  # a run of edges is added as one list

        def flush() -> None:
            if pending:
                builder.Add(_shape_list(pending))
                pending.clear()

        for shape in shapes:
            if shape.ShapeType() == ta.TopAbs_WIRE:
                flush()
                builder.Add(TopoDS.Wire(shape))
            else:
                pending.append(shape)
        flush()
        builder.Build()
        if not builder.IsDone():
            error = builder.Error()
            if error == BRepBuilderAPI_NonManifoldWire:
                return builder.Wire(), WireStatus.NON_MANIFOLD
            if error == BRepBuilderAPI_EmptyWire:
                return None, WireStatus.EMPTY
            if error == BRepBuilderAPI_DisconnectedWire:
                return None, WireStatus.DISCONNECTED
        return builder.Wire(), WireStatus.OK

    def make_wire_placed(
        self, edges: Sequence[TopoDS_Shape]
    ) -> tuple[TopoDS_Shape, list[TopoDS_Shape]]:
        builder = BRepBuilderAPI_MakeWire()
        placed: list[TopoDS_Shape] = []
        for edge in edges:
            builder.Add(TopoDS.Edge(edge))
            placed.append(builder.Edge())
        builder.Build()
        return builder.Wire(), placed

    def make_planar_face(
        self, outer: TopoDS_Shape, inners: Sequence[TopoDS_Shape]
    ) -> TopoDS_Shape:
        inners = list(inners)
        if inners and not BRep_Tool.IsClosed_s(outer):
            raise ValueError("Cannot build face(s): outer wire is not closed")
        # every wire must lie in one plane
        if not BRepLib_FindSurface(
            self.make_compound([outer, *inners]), OnlyPlane=True
        ).Found():
            raise ValueError("Cannot build face(s): wires not planar")

        builder = BRepBuilderAPI_MakeFace(TopoDS.Wire(_fixed(outer)), True)
        for inner in inners:
            if not BRep_Tool.IsClosed_s(inner):
                raise ValueError("Cannot build face(s): inner wire is not closed")
            builder.Add(TopoDS.Wire(_fixed(inner)))
        builder.Build()
        if not builder.IsDone():
            raise ValueError(f"Cannot build face(s): {builder.Error()}")

        fixer = ShapeFix_Face(builder.Face())
        fixer.FixOrientation()
        fixer.Perform()
        return TopoDS.Face(fixer.Result())

    def add_holes(
        self, face: TopoDS_Shape, wires: Sequence[TopoDS_Shape]
    ) -> TopoDS_Shape:
        builder = BRepBuilderAPI_MakeFace(TopoDS.Face(face))
        for wire in wires:
            builder.Add(TopoDS.Wire(wire))
        return builder.Face()

    def face_of_plane(
        self, plane: gp_Pln, bounds: tuple[float, float, float, float] | None = None
    ) -> TopoDS_Shape:
        if bounds is None:
            return BRepBuilderAPI_MakeFace(plane).Face()
        return BRepBuilderAPI_MakeFace(plane, *bounds).Face()

    def face_frame(self, face: TopoDS_Shape) -> tuple[Point, Point, Point]:
        topods_face = TopoDS.Face(face)
        surface = BRep_Tool.Surface_s(topods_face)
        properties = GProp_GProps()
        BRepGProp.SurfaceProperties_s(topods_face, properties)
        if isinstance(surface, Geom_BoundedSurface):
            point, face_x_dir, tangent_v = gp_Pnt(), gp_Vec(), gp_Vec()
            surface.D1(0.5, 0.5, point, face_x_dir, tangent_v)
        else:
            face_x_dir = gp_Vec(surface.Position().XDirection())  # type: ignore[attr-defined]
        origin, normal = gp_Pnt(), gp_Vec()
        projector = GeomAPI_ProjectPointOnSurf(origin, surface)
        u_val, v_val = projector.LowerDistanceParameters()
        BRepGProp_Face(topods_face).Normal(u_val, v_val, origin, normal)
        return _pt(properties.CentreOfMass()), _pt(face_x_dir), _pt(normal)

    def planar_surface(self, face: TopoDS_Shape) -> gp_Pln | None:
        searcher = GeomLib_IsPlanarSurface(
            BRep_Tool.Surface_s(TopoDS.Face(face)), _PLANAR_TOLERANCE
        )
        if not searcher.IsPlanar():
            return None
        pln = searcher.Plan()
        if not pln.Position().Direct():  # a left-handed plane was returned
            pln = gp_Pln(gp_Ax3(pln.Position().Ax2()))
        return pln

    def surface_axis(self, face: TopoDS_Shape) -> tuple[str, Point, Point] | None:
        surface: Geom_Surface = BRep_Tool.Surface_s(TopoDS.Face(face))
        # strip trimming and offsetting to get at the basis surface
        while isinstance(surface, (Geom_RectangularTrimmedSurface, Geom_OffsetSurface)):
            surface = surface.BasisSurface()
        geom_type = _SURFACE_TYPE_LABELS[GeomAdaptor_Surface(surface).GetType()]
        axis: gp_Ax1
        if geom_type == "CONE":
            axis = surface.Cone().Axis()  # type: ignore[attr-defined]
        elif geom_type == "CYLINDER":
            axis = surface.Cylinder().Axis()  # type: ignore[attr-defined]
        elif geom_type == "SPHERE":
            frame = surface.Position()  # type: ignore[attr-defined]
            axis = gp_Ax1(frame.Location(), frame.Direction())
        elif geom_type == "TORUS":
            axis = surface.Torus().Axis()  # type: ignore[attr-defined]
        elif geom_type == "REVOLUTION":
            axis = surface.Axis()  # type: ignore[attr-defined]
        else:
            return None
        return geom_type, _pt(axis.Location()), _pt(axis.Direction())

    def half_space(self, shape: TopoDS_Shape, point: Point) -> TopoDS_Shape:
        if shape.ShapeType() == ta.TopAbs_SHELL:
            return BRepPrimAPI_MakeHalfSpace(
                TopoDS.Shell(shape), gp_Pnt(*point)
            ).Solid()
        return BRepPrimAPI_MakeHalfSpace(TopoDS.Face(shape), gp_Pnt(*point)).Solid()

    def ruled_surface(self, a: TopoDS_Shape, b: TopoDS_Shape) -> TopoDS_Shape:
        if a.ShapeType() == ta.TopAbs_WIRE:
            return BRepFill.Shell_s(TopoDS.Wire(a), TopoDS.Wire(b))
        return BRepFill.Face_s(TopoDS.Edge(a), TopoDS.Edge(b))

    def make_shell(self, faces: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
        shell = TopoDS_Shell()
        builder = BRep_Builder()
        builder.MakeShell(shell)
        for face in faces:
            builder.Add(shell, face)
        return shell

    def sew(self, faces: Sequence[TopoDS_Shape]) -> tuple[TopoDS_Shape, Any]:
        sewing = BRepBuilderAPI_Sewing()
        for face in faces:
            sewing.Add(face)
        sewing.Perform()
        return self.downcast(sewing.SewedShape()), sewing

    def make_compound(self, shapes: Sequence[TopoDS_Shape]) -> TopoDS_Shape:
        compound = TopoDS_Compound()
        builder = TopoDS_Builder()
        builder.MakeCompound(compound)
        for shape in shapes:
            builder.Add(compound, shape)
        return compound

    def remove_from_compound(self, compound: TopoDS_Shape, shape: TopoDS_Shape) -> None:
        TopoDS_Builder().Remove(compound, shape)

    def transformed(
        self, shape: TopoDS_Shape, transform: gp_Trsf, copy: bool = True
    ) -> tuple[TopoDS_Shape, Any]:
        builder = BRepBuilderAPI_Transform(shape, transform, copy)
        return self.downcast(builder.Shape()), builder

    def transformed_general(
        self, shape: TopoDS_Shape, transform: gp_GTrsf
    ) -> TopoDS_Shape:
        return self.downcast(BRepBuilderAPI_GTransform(shape, transform, True).Shape())

    # ---- distance and classification ----------------------------------------

    def distance(self, a: TopoDS_Shape, b: TopoDS_Shape) -> float:
        return BRepExtrema_DistShapeShape(a, b).Value()

    def closest_points(
        self, a: TopoDS_Shape, b: TopoDS_Shape
    ) -> tuple[float, Point, Point]:
        calculator = BRepExtrema_DistShapeShape()
        calculator.LoadS1(a)
        calculator.LoadS2(b)
        calculator.Perform()
        return (
            calculator.Value(),
            _pt(calculator.PointOnShape1(1)),
            _pt(calculator.PointOnShape2(1)),
        )

    def extrema(
        self, a: TopoDS_Shape, b: TopoDS_Shape, deflection: float | None = None
    ) -> tuple[float, list[tuple[Point, Point]]]:
        calculator = BRepExtrema_DistShapeShape()
        if deflection is not None:
            calculator.SetDeflection(deflection)
        calculator.LoadS1(a)
        calculator.LoadS2(b)
        calculator.Perform()
        if not calculator.IsDone():
            return float("inf"), []
        return calculator.Value(), [
            (_pt(calculator.PointOnShape1(i)), _pt(calculator.PointOnShape2(i)))
            for i in range(1, calculator.NbSolution() + 1)
        ]

    def point_in_solid(
        self, solid: TopoDS_Shape, point: Point, tolerance: float
    ) -> bool:
        classifier = BRepClass3d_SolidClassifier(solid)
        classifier.Perform(gp_Pnt(*point), tolerance)
        return classifier.State() == ta.TopAbs_IN or classifier.IsOnAFace()

    def point_on_face(self, face: TopoDS_Shape, point: Point, tolerance: float) -> bool:
        classifier = BRepClass3d_SolidClassifier(face)
        classifier.Perform(gp_Pnt(*point), tolerance)
        return classifier.IsOnAFace()

    def uv_in_face(
        self, face: TopoDS_Shape, uv: tuple[float, float], tolerance: float
    ) -> bool:
        classifier = IntTools_FClass2d(TopoDS.Face(face), tolerance)
        return classifier.Perform(gp_Pnt2d(*uv)) == ta.TopAbs_IN

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

    # OCP binds each OCCT exception straight to Exception, so there is no
    # common base class to catch; this is the set an operation may raise
    failure_types = (
        Standard_Failure,
        StdFail_NotDone,
        Standard_ConstructionError,
        Standard_NoSuchObject,
        Standard_DomainError,
    )

    # ---- handles and identity ------------------------------------------------

    def kind(self, shape: TopoDS_Shape) -> TopoKind:
        if shape is None or shape.IsNull():
            raise ValueError("Null TopoDS_Shape object")
        return _TOPABS_TO_KIND[shape.ShapeType()]

    def is_null(self, shape: TopoDS_Shape | None) -> bool:
        return shape is None or shape.IsNull()

    def is_handle(self, obj: Any) -> bool:
        return isinstance(obj, TopoDS_Shape)

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

    def share_geometry(self, copy: TopoDS_Shape, original: TopoDS_Shape) -> None:
        copy.TShape(original.TShape())

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

    def direct_children(self, shape: TopoDS_Shape) -> list[TopoDS_Shape]:
        children: list[TopoDS_Shape] = []
        iterator = TopoDS_Iterator(shape, True, True)
        while iterator.More():
            children.append(iterator.Value())
            iterator.Next()
        return children

    def ancestors(
        self, shape: TopoDS_Shape, sub_kind: TopoKind, ancestor_kind: TopoKind
    ) -> AncestorMap:
        occt_map = (
            IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher()
        )
        TopExp.MapShapesAndAncestors_s(
            shape, _KIND_TO_TOPABS[sub_kind], _KIND_TO_TOPABS[ancestor_kind], occt_map
        )
        entries = []
        for index in range(1, occt_map.Extent() + 1):
            ancestors: list[TopoDS_Shape] = []
            # the kernel list is walked once here; indexing it from Python is slow
            for ancestor in occt_map.FindFromIndex(index):
                ancestors.append(ancestor)
            entries.append((occt_map.FindKey(index), ancestors))
        return AncestorMap(self, entries)

    def edge_vertices(self, edge: TopoDS_Shape) -> tuple[TopoDS_Shape, TopoDS_Shape]:
        topods_edge = TopoDS.Edge(edge)
        return TopExp.FirstVertex_s(topods_edge), TopExp.LastVertex_s(topods_edge)

    def is_closed(self, shape: TopoDS_Shape) -> bool:
        return BRep_Tool.IsClosed_s(shape)

    def is_degenerate(self, edge: TopoDS_Shape) -> bool:
        return BRep_Tool.Degenerated_s(TopoDS.Edge(edge))

    # ---- orientation --------------------------------------------------------

    def is_forward(self, shape: TopoDS_Shape) -> bool:
        return shape.Orientation() == ta.TopAbs_FORWARD

    def is_reversed(self, shape: TopoDS_Shape) -> bool:
        return shape.Orientation() == ta.TopAbs_REVERSED

    def reverse(self, shape: TopoDS_Shape) -> None:
        shape.Reverse()

    def reversed(self, shape: TopoDS_Shape) -> TopoDS_Shape:
        return self.downcast(shape.Reversed())

    def same_orientation(self, a: TopoDS_Shape, b: TopoDS_Shape) -> bool:
        return a.Orientation() == b.Orientation()

    def orient_like(self, shape: TopoDS_Shape, like: TopoDS_Shape) -> None:
        shape.Orientation(like.Orientation())

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

    def measure(self, shape: TopoDS_Shape, kind: TopoKind) -> float:
        properties = GProp_GProps()
        _MASS_PROPERTIES[kind](shape, properties)
        return properties.Mass()

    def center_of_mass(
        self, shape: TopoDS_Shape, kind: TopoKind
    ) -> tuple[float, float, float]:
        properties = GProp_GProps()
        _MASS_PROPERTIES[kind](shape, properties)
        center = properties.CentreOfMass()
        return (center.X(), center.Y(), center.Z())

    def inertia_matrix(self, shape: TopoDS_Shape) -> list[list[float]]:
        matrix = _volume_properties(shape).MatrixOfInertia()
        return [[matrix.Value(i, j) for j in range(1, 4)] for i in range(1, 4)]

    def principal_inertia(self, shape: TopoDS_Shape) -> list[tuple[Point, float]]:
        principal = _volume_properties(shape).PrincipalProperties()
        moments = principal.Moments()
        return [
            (_pt(principal.FirstAxisOfInertia()), moments[0]),
            (_pt(principal.SecondAxisOfInertia()), moments[1]),
            (_pt(principal.ThirdAxisOfInertia()), moments[2]),
        ]

    def static_moments(self, shape: TopoDS_Shape) -> tuple[float, float, float]:
        return _volume_properties(shape).StaticMoments()

    def radius_of_gyration(self, shape: TopoDS_Shape, axis: gp_Ax1) -> float:
        return _volume_properties(shape).RadiusOfGyration(axis)

    def has_inconsistent_orientation(self, shape: TopoDS_Shape) -> bool:
        analyzer = ShapeAnalysis_Shell()
        analyzer.LoadShells(shape)
        analyzer.CheckOrientedShells(shape, True, True)
        return analyzer.HasBadEdges()

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

    # ---- operation records ----------------------------------------------------

    @staticmethod
    def _answers(
        query: Callable[[TopoDS_Shape], Any], shape: TopoDS_Shape
    ) -> list[TopoDS_Shape]:
        """What an algorithm says about one sub-shape, or nothing if it will not say."""
        # OCP binds each OCCT failure straight to Exception, with no common base:
        # BRepFilletAPI_MakeFillet2d raises Standard_Failure on one build and
        # Standard_TypeMismatch on another when asked about a vertex or a face
        try:
            return list(query(shape))
        except Exception:  # pylint: disable=broad-exception-caught
            return []

    def record_modified(self, record: Any, shape: TopoDS_Shape) -> list[TopoDS_Shape]:
        if isinstance(record, BRepTools_History):
            if not BRepTools_History.IsSupportedType_s(shape):
                return []
            return list(record.Modified(shape))
        if isinstance(record, BRepBuilderAPI_Sewing):
            if shape.ShapeType() == ta.TopAbs_FACE:
                return [record.Modified(shape)] if record.IsModified(shape) else []
            return (
                [record.ModifiedSubShape(shape)]
                if record.IsModifiedSubShape(shape)
                else []
            )
        return self._answers(record.Modified, shape)

    def record_generated(self, record: Any, shape: TopoDS_Shape) -> list[TopoDS_Shape]:
        if isinstance(record, BRepTools_History):
            if not BRepTools_History.IsSupportedType_s(shape):
                return []
            return list(record.Generated(shape))
        if isinstance(record, BRepBuilderAPI_Sewing):
            return []
        return [
            s
            for s in self._answers(record.Generated, shape)
            if BRepTools_History.IsSupportedType_s(s)
        ]

    def record_removed(self, record: Any, shape: TopoDS_Shape) -> bool:
        if isinstance(record, BRepTools_History):
            return BRepTools_History.IsSupportedType_s(shape) and record.IsRemoved(
                shape
            )
        return False

    def sewn_shape(self, record: Any) -> TopoDS_Shape:
        return record.SewedShape()

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

    def move(self, shape: TopoDS_Shape, location: TopLoc_Location) -> None:
        shape.Move(location)

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


def _volume_properties(shape: TopoDS_Shape) -> GProp_GProps:
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return properties


def _point_grid(points: Sequence[Sequence[Point]]) -> HArray2_gp_Pnt:
    grid = HArray2_gp_Pnt(1, len(points), 1, len(points[0]))
    for i, row in enumerate(points, start=1):
        for j, point in enumerate(row, start=1):
            grid.SetValue(i, j, gp_Pnt(*point))
    return grid


def _double_grid(values: Sequence[Sequence[float]]) -> HArray2_double:
    grid = HArray2_double(1, len(values), 1, len(values[0]))
    for i, row in enumerate(values, start=1):
        for j, value in enumerate(row, start=1):
            grid.SetValue(i, j, float(value))
    return grid


def _point_array(points: Sequence[Point]) -> Array1_gp_Pnt:
    array = Array1_gp_Pnt(1, len(points))
    for index, point in enumerate(points, start=1):
        array.SetValue(index, gp_Pnt(*point))
    return array


def _double_array(values: Sequence[float]) -> Array1_double:
    array = Array1_double(1, len(values))
    for index, value in enumerate(values, start=1):
        array.SetValue(index, float(value))
    return array


def _fixed(shape: TopoDS_Shape) -> TopoDS_Shape:
    """``shape`` after a general shape fix (used on wires before making faces)."""
    fixer = ShapeFix_Shape(shape)
    fixer.Perform()
    return fixer.Shape()


def _shape_list(shapes: Sequence[TopoDS_Shape]) -> List_TopoDS_Shape:
    """A kernel list holding ``shapes`` in order."""
    result = List_TopoDS_Shape()
    for shape in shapes:
        result.Append(shape)
    return result


def kind_to_topabs(kind: TopoKind) -> ta.TopAbs_ShapeEnum:
    """The OCCT enum for ``kind`` (for modules that still drive OCP directly)."""
    return _KIND_TO_TOPABS[kind]
