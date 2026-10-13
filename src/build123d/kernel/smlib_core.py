"""
build123d kernel facade

name: smlib_core.py
by:   Gumyr
date: October 12, 2026

desc:
    The SMLib backend of :class:`~build123d.kernel.base.Kernel`, built on the
    kernel-level ``_smlib_core`` bindings generated from the SMLib headers
    (``usd-brep/source/SmPyCore``). This is Phase 2 of the facade work.

    Handles are :class:`SmHandle` objects. A solid or shell handle owns an
    ``SmBrep``; faces, edges and vertices are the kernel's own entities,
    owned by their brep; wires and compounds, which SMLib does not have,
    are Python containers of handles. Orientation is a flag on the handle,
    placement is baked into the geometry and only remembered on the handle,
    and identity is the kernel entity's address.

    Operations that SMLib does not offer raise NotImplementedError naming
    the method, so the gaps are visible rather than silently wrong.

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

# pylint: disable=too-many-public-methods, too-many-lines

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Sequence
from typing import Any

from OCP.gp import (  # placement bookkeeping only
    gp_Ax1,
    gp_Ax2,
    gp_Ax3,
    gp_Dir,
    gp_Pln,
    gp_Pnt,
    gp_Trsf,
    gp_Vec,
)
from OCP.TopLoc import TopLoc_Location

from build123d.kernel.base import (
    AncestorMap,
    BoolOp,
    CurveView,
    JoinKind,
    Kernel,
    Point,
    SurfaceView,
    TopoKind,
    WireStatus,
)

SM_SUCCESS = 1000
_TOLERANCE = 1e-6
_MASS_ACCURACY = 1e-4
_INFINITE = 1e4  # half-size standing in for OCCT's unbounded lines and planes
_HYPERBOLA_SPAN = 20.0  # parameter span standing in for a full hyperbola
_HUGE = 1e300  # "best answer so far" that never prunes a solver


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _cross(a: Sequence[float], b: Sequence[float]) -> Point:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _point_key(point: Sequence[float]) -> tuple[Any, ...]:
    """Identity of a free vertex: its position, rounded well below the kernel
    tolerance so that the ends of meeting edges agree."""
    return ("at",) + tuple(round(c, 9) + 0.0 for c in point)


def _unit(a: Sequence[float]) -> Point:
    length = math.sqrt(_dot(a, a))
    if length == 0:
        raise ValueError("zero vector")
    return (a[0] / length, a[1] / length, a[2] / length)


class SmlibUnavailable(ImportError):
    """``_smlib_core`` could not be imported."""


def _import_core() -> Any:
    try:
        import _smlib_core  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise SmlibUnavailable(
            "the smlib kernel needs the generated _smlib_core module: build it with "
            "usd-brep/source/SmPyCore/gen/build_core.py and run under "
            "~/usdbrep-venv/bridge/run312.sh (Python 3.12 only)"
        ) from err
    return _smlib_core


def _check(result: Any, what: str) -> Any:
    """Raise unless a kernel call succeeded; ``result`` is the status or the
    tuple a call with out-parameters returns. Returns ``result``."""
    status = result[0] if isinstance(result, tuple) else result
    if status != SM_SUCCESS:
        raise RuntimeError(f"SMLib {what} failed with status {status}")
    return result


_partner_ids = itertools.count(1)


class SmHandle:
    """What build123d holds for one SMLib entity, wire or compound.

    ``entity`` is the kernel object (``SmBrep`` for solids and shells,
    ``SmFace``, ``SmEdge`` or ``SmVertex``) or None for containers;
    ``members`` are the handles a wire or compound is made of; ``flip``
    is the orientation; ``location`` is the placement remembered for
    :meth:`Kernel.location_of` (the geometry already carries it);
    ``partner`` groups the copies and moves of one entity.
    """

    __slots__ = (
        "kind",
        "entity",
        "members",
        "flip",
        "location",
        "partner",
        "point",
        "curve",
        "interval",
    )

    def __init__(
        self,
        kind: TopoKind,
        entity: Any = None,
        members: Sequence[SmHandle] = (),
        flip: bool = False,
        location: TopLoc_Location | None = None,
        partner: int | None = None,
        point: Point | None = None,
        curve: Any = None,
        interval: tuple[float, float] | None = None,
    ):
        self.kind = kind
        self.entity = entity
        self.members = tuple(members)
        self.flip = flip
        self.location = TopLoc_Location() if location is None else location
        self.partner = next(_partner_ids) if partner is None else partner
        self.point = point  # a vertex that belongs to no brep yet
        self.curve = curve  # a free edge: a curve that belongs to no brep yet
        self.interval = interval

    def key(self) -> Hashable:
        """Identity without orientation (``IsSame``)."""
        if self.entity is not None:
            return (self.kind, self.entity.address)
        if self.curve is not None:
            return (self.kind, self.curve.address)
        if self.point is not None:
            return (self.kind, self.partner)
        return (self.kind,) + tuple(m.key() for m in self.members)

    def with_flip(self, flip: bool) -> SmHandle:
        return SmHandle(
            self.kind,
            self.entity,
            self.members,
            flip,
            self.location,
            self.partner,
            self.point,
            self.curve,
            self.interval,
        )

    def __repr__(self) -> str:
        return f"<SmHandle {self.kind.name} {self.key()!r}>"


class _Record:
    """What an SMLib operation did, in terms of the input handles.

    ``modified`` maps an input sub-shape key to the handles it became (the
    copy the operation worked on, changed or not); ``removed`` holds the keys
    of input sub-shapes whose copy did not survive.
    """

    def __init__(self) -> None:
        self.modified: dict[Hashable, list[SmHandle]] = {}
        self.removed: set[Hashable] = set()
        self.result: SmHandle | None = None


class SmCurveView(CurveView):
    """``CurveView`` over an ``SmCurve`` and the interval an edge uses of it."""

    def __init__(self, kernel: SmlibCoreKernel, shape: SmHandle):
        self.kernel = kernel
        self.shape = shape
        self.sc = kernel.sc
        if shape.kind is TopoKind.WIRE:
            raise NotImplementedError("smlib: curve views of wires")
        if shape.entity is not None:
            edge = shape.entity
            self.curve = edge.GetCurve()
            interval = edge.GetInterval()
            self.first, self.last = interval.GetMin(), interval.GetMax()
        else:
            self.curve = shape.curve
            assert shape.interval is not None
            self.first, self.last = shape.interval
        self.geom_type = kernel.curve_type(self.curve)

    def _point(self, t: float) -> Point:
        point = self.sc.SmVector3d()
        _check(self.curve.EvaluatePoint(t, point), "curve evaluation")
        return (point.x, point.y, point.z)

    def is_periodic(self) -> bool:
        return bool(self.curve.IsPeriodic(self.sc.SmExtent1d(self.first, self.last)))

    def period(self) -> float:
        return self.last - self.first

    def value(self, t: float) -> Point:
        return self._point(t)

    def _derivatives(self, t: float, order: int) -> list[Point]:
        """Point and derivatives up to ``order`` at ``t``."""
        evaluate = getattr(self.curve, "EvaluateDerivatives", None)
        if evaluate is not None:
            status, vectors = evaluate(t, order, False, False)
            _check(status, "curve evaluation")
            return [(v.x, v.y, v.z) for v in vectors]
        # finite differences when the hand-written binding is missing
        step = max(1e-7, 1e-7 * abs(self.last - self.first))
        points = [self._point(t + i * step) for i in range(order + 1)]
        result = [points[0]]
        level = points
        for _ in range(order):
            level = [
                tuple((b - a) / step for a, b in zip(p, q))
                for p, q in zip(level, level[1:])
            ]
            result.append(level[0])  # type: ignore[arg-type]
        return result

    def d1(self, t: float) -> tuple[Point, Point]:
        point, tangent = self._derivatives(t, 1)
        return point, tangent

    def length(self, t0: float | None = None, t1: float | None = None) -> float:
        a = self.first if t0 is None else t0
        b = self.last if t1 is None else t1
        status, length = self.curve.Length(self.sc.SmExtent1d(a, b), _TOLERANCE)
        _check(status, "curve length")
        return length

    def param_at_length(self, length: float, t0: float | None = None) -> float:
        start = self.first if t0 is None else t0
        result = self.curve.FindParameterAtArcLength(start, length, None)
        status, param = result[0], result[1]
        _check(status, "parameter at length")
        return param

    def closest_param(self, point: Point) -> float:
        return self.drop_point(point)[0]

    def drop_point(self, point: Point) -> tuple[float, float]:
        """The parameter of the nearest point of the curve and its distance."""
        result = self.curve.DropPoint(
            self.sc.SmExtent1d(self.first, self.last),
            self.kernel._vec(point),  # pylint: disable=protected-access
            None,
            _TOLERANCE,
            None,
        )
        _check(result, "point drop")
        return float(result[-2]), float(result[-1])

    def positions_by_deflection(self, deflection: float) -> list[Point]:
        count = 32
        return [
            self._point(self.first + (self.last - self.first) * i / count)
            for i in range(count + 1)
        ]

    def _frame(self) -> tuple[Point, Point, Point, Point]:
        """Origin, x, y and z axes of an analytic curve's placement."""
        position = self.curve.GetPosition()
        pt = self.kernel._pt  # pylint: disable=protected-access
        return (
            pt(position.GetOrigin()),
            pt(position.GetXAxis()),
            pt(position.GetYAxis()),
            pt(position.GetZAxis()),
        )

    def circle(self) -> tuple[Point, Point, float]:
        if self.geom_type != "CIRCLE":
            raise ValueError("not a circle")
        origin, _, _, axis = self._frame()
        return origin, axis, float(self.curve.GetRadius())

    def ellipse(self) -> tuple[Point, Point, float, float]:
        if self.geom_type != "ELLIPSE":
            raise ValueError("not an ellipse")
        origin, _, _, axis = self._frame()
        return origin, axis, float(self.curve.GetXRadius()), float(self.curve.GetYRadius())

    def hyperbola(self) -> tuple[Point, Point, float, float]:
        raise NotImplementedError("smlib: hyperbola parameters")

    def parabola(self) -> tuple[Point, Point, float]:
        raise NotImplementedError("smlib: parabola parameters")

    def _polygon(self) -> tuple[list[Point], list[float] | None]:
        status, poles, weights = self.curve.GetControlPolygon([], [])
        _check(status, "control polygon")
        points = [(p.x, p.y, p.z) for p in poles]
        rational = bool(self.curve.IsRational())
        return points, ([float(w) for w in weights] if rational else None)

    def bezier(self) -> tuple[int, list[Point], list[float] | None]:
        poles, weights = self._polygon()
        return int(self.curve.GetDegree()), poles, weights

    def bspline(self):  # type: ignore[override]
        poles, weights = self._polygon()
        status, flat = self.curve.GetKnotsAll([])
        _check(status, "knots")
        knots: list[float] = []
        multiplicities: list[int] = []
        for knot in flat:
            if knots and abs(knot - knots[-1]) < 1e-12:
                multiplicities[-1] += 1
            else:
                knots.append(float(knot))
                multiplicities.append(1)
        return (
            int(self.curve.GetDegree()),
            self.is_periodic(),
            poles,
            weights,
            knots,
            multiplicities,
        )

    def offset(self) -> tuple[float, Point, Any]:
        raise NotImplementedError("smlib: offset curve parameters")

    def dn(self, t: float, order: int) -> Point:
        return self._derivatives(t, order)[order]

    def is_closed(self) -> bool:
        return bool(
            self.curve.IsClosed(self.sc.SmExtent1d(self.first, self.last), _TOLERANCE)
        )


class SmSurfaceView(SurfaceView):
    """``SurfaceView`` over the surface of an ``SmFace``."""

    def __init__(self, kernel: SmlibCoreKernel, face: SmHandle):
        self.kernel = kernel
        self.sc = kernel.sc
        self.face = face
        self.surface = face.entity.GetSurface()
        self.geom_type = kernel.surface_type(self.surface)
        domain = face.entity.GetUVDomain()
        low, high = domain.GetMin(), domain.GetMax()
        self._bounds = (low.x, high.x, low.y, high.y)

    def uv_bounds(self) -> tuple[float, float, float, float]:
        return self._bounds

    def is_u_periodic(self) -> bool:
        return False

    def is_v_periodic(self) -> bool:
        return False

    def u_period(self) -> float:
        return self._bounds[1] - self._bounds[0]

    def v_period(self) -> float:
        return self._bounds[3] - self._bounds[2]

    def value(self, u: float, v: float) -> Point:
        point = self.sc.SmVector3d()
        _check(
            self.surface.EvaluatePoint(self.sc.SmVector2d(u, v), point),
            "surface evaluation",
        )
        return (point.x, point.y, point.z)

    def d1(self, u: float, v: float) -> tuple[Point, Point, Point]:
        point, du, dv = self.sc.SmVector3d(), self.sc.SmVector3d(), self.sc.SmVector3d()
        _check(
            self.surface.Evaluate1stDerivatives(
                self.sc.SmVector2d(u, v), 0, 0, point, du, dv
            ),
            "surface derivatives",
        )
        return (point.x, point.y, point.z), (du.x, du.y, du.z), (dv.x, dv.y, dv.z)

    def dn(self, u: float, v: float, u_order: int, v_order: int) -> Point:
        evaluate = getattr(self.surface, "EvaluateDerivatives", None)
        if evaluate is None:
            raise NotImplementedError("smlib: higher surface derivatives")
        status, rows = evaluate(self.sc.SmVector2d(u, v), u_order, v_order, False, False)
        _check(status, "surface evaluation")
        vector = rows[u_order][v_order]
        return (vector.x, vector.y, vector.z)

    def normal(self, u: float, v: float) -> tuple[Point, Point]:
        point = self.value(u, v)
        normal = self.sc.SmVector3d()
        _check(
            self.surface.EvaluateNormal(self.sc.SmVector2d(u, v), 0, 0, normal),
            "surface normal",
        )
        sign = -1.0 if self.kernel.face_outward_is_flipped(self.face) else 1.0
        return point, (sign * normal.x, sign * normal.y, sign * normal.z)

    def closest_uv(self, point: Point) -> tuple[float, float]:
        solutions = self.sc.SmSolutionArray()
        _check(
            self.surface.GlobalPointSolve(
                self.face.entity.GetUVDomain(),
                self.sc.SM_SO_MINIMIZE,
                self.kernel._vec(point),  # pylint: disable=protected-access
                self.sc.SmZoneTol3d(_TOLERANCE),
                None,
                self.sc.SM_SR_SINGLE,
                solutions,
            ),
            "point to surface",
        )
        if solutions.GetSize() == 0:
            raise RuntimeError("SMLib found no nearest surface point")
        located, params = self.sc.SmVector3d(), self.sc.SmVector3d()
        _check(solutions.GetAt(0).GetPoint(0, located, params, 0), "solution point")
        return params.x, params.y

    def plane(self) -> tuple[Point, Point, Point] | None:
        if self.geom_type != "PLANE":
            return None
        origin, du, dv = self.d1(*self._mid())
        _, normal = self.normal(*self._mid())
        length = math.sqrt(sum(c * c for c in du)) or 1.0
        return origin, normal, tuple(c / length for c in du)  # type: ignore[return-value]

    def _mid(self) -> tuple[float, float]:
        u0, u1, v0, v1 = self._bounds
        return (u0 + u1) / 2, (v0 + v1) / 2

    def radius(self) -> float:
        if self.geom_type == "CYLINDER":
            return float(self.surface.GetBotRadius())
        if self.geom_type == "SPHERE":
            return float(self.surface.GetRadius())
        raise ValueError("no radius for a " + self.geom_type)

    def semi_angle(self) -> float:
        if self.geom_type != "CONE":
            raise ValueError("not a cone")
        height = float(self.surface.GetHeight())
        bottom = float(self.surface.GetBotRadius())
        top = float(self.surface.GetTopRadius())
        return math.atan2(top - bottom, height)

    def torus_radii(self) -> tuple[float, float]:
        if self.geom_type != "TORUS":
            raise ValueError("not a torus")
        return float(self.surface.GetMajorRadius()), float(self.surface.GetMinorRadius())

    def curvatures(self, u: float, v: float, tolerance: float):  # type: ignore[override]
        raise NotImplementedError("smlib: surface curvatures")

    def uv_of_point(self, point: Point, tolerance: float) -> tuple[float, float, float]:
        u, v = self.closest_uv(point)
        return u, v, math.dist(self.value(u, v), point)

    def iso_lengths(self, u_min, u_max, v_min, v_max, tolerance):  # type: ignore[override]
        raise NotImplementedError("smlib: iso-curve lengths")


def _unsupported(name: str):
    def method(self, *args, **kwargs):  # pylint: disable=unused-argument
        raise NotImplementedError(f"smlib: {name}")

    method.__name__ = name
    return method


class SmlibCoreKernel(Kernel):
    """SMLib through the generated kernel-level bindings."""

    name = "smlib"
    failure_types = (RuntimeError,)

    def __init__(self) -> None:
        self.sc = _import_core()
        self._probe = self._new_brep()
        self.context = self._probe.GetContext()
        self._meshes: dict[int, tuple[Any, float]] = {}  # brep address -> (poly brep, tolerance)
        self._bezier: set[int] = set()  # curve addresses made by make_bezier

    # ---- helpers --------------------------------------------------------------

    def _new_brep(self) -> Any:
        try:
            return self.sc.SmBrep()
        except TypeError:
            return self.sc.SmBrep(9876543.21, 9876543.21, None)

    def _vec(self, point: Sequence[float]) -> Any:
        return self.sc.SmVector3d(float(point[0]), float(point[1]), float(point[2]))

    @staticmethod
    def _pt(vector: Any) -> Point:
        return (vector.x, vector.y, vector.z)

    def _placement(self, frame: Any) -> Any:
        """An ``SmAxis2Placement`` for a ``gp_Ax2`` frame."""
        origin = frame.Location()
        x_dir = frame.XDirection()
        y_dir = frame.YDirection()
        return self.sc.SmAxis2Placement(
            self._vec((origin.X(), origin.Y(), origin.Z())),
            self._vec((x_dir.X(), x_dir.Y(), x_dir.Z())),
            self._vec((y_dir.X(), y_dir.Y(), y_dir.Z())),
        )

    def _placement_of_trsf(self, trsf: gp_Trsf) -> Any:
        """An ``SmAxis2Placement`` for a rigid ``gp_Trsf``."""
        origin = gp_Pnt(0, 0, 0).Transformed(trsf)
        x_dir = gp_Dir(1, 0, 0).Transformed(trsf)
        y_dir = gp_Dir(0, 1, 0).Transformed(trsf)
        return self.sc.SmAxis2Placement(
            self._vec((origin.X(), origin.Y(), origin.Z())),
            self._vec((x_dir.X(), x_dir.Y(), x_dir.Z())),
            self._vec((y_dir.X(), y_dir.Y(), y_dir.Z())),
        )

    def _body(self, shape: SmHandle) -> Any:
        """The ``SmBrep`` holding ``shape``."""
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            return shape.entity
        if shape.entity is not None:
            return shape.entity.GetBrep()
        raise ValueError(f"a {shape.kind.name} has no body")

    def _solid_handle(self, brep: Any) -> SmHandle:
        kind = TopoKind.SOLID if brep.IsManifoldSolid() else TopoKind.SHELL
        return SmHandle(kind, brep)

    def _copy_body(self, brep: Any) -> tuple[Any, Any]:
        """A copy of ``brep`` and the map from its entities to the copy's."""
        forward = self.sc.SmCopyBrepMap()
        copy = self.sc.SmBrep(brep, None, forward, 1, 1)
        return copy, forward

    def _mapped(self, forward: Any, entity: Any) -> Any:
        return forward.GetAt(entity)

    def curve_type(self, curve: Any) -> str:
        sc = self.sc
        for cls, label in (
            (sc.SmLine, "LINE"),
            (sc.SmCircle, "CIRCLE"),
            (sc.SmEllipse, "ELLIPSE"),
            (sc.SmHyperbola, "HYPERBOLA"),
            (sc.SmParabola, "PARABOLA"),
            (sc.SmOffsetCurve, "OFFSET"),
            (sc.SmBSplineCurve, "BSPLINE"),
        ):
            if isinstance(curve, cls):
                if label == "BSPLINE" and curve.address in self._bezier:
                    return "BEZIER"
                return label
        return "OTHER"

    def surface_type(self, surface: Any) -> str:
        sc = self.sc
        for cls, label in (
            (sc.SmPlane, "PLANE"),
            (sc.SmCylinder, "CYLINDER"),
            (sc.SmCone, "CONE"),
            (
                (sc.SmSphere, "SPHERE")
                if hasattr(sc, "SmSphere")
                else (sc.SmPlane, "PLANE")
            ),
            (sc.SmTorus, "TORUS") if hasattr(sc, "SmTorus") else (sc.SmPlane, "PLANE"),
            (sc.SmSurfOfRevolution, "REVOLUTION"),
            (
                (sc.SmSurfOfExtrusion, "EXTRUSION")
                if hasattr(sc, "SmSurfOfExtrusion")
                else (sc.SmPlane, "PLANE")
            ),
            (sc.SmOffsetSurface, "OFFSET"),
            (sc.SmBSplineSurface, "BSPLINE"),
        ):
            if isinstance(surface, cls):
                if label == "CONE" and abs(
                    surface.GetBotRadius() - surface.GetTopRadius()
                ) <= _TOLERANCE * max(1.0, abs(surface.GetBotRadius())):
                    return "CYLINDER"  # SMLib makes cylinders as cones of equal radii
                return label
        return "OTHER"

    def _copy_curve(self, curve: Any) -> Any:
        """A copy of a curve in the kernel context, keeping the bezier tag."""
        status, copy = curve.Copy(self.context)
        _check(status, "curve copy")
        if curve.address in self._bezier:
            self._bezier.add(copy.address)
        return copy

    def face_outward_is_flipped(self, face: SmHandle) -> bool:
        """Whether the face's outward normal is opposite to its surface normal."""
        brep = face.entity.GetBrep()
        upward = face.entity.GetUpwardFaceuse()
        flipped = False
        if brep is not None and upward is not None:
            shell = upward.GetShell()
            if shell is not None and shell.GetRegion() != brep.GetInfiniteRegion():
                flipped = True
        return flipped != face.flip

    def _mass(self, brep: Any) -> tuple[float, float, Point]:
        """Area, volume and centre of mass of a body."""
        status, area, volume, mass, moments = brep.ComputePreciseProperties(
            _MASS_ACCURACY,
            self._vec((0.0, 0.0, 0.0)),
            self.sc.SmMassPropertyBehavior(1.0, 0.0, 0.0, 0.0, 0, 0),
            [],
        )
        _check(status, "mass properties")
        first = moments[4] if len(moments) > 4 else moments[0]
        divisor = mass if abs(mass) > 1e-300 else (area or 1.0)
        return area, volume, (first.x / divisor, first.y / divisor, first.z / divisor)

    # ---- handles and identity ------------------------------------------------

    def kind(self, shape: SmHandle) -> TopoKind:
        if shape is None:
            raise ValueError("Null shape handle")
        return shape.kind

    def is_null(self, shape: SmHandle | None) -> bool:
        return shape is None

    def is_handle(self, obj: Any) -> bool:
        return isinstance(obj, SmHandle)

    def downcast(self, shape: SmHandle) -> SmHandle:
        if shape is None:
            raise ValueError("Null shape handle")
        return shape

    def is_same(self, a: SmHandle, b: SmHandle) -> bool:
        if a.key() == b.key():
            return True
        if a.point is not None and b.point is not None:  # free vertices: by position
            return math.dist(a.point, b.point) < _TOLERANCE
        return False

    def is_equal(self, a: SmHandle, b: SmHandle) -> bool:
        return a.key() == b.key() and a.flip == b.flip

    def is_partner(self, a: SmHandle, b: SmHandle) -> bool:
        return a.key() == b.key() or (a.partner == b.partner and a.kind == b.kind)

    def hash_key(self, shape: SmHandle) -> Hashable:
        return shape.key()

    def copy_deep(self, shape: SmHandle) -> SmHandle:
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            copy, _ = self._copy_body(shape.entity)
            return SmHandle(
                shape.kind,
                copy,
                flip=shape.flip,
                location=shape.location,
                partner=shape.partner,
            )
        if shape.kind in (TopoKind.WIRE, TopoKind.COMPOUND):
            return SmHandle(
                shape.kind,
                members=[self.copy_deep(m) for m in shape.members],
                flip=shape.flip,
                location=shape.location,
                partner=shape.partner,
            )
        if shape.point is not None:
            return SmHandle(TopoKind.VERTEX, point=shape.point, partner=shape.partner)
        if shape.curve is not None:
            return SmHandle(
                TopoKind.EDGE,
                curve=self._copy_curve(shape.curve),
                interval=shape.interval,
                flip=shape.flip,
                location=shape.location,
                partner=shape.partner,
            )
        brep = shape.entity.GetBrep()
        copy, forward = self._copy_body(brep)
        return SmHandle(
            shape.kind,
            self._mapped(forward, shape.entity),
            flip=shape.flip,
            location=shape.location,
            partner=shape.partner,
        )

    # ---- traversal ----------------------------------------------------------

    def _entities(self, brep: Any, kind: TopoKind) -> list[Any]:
        if kind is TopoKind.FACE:
            return list(brep.GetFaces([], None))
        if kind is TopoKind.EDGE:
            return list(brep.GetEdges([], None))
        if kind is TopoKind.VERTEX:
            return list(brep.GetVertices([], None))
        return []

    def sub_shapes(self, shape: SmHandle, kind: TopoKind) -> list[SmHandle]:
        found: dict[Hashable, SmHandle] = {}
        for item in self.sub_shape_uses(shape, kind):
            found.setdefault(item.key(), item)
        return list(found.values())

    def sub_shape_uses(self, shape: SmHandle, kind: TopoKind) -> list[SmHandle]:
        if kind == shape.kind:
            return [shape]
        if shape.kind is TopoKind.COMPOUND:
            return [h for m in shape.members for h in self.sub_shape_uses(m, kind)]
        if shape.kind is TopoKind.WIRE:
            if kind is TopoKind.EDGE:
                return list(shape.members)
            if kind is TopoKind.VERTEX:
                return [v for e in shape.members for v in self.sub_shape_uses(e, kind)]
            return []
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            if kind is TopoKind.SHELL and shape.kind is TopoKind.SOLID:
                return [SmHandle(TopoKind.SHELL, shape.entity, partner=shape.partner)]
            return [SmHandle(kind, e) for e in self._entities(shape.entity, kind)]
        if shape.kind is TopoKind.FACE:
            if kind is TopoKind.WIRE:
                return [
                    self._loop_wire(loop, shape)
                    for loop in shape.entity.GetLoops([], 0, None)
                ]
            if kind is TopoKind.EDGE:
                return [
                    e
                    for w in self.sub_shape_uses(shape, TopoKind.WIRE)
                    for e in w.members
                ]
            if kind is TopoKind.VERTEX:
                return [
                    SmHandle(kind, v)
                    for v in shape.entity.GetVertices([], None, 0, None)
                ]
            return []
        if shape.kind is TopoKind.EDGE:
            if kind is TopoKind.VERTEX:
                if shape.entity is None:
                    start, end = self._free_edge_vertices(shape)
                    return [end, start] if shape.flip else [start, end]
                return [SmHandle(kind, v) for v in self.edge_vertices_entities(shape)]
            return []
        return []

    def edge_vertices_entities(self, edge: SmHandle) -> list[Any]:
        start, end = edge.entity.GetStartVertex(), edge.entity.GetEndVertex()
        if edge.flip:
            start, end = end, start
        return [v for v in (start, end) if v is not None]

    def _free_edge_vertices(self, edge: SmHandle) -> tuple[SmHandle, SmHandle]:
        """The ends of a free edge as point vertices with stable identity."""
        # free edges share no topology, so their end vertices are the same
        # when they are at the same place
        view = SmCurveView(self, edge)
        start, end = view.value(view.first), view.value(view.last)
        return (
            SmHandle(TopoKind.VERTEX, point=start, partner=_point_key(start)),
            SmHandle(TopoKind.VERTEX, point=end, partner=_point_key(end)),
        )

    def _loop_wire(self, loop: Any, face: SmHandle) -> SmHandle:
        """The wire of a face loop, edges in loop order with their orientation."""
        edges = []
        for edgeuse in loop.GetEdgeuses([]):
            edge = edgeuse.GetEdge()
            reversed_use = edgeuse.GetOrientation() == self.sc.SM_OT_OPPOSITE
            edges.append(SmHandle(TopoKind.EDGE, edge, flip=reversed_use != face.flip))
        return SmHandle(TopoKind.WIRE, members=edges)

    def direct_children(self, shape: SmHandle) -> list[SmHandle]:
        if shape.kind in (TopoKind.COMPOUND, TopoKind.WIRE):
            return list(shape.members)
        if shape.kind is TopoKind.SOLID:
            return [SmHandle(TopoKind.SHELL, shape.entity, partner=shape.partner)]
        if shape.kind is TopoKind.SHELL:
            return self.sub_shape_uses(shape, TopoKind.FACE)
        if shape.kind is TopoKind.FACE:
            return self.sub_shape_uses(shape, TopoKind.WIRE)
        if shape.kind is TopoKind.EDGE:
            return self.sub_shape_uses(shape, TopoKind.VERTEX)
        return []

    def ancestors(
        self, shape: SmHandle, sub_kind: TopoKind, ancestor_kind: TopoKind
    ) -> AncestorMap:
        entries = []
        subs = self.sub_shapes(shape, sub_kind)
        if sub_kind is TopoKind.EDGE and ancestor_kind is TopoKind.FACE:
            for sub in subs:
                entries.append(
                    (
                        sub,
                        [
                            SmHandle(TopoKind.FACE, f)
                            for f in sub.entity.GetFaces([], None)
                        ],
                    )
                )
        elif sub_kind is TopoKind.VERTEX and ancestor_kind is TopoKind.EDGE:
            edges = self.sub_shapes(shape, TopoKind.EDGE)
            for sub in subs:
                if sub.entity is None:  # a point vertex of free edges
                    entries.append(
                        (
                            sub,
                            [
                                e
                                for e in edges
                                if any(
                                    self.is_same(v, sub)
                                    for v in self.sub_shape_uses(e, TopoKind.VERTEX)
                                )
                            ],
                        )
                    )
                else:
                    entries.append(
                        (
                            sub,
                            [
                                SmHandle(TopoKind.EDGE, e)
                                for e in sub.entity.GetEdges([], None)
                            ],
                        )
                    )
        else:
            ancestors = self.sub_shapes(shape, ancestor_kind)
            for sub in subs:
                entries.append(
                    (
                        sub,
                        [
                            a
                            for a in ancestors
                            if any(
                                self.is_same(s, sub)
                                for s in self.sub_shapes(a, sub_kind)
                            )
                        ],
                    )
                )
        return AncestorMap(self, entries)

    def edge_vertices(self, edge: SmHandle) -> tuple[SmHandle, SmHandle]:
        if edge.entity is None:
            return self._free_edge_vertices(edge)
        start, end = edge.entity.GetStartVertex(), edge.entity.GetEndVertex()
        return SmHandle(TopoKind.VERTEX, start), SmHandle(TopoKind.VERTEX, end)

    def is_closed(self, shape: SmHandle) -> bool:
        if shape.kind is TopoKind.EDGE:
            if shape.entity is None:
                start, end = self._free_edge_vertices(shape)
                return all(
                    abs(a - b) < _TOLERANCE
                    for a, b in zip(self.vertex_point(start), self.vertex_point(end))
                )
            return bool(shape.entity.IsClosed())
        if shape.kind is TopoKind.WIRE:
            if not shape.members:
                return False
            first = self.vertex_point(
                self.edge_vertices(shape.members[0])[1 if shape.members[0].flip else 0]
            )
            last = self.vertex_point(
                self.edge_vertices(shape.members[-1])[
                    0 if shape.members[-1].flip else 1
                ]
            )
            return all(abs(a - b) < _TOLERANCE for a, b in zip(first, last))
        if shape.kind is TopoKind.SOLID:
            return True
        if shape.kind is TopoKind.SHELL:
            return bool(shape.entity.IsManifoldSolid())
        return False

    def is_degenerate(self, edge: SmHandle) -> bool:
        return self.length(edge) < _TOLERANCE

    # ---- orientation --------------------------------------------------------

    def is_forward(self, shape: SmHandle) -> bool:
        return not shape.flip

    def is_reversed(self, shape: SmHandle) -> bool:
        return shape.flip

    def reverse(self, shape: SmHandle) -> None:
        shape.flip = not shape.flip

    def reversed(self, shape: SmHandle) -> SmHandle:
        return shape.with_flip(not shape.flip)

    def same_orientation(self, a: SmHandle, b: SmHandle) -> bool:
        return a.flip == b.flip

    def orient_like(self, shape: SmHandle, like: SmHandle) -> None:
        shape.flip = like.flip

    # ---- validity and repair ------------------------------------------------

    def is_valid(self, shape: SmHandle) -> bool:
        return True

    def fix(self, shape: SmHandle) -> SmHandle:
        return shape

    # ---- measures -----------------------------------------------------------

    def area(self, shape: SmHandle) -> float:
        if shape.kind is TopoKind.SOLID:
            return self._mass(shape.entity)[0]
        if shape.kind is TopoKind.SHELL:
            return sum(
                self._face_properties(f)[0]
                for f in self.sub_shape_uses(shape, TopoKind.FACE)
            )
        if shape.kind is TopoKind.FACE:
            return self._face_properties(shape)[0]
        if shape.kind is TopoKind.COMPOUND:
            return sum(self.area(m) for m in shape.members)
        return 0.0

    def _face_properties(self, face: SmHandle) -> tuple[float, Point]:
        status, area, _, moments = face.entity.ComputePreciseProperties(
            self.sc.SM_OT_SAME, _MASS_ACCURACY, self._vec((0.0, 0.0, 0.0)), 0.0, 0.0, []
        )
        _check(status, "face properties")
        first = moments[0]
        divisor = area if area > 1e-300 else 1.0
        return area, (first.x / divisor, first.y / divisor, first.z / divisor)

    def volume(self, shape: SmHandle) -> float:
        if shape.kind is TopoKind.SOLID:
            return self._mass(shape.entity)[1]
        if shape.kind is TopoKind.COMPOUND:
            return sum(self.volume(m) for m in shape.members)
        return 0.0

    def length(self, shape: SmHandle) -> float:
        if shape.kind is TopoKind.EDGE:
            return SmCurveView(self, shape).length()
        return sum(self.length(e) for e in self.sub_shapes(shape, TopoKind.EDGE))

    def measure(self, shape: SmHandle, kind: TopoKind) -> float:
        if kind in (TopoKind.EDGE, TopoKind.WIRE):
            return self.length(shape)
        if kind in (TopoKind.FACE, TopoKind.SHELL):
            return self.area(shape)
        return self.volume(shape)

    def center_of_mass(self, shape: SmHandle, kind: TopoKind) -> Point:
        if shape.kind is TopoKind.VERTEX:
            return self.vertex_point(shape)
        if (
            kind in (TopoKind.SOLID, TopoKind.COMPSOLID, TopoKind.COMPOUND)
            and shape.kind is TopoKind.SOLID
        ):
            return self._mass(shape.entity)[2]
        if shape.kind is TopoKind.FACE:
            area, center = self._face_properties(shape)
            return center
        if shape.kind is TopoKind.EDGE:
            view = SmCurveView(self, shape)
            points = view.positions_by_deflection(0.0)
            n = len(points)
            return tuple(sum(p[i] for p in points) / n for i in range(3))  # type: ignore[return-value]
        # weighted average of the members' centres
        parts = []
        for member in (
            shape.members if shape.members else self.sub_shapes(shape, TopoKind.FACE)
        ):
            weight = self.measure(
                member, kind if kind is not TopoKind.COMPOUND else member.kind
            )
            parts.append((weight, self.center_of_mass(member, member.kind)))
        total = sum(w for w, _ in parts)
        if total <= 0:
            total = len(parts) or 1
            parts = [(1.0, c) for _, c in parts]
        return tuple(sum(w * c[i] for w, c in parts) / total for i in range(3))  # type: ignore[return-value]

    def bounding_box(
        self, shape: SmHandle, optimal: bool = True
    ) -> tuple[float, float, float, float, float, float] | None:
        boxes = []
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            extent = self.sc.SmExtent3d()
            _check(shape.entity.CalculateTightBoundingBox(extent), "bounding box")
            boxes.append(
                tuple(extent.GetMin(i) for i in range(3))
                + tuple(extent.GetMax(i) for i in range(3))
            )
        elif shape.kind is TopoKind.VERTEX:
            p = self.vertex_point(shape)
            boxes.append(p + p)
        elif shape.kind is TopoKind.EDGE:
            points = SmCurveView(self, shape).positions_by_deflection(0.0)
            boxes.append(
                tuple(min(p[i] for p in points) for i in range(3))
                + tuple(max(p[i] for p in points) for i in range(3))
            )
        else:
            members = (
                shape.members
                if shape.members
                else self.sub_shapes(shape, TopoKind.EDGE)
            )
            for member in members:
                box = self.bounding_box(member, optimal)
                if box is not None:
                    boxes.append(box)
        if not boxes:
            return None
        return tuple(min(b[i] for b in boxes) for i in range(3)) + tuple(max(b[i + 3] for b in boxes) for i in range(3))  # type: ignore[return-value]

    def inertia_matrix(self, shape: SmHandle) -> list[list[float]]:
        raise NotImplementedError("smlib: inertia matrix")

    def principal_inertia(self, shape: SmHandle) -> list[tuple[Point, float]]:
        raise NotImplementedError("smlib: principal inertia")

    def static_moments(self, shape: SmHandle) -> Point:
        raise NotImplementedError("smlib: static moments")

    def radius_of_gyration(self, shape: SmHandle, axis: Any) -> float:
        raise NotImplementedError("smlib: radius of gyration")

    def has_inconsistent_orientation(self, shape: SmHandle) -> bool:
        return False

    # ---- operations ---------------------------------------------------------

    def _tracked_subshapes(self, shape: SmHandle) -> list[SmHandle]:
        """The sub-shapes history keeps track of, the shape itself included."""
        found: dict[Hashable, SmHandle] = {}
        for kind in (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE, TopoKind.SOLID):
            for sub in self.sub_shapes(shape, kind):
                found.setdefault(sub.key(), sub)
        return list(found.values())

    def _is_one_dimensional(self, shape: SmHandle) -> bool:
        if shape.kind in (TopoKind.EDGE, TopoKind.WIRE, TopoKind.VERTEX):
            return True
        if shape.kind is TopoKind.COMPOUND:
            return all(self._is_one_dimensional(m) for m in shape.members)
        return False

    def _bodies(self, shape: SmHandle) -> list[SmHandle]:
        """The solid and shell handles inside ``shape``."""
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            return [shape]
        if shape.kind is TopoKind.COMPOUND:
            return [b for m in shape.members for b in self._bodies(m)]
        if shape.kind is TopoKind.FACE:
            return [SmHandle(TopoKind.SHELL, shape.entity.GetBrep())]
        raise ValueError(f"booleans need solids, got a {shape.kind.name}")

    def _tracked_copy(self, shape: SmHandle, record: _Record) -> Any:
        """A working copy of a body whose entities are recorded as what the
        input's entities became."""
        if shape.kind is TopoKind.COMPOUND:
            bodies = self._bodies(shape)
            if len(bodies) != 1:
                raise NotImplementedError(
                    "smlib: operations on compounds of several bodies"
                )
            inner = self._tracked_copy(bodies[0], record)
            record.modified[shape.key()] = record.modified[bodies[0].key()]
            return inner
        copy, forward = self._copy_body(shape.entity)
        for kind in (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE):
            for sub in self.sub_shapes(shape, kind):
                mapped = self._mapped(forward, sub.entity)
                if mapped is not None:
                    record.modified[sub.key()] = [
                        SmHandle(kind, mapped, partner=sub.partner)
                    ]
        record.modified[shape.key()] = [
            SmHandle(shape.kind, copy, partner=shape.partner)
        ]
        return copy

    def _finish_record(self, record: _Record, result: Any) -> None:
        """Drop from the record the copies that did not survive."""
        alive = {
            e.address
            for kind in (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE)
            for e in self._entities(result, kind)
        }
        alive.add(result.address)
        for key, became in list(record.modified.items()):
            survivors = [
                h for h in became if h.entity is None or h.entity.address in alive
            ]
            if survivors:
                record.modified[key] = survivors
            else:
                del record.modified[key]
                record.removed.add(key)

    def boolean(
        self,
        op: BoolOp,
        args: Sequence[SmHandle],
        tools: Sequence[SmHandle],
        *,
        glue: bool = False,
        fuzzy: float | None = None,
    ) -> tuple[SmHandle, Any]:
        ops = {
            BoolOp.FUSE: self.sc.SM_BO_UNION,
            BoolOp.CUT: self.sc.SM_BO_DIFFERENCE,
            BoolOp.COMMON: self.sc.SM_BO_INTERSECTION,
        }
        if op is BoolOp.SPLIT:
            return self._split(args, tools)
        if op not in ops:
            raise NotImplementedError(f"smlib: boolean {op.value}")
        record = _Record()
        if all(self._is_one_dimensional(s) for s in (*args, *tools)):
            if op is not BoolOp.FUSE:
                raise NotImplementedError(f"smlib: {op.value} of edges and wires")
            # the edges are kept as they are (no splitting at crossings)
            edges: dict[Hashable, SmHandle] = {}
            for shape in (*args, *tools):
                for edge in self.sub_shapes(shape, TopoKind.EDGE):
                    edges.setdefault(edge.key(), edge)
                for sub in self._tracked_subshapes(shape):
                    record.modified[sub.key()] = [sub]
            return SmHandle(TopoKind.COMPOUND, members=list(edges.values())), record
        arg_bodies = [
            self._tracked_copy(b, record) for a in args for b in self._bodies(a)
        ]
        tool_bodies = [
            self._tracked_copy(b, record) for t in tools for b in self._bodies(t)
        ]
        if not arg_bodies:
            raise ValueError("a boolean needs at least one argument body")
        result = arg_bodies[0]
        for body in arg_bodies[1:]:
            result = self._merge(result, body, self.sc.SM_BO_UNION)
        for body in tool_bodies:
            result = self._merge(result, body, ops[op])
        self._finish_record(record, result)
        return self._solid_handle(result), record

    def _split(
        self, args: Sequence[SmHandle], tools: Sequence[SmHandle]
    ) -> tuple[SmHandle, Any]:
        """Split bodies by planar tools: each side of a tool's plane is kept
        as the part of the body inside a large box on that side."""
        record = _Record()
        pieces: list[SmHandle] = [
            SmHandle(b.kind, b.entity, partner=b.partner) for a in args for b in self._bodies(a)
        ]
        for tool in tools:
            faces = self.sub_shapes(tool, TopoKind.FACE)
            if len(faces) != 1 or self.planar_surface(faces[0]) is None:
                raise NotImplementedError("smlib: splitting by anything but one plane")
            plane = self.planar_surface(faces[0])
            halves = []
            for side in (1.0, -1.0):
                frame = plane.Position().Ax2()
                if side < 0:
                    frame = gp_Ax2(frame.Location(), frame.Direction().Reversed(), frame.XDirection())
                # a box standing on the plane, 2 * _INFINITE wide, on one side
                corner = frame.Location().Translated(
                    gp_Vec(frame.XDirection()).Scaled(-_INFINITE)
                    + gp_Vec(frame.YDirection()).Scaled(-_INFINITE)
                )
                half_frame = gp_Ax2(corner, frame.Direction(), frame.XDirection())
                halves.append(self.make_box(half_frame, 2 * _INFINITE, 2 * _INFINITE, _INFINITE))
            next_pieces = []
            for piece in pieces:
                for half in halves:
                    try:
                        part, part_record = self.boolean(BoolOp.COMMON, [piece], [half])
                    except RuntimeError:
                        continue
                    if self.is_null(part) or self.volume(part) <= _TOLERANCE:
                        continue
                    for key, became in part_record.modified.items():
                        record.modified.setdefault(key, []).extend(became)
                    next_pieces.append(part)
            pieces = next_pieces
        if len(pieces) == 1:
            return pieces[0], record
        return SmHandle(TopoKind.COMPOUND, members=pieces), record

    def _merge(self, a: Any, b: Any, op: Any) -> Any:
        merge = self.sc.SmMerge(self.context, a, b, 0.0, 0.0, None)
        status, result = merge.ManifoldBoolean(op, None)
        _check(status, "boolean")
        if result is None:
            raise RuntimeError("SMLib boolean produced no body")
        return result

    def record_modified(self, record: Any, shape: SmHandle) -> list[SmHandle]:
        if isinstance(record, _Record):
            return list(record.modified.get(shape.key(), []))
        return []

    def record_generated(self, record: Any, shape: SmHandle) -> list[SmHandle]:
        return []

    def record_removed(self, record: Any, shape: SmHandle) -> bool:
        return isinstance(record, _Record) and shape.key() in record.removed

    def sewn_shape(self, record: Any) -> SmHandle:
        if isinstance(record, _Record) and record.result is not None:
            return record.result
        raise ValueError("not a sewing record")

    def section(self, a: SmHandle, b: SmHandle) -> tuple[SmHandle, Any]:
        raise NotImplementedError("smlib: section")

    def unify_same_domain(self, shape: SmHandle) -> tuple[SmHandle, Any]:
        return shape, _Record()

    # ---- placement ----------------------------------------------------------

    def identity_location(self) -> TopLoc_Location:
        return TopLoc_Location()

    def location_of(self, shape: SmHandle) -> TopLoc_Location:
        return shape.location

    def _bake(
        self,
        shape: SmHandle,
        trsf: gp_Trsf,
        copies: dict[int, tuple[Any, Any]] | None = None,
    ) -> SmHandle:
        """A moved copy of ``shape``: the geometry transformed, the placement
        remembered. ``copies`` maps a body's address to its moved copy and
        forward map, so that the members of one container share one copy."""
        if trsf.Form() == 0 and trsf.TranslationPart().Modulus() == 0:  # gp_Identity
            return shape
        if copies is None:
            copies = {}
        placement = self._placement_of_trsf(trsf)
        new_location = TopLoc_Location(trsf).Multiplied(shape.location)
        if shape.kind in (TopoKind.WIRE, TopoKind.COMPOUND):
            return SmHandle(
                shape.kind,
                members=[self._bake(m, trsf, copies) for m in shape.members],
                flip=shape.flip,
                location=new_location,
                partner=shape.partner,
            )
        if shape.point is not None:
            moved = gp_Pnt(*shape.point).Transformed(trsf)
            return SmHandle(
                TopoKind.VERTEX,
                point=(moved.X(), moved.Y(), moved.Z()),
                location=new_location,
                partner=shape.partner,
            )
        if shape.curve is not None:
            curve = self._copy_curve(shape.curve)
            _check(curve.Transform(placement, None), "curve transform")
            return SmHandle(
                TopoKind.EDGE,
                curve=curve,
                interval=shape.interval,
                flip=shape.flip,
                location=new_location,
                partner=shape.partner,
            )
        brep = self._body(shape)
        if brep.address in copies:
            copy, forward = copies[brep.address]
        else:
            copy, forward = self._copy_body(brep)
            _check(copy.Transform(placement, None, 1), "transform")
            copies[brep.address] = (copy, forward)
        entity = (
            copy
            if shape.kind in (TopoKind.SOLID, TopoKind.SHELL)
            else self._mapped(forward, shape.entity)
        )
        return SmHandle(
            shape.kind,
            entity,
            flip=shape.flip,
            location=new_location,
            partner=shape.partner,
        )

    def face_frame(self, face: SmHandle) -> tuple[Point, Point, Point]:
        view = SmSurfaceView(self, face)
        u_mid, v_mid = view._mid()  # pylint: disable=protected-access
        _, du, _ = view.d1(u_mid, v_mid)
        _, normal = view.normal(u_mid, v_mid)
        return self.center_of_mass(face, TopoKind.FACE), du, normal

    def planar_surface(self, face: SmHandle) -> Any:
        plane = SmSurfaceView(self, face).plane()
        if plane is None:
            return None
        origin, normal, x_dir = plane
        return gp_Pln(gp_Ax3(gp_Pnt(*origin), gp_Dir(*normal), gp_Dir(*x_dir)))

    def move(self, shape: SmHandle, location: TopLoc_Location) -> None:
        moved = self._bake(shape, location.Transformation())
        shape.entity, shape.members, shape.location, shape.point = (
            moved.entity,
            moved.members,
            moved.location,
            moved.point,
        )

    def set_location(self, shape: SmHandle, location: TopLoc_Location) -> None:
        moved = self.located(shape, location)
        shape.entity, shape.members, shape.location, shape.point = (
            moved.entity,
            moved.members,
            moved.location,
            moved.point,
        )

    def located(self, shape: SmHandle, location: TopLoc_Location) -> SmHandle:
        delta = location.Multiplied(shape.location.Inverted())
        return self._bake(shape, delta.Transformation())

    def moved(self, shape: SmHandle, location: TopLoc_Location) -> SmHandle:
        return self._bake(shape, location.Transformation())

    def relocation_between(self, source: SmHandle, target: SmHandle) -> TopLoc_Location:
        return target.location.Multiplied(source.location.Inverted())

    # ---- views --------------------------------------------------------------

    def curve_of(self, shape: SmHandle) -> SmCurveView:
        return SmCurveView(self, shape)

    def surface_of(self, face: SmHandle) -> SmSurfaceView:
        return SmSurfaceView(self, face)

    # ---- construction -------------------------------------------------------

    def _primitive(self, build) -> SmHandle:
        brep = self._new_brep()
        creation = self.sc.SmPrimitiveCreation(brep.GetInfiniteRegion(), 0, 1)
        _check(build(creation), "primitive creation")
        return self._solid_handle(brep)

    def make_box(
        self, frame: Any, length: float, width: float, height: float
    ) -> SmHandle:
        placement = self._placement(frame)
        return self._primitive(lambda c: c.CreateBox(length, width, height, placement))

    def make_cylinder(
        self, frame: Any, radius: float, height: float, angle: float
    ) -> SmHandle:
        placement = self._placement(frame)
        degrees = math.degrees(angle)
        return self._primitive(
            lambda c: c.CreateCone(height, radius, radius, 0.0, degrees, placement)
        )

    def make_cone(
        self,
        frame: Any,
        base_radius: float,
        top_radius: float,
        height: float,
        angle: float,
    ) -> SmHandle:
        placement = self._placement(frame)
        degrees = math.degrees(angle)
        return self._primitive(
            lambda c: c.CreateCone(
                height, base_radius, top_radius, 0.0, degrees, placement
            )
        )

    def make_sphere(
        self, frame: Any, radius: float, angle1: float, angle2: float, angle3: float
    ) -> SmHandle:
        placement = self._placement(frame)
        degrees = math.degrees(angle3)
        return self._primitive(
            lambda c: c.CreateSphere(radius, 0.0, degrees, placement)
        )

    def make_torus(
        self,
        frame: Any,
        major_radius: float,
        minor_radius: float,
        angle1: float,
        angle2: float,
        angle3: float,
    ) -> SmHandle:
        placement = self._placement(frame)
        degrees = math.degrees(angle3)
        return self._primitive(
            lambda c: c.CreateTorus(major_radius, minor_radius, 0.0, degrees, placement)
        )

    make_wedge = _unsupported("make_wedge")

    def _loop_curves(self, wire: SmHandle) -> list[Any]:
        """Copies of the curves of a wire's edges, trimmed to the edges and
        running the wire's way, for the kernel to consume."""
        curves = []
        for edge in wire.members:
            if edge.entity is not None:
                curve = edge.entity.GetCurve()
                interval = edge.entity.GetInterval()
                t0, t1 = interval.GetMin(), interval.GetMax()
            else:
                curve = edge.curve
                assert edge.interval is not None
                t0, t1 = edge.interval
            status, copy = curve.Copy(self.context)
            _check(status, "curve copy")
            domain = self.sc.SmExtent1d(t0, t1)
            _check(copy.Trim(domain, 0, 0), "curve trim")
            if edge.flip:
                reversed_domain = self.sc.SmExtent1d()
                _check(
                    copy.ReverseParameterization(domain, reversed_domain),
                    "curve reverse",
                )
            curves.append(copy)
        return curves

    def extrude(self, shape: SmHandle, direction: Point) -> SmHandle:
        distance = math.sqrt(sum(c * c for c in direction))
        unit = self._vec(tuple(c / distance for c in direction))
        if shape.kind is TopoKind.FACE:
            wires = self.sub_shape_uses(shape, TopoKind.WIRE)
            curves = [c for w in wires for c in self._loop_curves(w)]
            return self._primitive(
                lambda c: c.CreateLinearSweep(curves, unit, distance, 1, 1, 1)
            )
        if shape.kind is TopoKind.WIRE:
            curves = self._loop_curves(shape)
            return self._primitive(
                lambda c: c.CreateLinearSweep(curves, unit, distance, 1, 0, 1)
            )
        if shape.kind is TopoKind.EDGE:
            curves = self._loop_curves(SmHandle(TopoKind.WIRE, members=[shape]))
            sheet = self._primitive(
                lambda c: c.CreateLinearSweep(curves, unit, distance, 1, 0, 1)
            )
            faces = self._entities(sheet.entity, TopoKind.FACE)
            if len(faces) == 1:
                return SmHandle(TopoKind.FACE, faces[0])
            return sheet
        if shape.kind is TopoKind.VERTEX:
            start = self.vertex_point(shape)
            end = tuple(a + b for a, b in zip(start, direction))
            return self.make_line(start, end)  # type: ignore[arg-type]
        raise NotImplementedError(f"smlib: extrude of a {shape.kind.name}")

    def revolve(self, shape: SmHandle, axis: Any, angle: float) -> SmHandle:
        origin = axis.Location()
        direction = axis.Direction()
        base = self._vec((origin.X(), origin.Y(), origin.Z()))
        axis_dir = self._vec((direction.X(), direction.Y(), direction.Z()))
        if shape.kind is TopoKind.FACE:
            curves = [
                c
                for w in self.sub_shape_uses(shape, TopoKind.WIRE)
                for c in self._loop_curves(w)
            ]
        elif shape.kind is TopoKind.WIRE:
            curves = self._loop_curves(shape)
        elif shape.kind is TopoKind.EDGE:
            curves = self._loop_curves(SmHandle(TopoKind.WIRE, members=[shape]))
            sheet = self._primitive(
                lambda c: c.CreateRotationalSweep(
                    curves, base, axis_dir, math.degrees(angle), 1, 0, 1
                )
            )
            faces = self._entities(sheet.entity, TopoKind.FACE)
            return SmHandle(TopoKind.FACE, faces[0]) if len(faces) == 1 else sheet
        else:
            raise NotImplementedError(f"smlib: revolve of a {shape.kind.name}")
        return self._primitive(
            lambda c: c.CreateRotationalSweep(
                curves, base, axis_dir, math.degrees(angle), 1, 1, 1
            )
        )

    def loft(
        self, sections: Sequence[SmHandle], solid: bool, ruled: bool
    ) -> SmHandle:
        counts, curves = [], []
        for section in sections:
            if section.kind is TopoKind.VERTEX:
                raise NotImplementedError("smlib: loft to a vertex")
            wire = (
                section
                if section.kind is TopoKind.WIRE
                else SmHandle(TopoKind.WIRE, members=[section])
            )
            loop = self._loop_curves(wire)
            counts.append(len(loop))
            curves.extend(loop)
        return self._primitive(
            lambda c: c.CreateSkinPrimitive(
                counts, curves, 0.0, 1 if ruled else 3, 1 if solid else 0, [], [], []
            )
        )

    def _as_nurbs(self, curve: Any) -> Any:
        """A plain NURBS rebuilt from a curve's control polygon and knots: the
        sweeps walk the NURBS data directly, which the analytic curves and
        their trimmed copies do not always carry in a form they accept."""
        status, poles, weights = curve.GetControlPolygon([], [])
        _check(status, "control polygon")
        status, knots = curve.GetKnotsAll([])
        _check(status, "knots")
        points = [(p.x, p.y, p.z) for p in poles]
        rational = bool(curve.IsRational()) and len(weights) == len(points)
        return self._nurbs(
            points,
            [float(k) for k in knots],
            int(curve.GetDegree()),
            [float(w) for w in weights] if rational else None,
        )

    def _joined_curve(self, wire: SmHandle) -> Any:
        """One NURBS curve running along a wire."""
        curves = [self._as_nurbs(c) for c in self._loop_curves(wire)]
        if len(curves) == 1:
            return curves[0]
        status, joined = self.sc.SmBSplineCurve.CreateByJoining(self.context, curves, None)
        _check(status, "curve joining")
        return joined

    def sweep(
        self,
        profile: SmHandle,
        path: SmHandle,
        *,
        is_frenet: bool,
        transition: Any,
        make_solid: bool,
        binormal: Point | None = None,
        binormal_origin: Point | None = None,
        auxiliary_spine: SmHandle | None = None,
        curvilinear_equivalence: bool = True,
    ) -> SmHandle:
        if auxiliary_spine is not None or binormal is not None:
            raise NotImplementedError("smlib: sweeps with an auxiliary spine or binormal")
        path_wire = path if path.kind is TopoKind.WIRE else SmHandle(TopoKind.WIRE, members=[path])
        if profile.kind is TopoKind.FACE:
            wires = self.sub_shape_uses(profile, TopoKind.WIRE)
            curves = [self._as_nurbs(c) for w in wires for c in self._loop_curves(w)]
            closed = True
        else:
            profile_wire = (
                profile if profile.kind is TopoKind.WIRE else SmHandle(TopoKind.WIRE, members=[profile])
            )
            curves = [self._as_nurbs(c) for c in self._loop_curves(profile_wire)]
            closed = self.is_closed(profile_wire)
        spine = self._joined_curve(path_wire)
        options = self.sc.SmSweepOptions()
        options.bCapEndsArg = 1 if (make_solid and closed) else 0
        options.bTranslationalSweep = 0
        return self._primitive(
            lambda c: c.CreateCurveSweep(
                curves, spine, None, None, _TOLERANCE, options, [], [], [], 1
            )
        )

    sweep_multi = _unsupported("sweep_multi")

    def fillet(
        self, shape: SmHandle, edges: Sequence[SmHandle], radius: float
    ) -> tuple[SmHandle, Any]:
        record = _Record()
        body = self._tracked_copy(shape, record)
        working_edges = [record.modified[e.key()][0].entity for e in edges]
        executive = self.sc.SmFilletExecutive(self.context, body)
        executive.SetDoGlobalMerge(1)
        _check(
            executive.SetFilletParameters(
                body,
                working_edges,
                self.sc.SM_FSG_CIRCULAR,
                self.sc.SM_FS_CONST_RADIUS,
                radius,
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
                1.0,
            ),
            "fillet parameters",
        )
        _check(executive.DoFillet(), "fillet")
        self._finish_record(record, body)
        return self._solid_handle(body), record

    def chamfer(
        self, shape: SmHandle, specs: Sequence[tuple[SmHandle, SmHandle, float, float]]
    ) -> tuple[SmHandle, Any]:
        record = _Record()
        body = self._tracked_copy(shape, record)
        working_edges = [record.modified[e.key()][0].entity for e, _, _, _ in specs]
        distances = {d for _, _, d1, d2 in specs for d in (d1, d2)}
        if len(distances) != 1:
            raise NotImplementedError("smlib: asymmetric chamfers")
        executive = self.sc.SmFilletExecutive(self.context, body)
        executive.SetDoGlobalMerge(1)
        _check(
            executive.SetFilletParameters(
                body,
                working_edges,
                self.sc.SM_FSG_LINEAR,
                self.sc.SM_FS_CONST_DIST,
                distances.pop(),
                None,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
                1.0,
            ),
            "chamfer parameters",
        )
        _check(executive.DoFillet(), "chamfer")
        self._finish_record(record, body)
        return self._solid_handle(body), record

    def thick_solid(
        self,
        shape: SmHandle,
        openings: Sequence[SmHandle],
        thickness: float,
        tolerance: float,
        join: JoinKind,
        remove_internal_edges: bool = False,
    ) -> SmHandle:
        record = _Record()
        body = self._tracked_copy(shape, record)
        faces = [record.modified[f.key()][0].entity for f in openings]
        status, result = self.sc.SmOffsetExecutive.ShellBrep(
            self.context, body, thickness, 1, 1, 1, faces
        )
        _check(status, "shell")
        return self._solid_handle(result)

    def offset_shape(
        self,
        shape: SmHandle,
        amount: float,
        tolerance: float,
        *,
        join: JoinKind = JoinKind.ARC,
        intersection: bool = False,
        thicken: bool = False,
        remove_internal_edges: bool = False,
    ) -> SmHandle:
        if thicken:
            # a planar face is thickened by extruding it along its normal;
            # other surfaces would need offset faces with side walls
            if shape.kind is TopoKind.FACE and self.planar_surface(shape) is not None:
                _, normal = SmSurfaceView(self, shape).normal(
                    *SmSurfaceView(self, shape)._mid()  # pylint: disable=protected-access
                )
                return self.extrude(shape, tuple(amount * c for c in normal))  # type: ignore[arg-type]
            raise NotImplementedError("smlib: thickening of curved faces")
        record = _Record()
        body = self._tracked_copy(shape, record)
        status, result = self.sc.SmOffsetExecutive.OffsetBrep(
            self.context, body, amount, 1, 1
        )
        _check(status, "offset")
        return self._solid_handle(result)

    def make_solid(self, shells: Sequence[SmHandle]) -> SmHandle:
        if len(shells) != 1:
            raise NotImplementedError("smlib: solids with voids")
        return SmHandle(TopoKind.SOLID, shells[0].entity, partner=shells[0].partner)

    def make_vertex(self, point: Point) -> SmHandle:
        return SmHandle(TopoKind.VERTEX, point=tuple(float(c) for c in point))  # type: ignore[arg-type]

    def vertex_point(self, vertex: SmHandle) -> Point:
        if vertex.point is not None:
            return vertex.point
        return self._pt(vertex.entity.GetPoint())

    def _edge_of_curve(self, curve: Any) -> SmHandle:
        """A free edge: one edge wire body around a curve."""
        brep = self._new_brep()
        raise NotImplementedError("smlib: free edges are not supported yet")

    def _free_edge(
        self, curve: Any, t0: float, t1: float, flip: bool = False
    ) -> SmHandle:
        return SmHandle(TopoKind.EDGE, curve=curve, interval=(t0, t1), flip=flip)

    def _edge_on(self, curve: Any, flip: bool = False) -> SmHandle:
        """A free edge over the whole of a curve."""
        domain = curve.GetNaturalInterval()
        return self._free_edge(curve, domain.GetMin(), domain.GetMax(), flip)

    def make_line(self, start: Point, end: Point) -> SmHandle:
        line = self.sc.SmLine(self._vec(start), self._vec(end), 3, self.context)
        return self._free_edge(line, 0.0, 1.0)

    def make_infinite_line(self, origin: Point, direction: Point) -> SmHandle:
        length = math.sqrt(sum(c * c for c in direction))
        unit = tuple(c / length for c in direction)
        line = self.sc.SmLine(
            self._vec(origin),
            self._vec(unit),
            self.sc.SmExtent1d(-_INFINITE, _INFINITE),
            1.0,
            3,
            self.context,
            None,
            0,
        )
        return self._edge_on(line)

    @staticmethod
    def _arc_domain(start: float, end: float, sense: bool) -> tuple[float, float]:
        """The degree domain of an arc given in radians, counter-clockwise
        in the kernel; a clockwise arc is the flipped counter-clockwise one."""
        a1, a2 = (
            (math.degrees(start), math.degrees(end))
            if sense
            else (math.degrees(end), math.degrees(start))
        )
        while a2 <= a1:
            a2 += 360.0
        return a1, a2

    def _rotated_frame(self, frame: Any, rotation: float) -> tuple[Any, Any, Any]:
        """Origin, x and y directions of ``frame`` turned about its axis."""
        origin = frame.Location()
        x_dir, y_dir = frame.XDirection(), frame.YDirection()
        if rotation:
            axis = gp_Ax1(origin, frame.Direction())
            x_dir, y_dir = x_dir.Rotated(axis, rotation), y_dir.Rotated(axis, rotation)
        return (
            self._vec((origin.X(), origin.Y(), origin.Z())),
            self._vec((x_dir.X(), x_dir.Y(), x_dir.Z())),
            self._vec((y_dir.X(), y_dir.Y(), y_dir.Z())),
        )

    def _circle(
        self, frame: Any, radius: float, start_deg: float, end_deg: float
    ) -> Any:
        origin = frame.Location()
        x_dir = frame.XDirection()
        y_dir = frame.YDirection()
        return self.sc.SmCircle(
            self._vec((origin.X(), origin.Y(), origin.Z())),
            self._vec((x_dir.X(), x_dir.Y(), x_dir.Z())),
            self._vec((y_dir.X(), y_dir.Y(), y_dir.Z())),
            self.sc.SmExtent1d(start_deg, end_deg),
            radius,
            3,
            self.context,
            None,
            0,
        )

    def make_circle(self, frame: Any, radius: float) -> SmHandle:
        return self._edge_on(self._circle(frame, radius, 0.0, 360.0))

    def make_circle_arc(
        self, frame: Any, radius: float, start: float, end: float, sense: bool
    ) -> SmHandle:
        # the circle's parameter is the angle in degrees; an arc always runs
        # counter-clockwise in the kernel, so a clockwise one is flipped
        a1, a2 = self._arc_domain(start, end, sense)
        return self._edge_on(self._circle(frame, radius, a1, a2), flip=not sense)

    def _ellipse(
        self, frame: Any, major: float, minor: float, rotation: float, a1: float, a2: float
    ) -> Any:
        origin, x_dir, y_dir = self._rotated_frame(frame, rotation)
        return self.sc.SmEllipse(
            origin,
            x_dir,
            y_dir,
            self.sc.SmExtent1d(a1, a2),
            major,
            minor,
            3,
            self.context,
            None,
            0,
        )

    def make_ellipse(
        self, frame: Any, major: float, minor: float, rotation: float = 0.0
    ) -> SmHandle:
        return self._edge_on(self._ellipse(frame, major, minor, rotation, 0.0, 360.0))

    def make_ellipse_arc(
        self,
        frame: Any,
        major: float,
        minor: float,
        start: float,
        end: float,
        sense: bool,
        rotation: float = 0.0,
    ) -> SmHandle:
        a1, a2 = self._arc_domain(start, end, sense)
        return self._edge_on(
            self._ellipse(frame, major, minor, rotation, a1, a2), flip=not sense
        )

    def make_parabola_arc(
        self, frame: Any, focal_length: float, start: float, end: float, sense: bool
    ) -> SmHandle:
        # SMLib's parabola is (f t^2, 2 f t); OCCT's is (t^2 / 4f, t)
        t0, t1 = start / (2 * focal_length), end / (2 * focal_length)
        if t0 > t1:
            t0, t1, sense = t1, t0, not sense
        origin, x_dir, y_dir = self._rotated_frame(frame, 0.0)
        parabola = self.sc.SmParabola(
            origin, x_dir, y_dir, self.sc.SmExtent1d(t0, t1), focal_length, 3, self.context
        )
        return self._edge_on(parabola, flip=not sense)

    def _hyperbola(
        self, frame: Any, major: float, minor: float, rotation: float, t0: float, t1: float
    ) -> Any:
        origin, x_dir, y_dir = self._rotated_frame(frame, rotation)
        return self.sc.SmHyperbola(
            origin, x_dir, y_dir, self.sc.SmExtent1d(t0, t1), major, minor, 3, self.context
        )

    def make_hyperbola(
        self, frame: Any, major: float, minor: float, rotation: float = 0.0
    ) -> SmHandle:
        return self._edge_on(
            self._hyperbola(frame, major, minor, rotation, -_HYPERBOLA_SPAN, _HYPERBOLA_SPAN)
        )

    def make_hyperbola_arc(
        self,
        frame: Any,
        major: float,
        minor: float,
        start: float,
        end: float,
        sense: bool,
        rotation: float = 0.0,
    ) -> SmHandle:
        if start > end:
            start, end, sense = end, start, not sense
        return self._edge_on(
            self._hyperbola(frame, major, minor, rotation, start, end), flip=not sense
        )

    def make_tangent_arc(self, start: Point, tangent: Point, end: Point) -> SmHandle:
        chord = [b - a for a, b in zip(start, end)]
        unit_t = _unit(tangent)
        normal = _unit(_cross(unit_t, chord))
        towards = _unit(_cross(normal, unit_t))  # from the start towards the centre
        chord_sq = sum(c * c for c in chord)
        radius = chord_sq / (2 * _dot(chord, towards))
        centre = tuple(a + radius * b for a, b in zip(start, towards))
        x_dir = tuple(-c for c in towards)
        y_dir = _cross(normal, x_dir)
        relative = [b - a for a, b in zip(centre, end)]
        angle = math.degrees(math.atan2(_dot(relative, y_dir), _dot(relative, x_dir)))
        if angle <= 0:
            angle += 360.0
        circle = self.sc.SmCircle(
            self._vec(centre),
            self._vec(x_dir),
            self._vec(y_dir),
            self.sc.SmExtent1d(0.0, angle),
            radius,
            3,
            self.context,
            None,
            0,
        )
        return self._edge_on(circle)

    def make_three_point_arc(self, p1: Point, p2: Point, p3: Point) -> SmHandle:
        arc = self.sc.SmCircle(
            self._vec(p1), self._vec(p2), self._vec(p3), 3, 0, self.context
        )
        domain = arc.GetNaturalInterval()
        return self._free_edge(arc, domain.GetMin(), domain.GetMax())

    def _nurbs(
        self,
        poles: Sequence[Point],
        knots: Sequence[float],
        degree: int,
        weights: Sequence[float] | None,
    ) -> Any:
        """A NURBS curve from its full knot vector and poles."""
        curve = self.sc.SmBSplineCurve(3)
        if weights:
            form, stride = self.sc.SM_CP_EUCLIDIAN_RATIONAL, 4
            flat = [c for p, w in zip(poles, weights) for c in (*p, w)]
        else:
            form, stride = self.sc.SM_CP_NON_RATIONAL, 3
            flat = [c for p in poles for c in p]
        _check(
            curve.SetExpert(
                3,
                degree,
                self.sc.SM_CF_UNSPECIFIED,
                self.sc.SM_EK_CLAMPPED,
                len(knots),
                [float(k) for k in knots],
                form,
                stride,
                [float(c) for c in flat],
            ),
            "spline construction",
        )
        return curve

    def make_bezier(
        self, poles: Sequence[Point], weights: Sequence[float] | None = None
    ) -> SmHandle:
        count = len(poles)
        curve = self._nurbs(poles, [0.0] * count + [1.0] * count, count - 1, weights)
        self._bezier.add(curve.address)
        return self._edge_on(curve)

    def make_bspline(
        self,
        poles: Sequence[Point],
        knots: Sequence[float],
        multiplicities: Sequence[int],
        degree: int,
        periodic: bool,
        weights: Sequence[float] | None = None,
    ) -> SmHandle:
        if periodic:
            raise NotImplementedError("smlib: periodic b-splines")
        flat = [k for k, m in zip(knots, multiplicities) for _ in range(m)]
        return self._edge_on(self._nurbs(poles, flat, degree, weights))

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
    ) -> SmHandle:
        if point_tangents is not None and any(t is not None for t in point_tangents):
            raise NotImplementedError("smlib: interpolation with tangents at points")
        start_tangent = end_tangent = None
        if end_tangents is not None:
            start_tangent = self._vec(end_tangents[0])
            end_tangent = self._vec(end_tangents[1])
        status, curve = self.sc.SmBSplineCurve.InterpolatePoints(
            self.context,
            [self._vec(p) for p in points],
            [float(v) for v in parameters] if parameters is not None else None,
            min(3, len(points) - 1),
            start_tangent,
            end_tangent,
            1 if periodic else 0,
            self.sc.SM_IT_CHORDLENGTH,
        )
        _check(status, "interpolation")
        return self._edge_on(curve)

    def make_spline_fit(
        self,
        points: Sequence[Point],
        tolerance: float,
        min_degree: int,
        max_degree: int,
        smoothing: tuple[float, float, float] | None = None,
    ) -> SmHandle:
        status, curve = self.sc.SmBSplineCurve.ApproximatePoints(
            self.context,
            [self._vec(p) for p in points],
            max(1, min(3, max_degree)),
            None,
            None,
            0,
            tolerance,
        )
        _check(status, "approximation")
        return self._edge_on(curve)

    make_helix = _unsupported("make_helix")

    def make_wire(
        self, shapes: Sequence[SmHandle]
    ) -> tuple[SmHandle | None, WireStatus]:
        edges = [
            e for s in shapes for e in (s.members if s.kind is TopoKind.WIRE else [s])
        ]
        if not edges:
            return None, WireStatus.EMPTY
        return SmHandle(TopoKind.WIRE, members=edges), WireStatus.OK

    def make_wire_placed(
        self, edges: Sequence[SmHandle]
    ) -> tuple[SmHandle, list[SmHandle]]:
        wire, _ = self.make_wire(edges)
        assert wire is not None
        return wire, list(wire.members)

    def make_planar_face(self, outer: SmHandle, inners: Sequence[SmHandle]) -> SmHandle:
        curves = self._loop_curves(outer)
        for inner in inners:
            curves.extend(self._loop_curves(inner))
        brep = self._new_brep()
        status, faces = brep.CreatePlanarFacesWith3DCurves(
            brep.GetInfiniteRegion(), curves, _TOLERANCE, []
        )
        _check(status, "planar face")
        if len(faces) != 1:
            raise ValueError(f"Cannot build face(s): the wires gave {len(faces)} faces")
        return SmHandle(TopoKind.FACE, faces[0])

    add_holes = _unsupported("add_holes")

    def face_of_plane(
        self, plane: Any, bounds: tuple[float, float, float, float] | None = None
    ) -> SmHandle:
        if bounds is None:
            # SMLib has no unbounded faces; a very large one stands in
            bounds = (-_INFINITE, _INFINITE, -_INFINITE, _INFINITE)
        u0, u1, v0, v1 = bounds
        origin = plane.Location()
        x_axis = plane.XAxis().Direction()
        y_axis = plane.YAxis().Direction()

        def corner(u: float, v: float) -> Point:
            return (
                origin.X() + u * x_axis.X() + v * y_axis.X(),
                origin.Y() + u * x_axis.Y() + v * y_axis.Y(),
                origin.Z() + u * x_axis.Z() + v * y_axis.Z(),
            )

        corners = [
            corner(u0, v0),
            corner(u1, v0),
            corner(u1, v1),
            corner(u0, v1),
            corner(u0, v0),
        ]
        return self.make_planar_face(self.make_polygon(corners), [])

    def make_shell(self, faces: Sequence[SmHandle]) -> SmHandle:
        if len(faces) == 1:
            return SmHandle(TopoKind.SHELL, faces[0].entity.GetBrep())
        raise NotImplementedError("smlib: shells of several faces")

    def sew(self, faces: Sequence[SmHandle]) -> tuple[SmHandle, Any]:
        record = _Record()
        target = self._new_brep()
        for face in faces:
            before = {f.address for f in self._entities(target, TopoKind.FACE)}
            _check(
                face.entity.GetBrep().CopyFaces([face.entity], target, None, 0, None),
                "face copy",
            )
            record.modified[face.key()] = [
                SmHandle(TopoKind.FACE, f)
                for f in self._entities(target, TopoKind.FACE)
                if f.address not in before
            ]
        stitcher = self.sc.SmStitch(self.sc.SmStitchCallback(), _TOLERANCE, 1, 1)
        _check(stitcher.StitchIntoShell(target, 0, 1.0), "stitch")
        self._finish_record(record, target)
        new_faces = self._entities(target, TopoKind.FACE)
        if len(new_faces) == 1:
            record.result = SmHandle(TopoKind.FACE, new_faces[0])
        else:
            record.result = SmHandle(TopoKind.SHELL, target)
        return record.result, record

    def make_compound(self, shapes: Sequence[SmHandle]) -> SmHandle:
        return SmHandle(TopoKind.COMPOUND, members=list(shapes))

    def remove_from_compound(self, compound: SmHandle, shape: SmHandle) -> None:
        compound.members = tuple(
            m for m in compound.members if not self.is_same(m, shape)
        )

    def transformed(
        self, shape: SmHandle, transform: gp_Trsf, copy: bool = True
    ) -> tuple[SmHandle, Any]:
        record = _Record()
        moved = self._bake(shape, transform)
        return moved, record

    transformed_general = _unsupported("transformed_general")

    # ---- distance and classification ----------------------------------------

    def _atoms(self, shape: SmHandle) -> list[tuple[str, Any]]:
        """What the solvers can work on: bodies, curves with intervals,
        surfaces with domains, and points."""
        if shape.kind in (TopoKind.WIRE, TopoKind.COMPOUND):
            return [a for m in shape.members for a in self._atoms(m)]
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            return [("body", shape.entity)]
        if shape.kind is TopoKind.FACE:
            brep = shape.entity.GetBrep()
            if len(self._entities(brep, TopoKind.FACE)) == 1:
                return [("body", brep)]
            return [("face", shape.entity)]
        if shape.kind is TopoKind.EDGE:
            view = SmCurveView(self, shape)
            return [("curve", (view.curve, self.sc.SmExtent1d(view.first, view.last)))]
        if shape.kind is TopoKind.VERTEX:
            return [("point", self.vertex_point(shape))]
        return []

    def _solutions(
        self, solutions: Any, count: int = 2
    ) -> list[tuple[float, Point, Point]]:
        """(distance, point on the first object, point on the second) per
        solution; a point solve has one object only and repeats its point."""
        found = []
        for i in range(solutions.GetSize()):
            solution = solutions.GetAt(i)
            points = []
            for j in range(count):
                located, params = self.sc.SmVector3d(), self.sc.SmVector3d()
                _check(solution.GetPoint(j, located, params, 0), "solution point")
                points.append(self._pt(located))
            found.append(
                (float(solution.m_vStart.m_dSolutionValue), points[0], points[-1])
            )
        return found

    def _solve_pair(self, a: tuple[str, Any], b: tuple[str, Any]) -> tuple[float, Point, Point]:
        """Least distance between two atoms and the points that realise it."""
        order = {"point": 0, "curve": 1, "face": 2, "body": 3}
        if order[a[0]] > order[b[0]]:
            distance, pb, pa = self._solve_pair(b, a)
            return distance, pa, pb
        sc = self.sc
        kinds = (a[0], b[0])
        solutions = sc.SmSolutionArray()
        if kinds == ("point", "point"):
            return math.dist(a[1], b[1]), a[1], b[1]
        if kinds[0] == "point":
            point = self._vec(a[1])
            if kinds[1] == "curve":
                curve, interval = b[1]
                _check(
                    curve.GlobalPointSolve(
                        interval, sc.SM_SO_MINIMIZE, point, _TOLERANCE, None, None,
                        sc.SM_SR_SINGLE, solutions,
                    ),
                    "point to curve",
                )
            elif kinds[1] == "face":
                _check(
                    b[1].GetSurface().GlobalPointSolve(
                        b[1].GetUVDomain(), sc.SM_SO_MINIMIZE, point,
                        sc.SmZoneTol3d(_TOLERANCE), None, sc.SM_SR_SINGLE, solutions,
                    ),
                    "point to face",
                )
            else:
                _check(
                    sc.SmTopologySolver.BrepPointSolve(
                        b[1], point, sc.SM_SO_MINIMIZE, sc.SM_SR_SINGLE, _TOLERANCE,
                        _HUGE, None, solutions,
                    ),
                    "point to body",
                )
            found = self._solutions(solutions, 1)
            if not found:
                raise RuntimeError("SMLib found no nearest point")
            distance, on_b, _ = min(found)
            return distance, a[1], on_b
        if kinds == ("curve", "curve"):
            (c1, i1), (c2, i2) = a[1], b[1]
            _check(
                c1.GlobalCurveSolve(
                    i1, c2, i2, sc.SM_SO_MINIMIZE, _TOLERANCE, None, None,
                    sc.SM_SR_SINGLE, solutions,
                ),
                "curve to curve",
            )
            found = self._solutions(solutions)
            if not found:
                raise RuntimeError("SMLib found no nearest points")
            return min(found)
        if kinds == ("curve", "body"):
            curve, interval = a[1]
            _check(
                sc.SmTopologySolver.BrepCurveSolve(
                    b[1], curve, interval, sc.SM_SO_MINIMIZE, sc.SM_SR_SINGLE,
                    _TOLERANCE, _HUGE, None, solutions,
                ),
                "curve to body",
            )
            found = self._solutions(solutions)
            if not found:
                raise RuntimeError("SMLib found no nearest points")
            return min(found)
        if kinds == ("body", "body"):
            _check(
                sc.SmTopologySolver.BrepBrepSolve(
                    a[1], b[1], sc.SM_SO_MINIMIZE, sc.SM_SR_SINGLE, _TOLERANCE, _HUGE,
                    None, solutions,
                ),
                "body to body",
            )
            found = self._solutions(solutions)
            if not found:
                raise RuntimeError("SMLib found no nearest points")
            return min(found)
        raise NotImplementedError(f"smlib: distance between a {a[0]} and a {b[0]}")

    def closest_points(self, a: SmHandle, b: SmHandle) -> tuple[float, Point, Point]:
        best: tuple[float, Point, Point] | None = None
        for atom_a in self._atoms(a):
            for atom_b in self._atoms(b):
                candidate = self._solve_pair(atom_a, atom_b)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if best is None:
            raise ValueError("nothing to measure the distance between")
        return best

    def distance(self, a: SmHandle, b: SmHandle) -> float:
        return self.closest_points(a, b)[0]

    def extrema(
        self, a: SmHandle, b: SmHandle, deflection: float | None = None
    ) -> tuple[float, list[tuple[Point, Point]]]:
        distance, on_a, on_b = self.closest_points(a, b)
        return distance, [(on_a, on_b)]

    def point_in_solid(self, solid: SmHandle, point: Point, tolerance: float) -> bool:
        zone = self.sc.SmZoneTol3d(tolerance)
        classification = self.sc.SmPointClassification(zone, None)
        _check(
            solid.entity.Point3DClassify(self._vec(point), zone, 1, classification),
            "point classification",
        )
        if classification.GetPointClass() != self.sc.SM_PC_REGION:
            return True  # on a face, edge or vertex
        region = classification.GetObject()
        return region is not None and region != solid.entity.GetInfiniteRegion()

    point_on_face = _unsupported("point_on_face")
    uv_in_face = _unsupported("uv_in_face")

    # ---- the rest of the facade -----------------------------------------------

    def _breps_of(self, shape: SmHandle) -> list[Any]:
        if shape.kind in (TopoKind.WIRE, TopoKind.COMPOUND):
            found: dict[int, Any] = {}
            for member in shape.members:
                for brep in self._breps_of(member):
                    found.setdefault(brep.address, brep)
            return list(found.values())
        if shape.kind in (TopoKind.SOLID, TopoKind.SHELL):
            return [shape.entity]
        if shape.entity is not None:
            return [shape.entity.GetBrep()]
        return []

    def mesh(self, shape: SmHandle, tolerance: float, angular_tolerance: float) -> None:
        for brep in self._breps_of(shape):
            known = self._meshes.get(brep.address)
            if known is not None and known[1] <= tolerance:
                continue
            degrees = math.degrees(angular_tolerance)
            result = brep.ConvertToPolyBrep(
                tolerance, degrees, degrees, 0.0, 0.0, 0, 0, 0, 0, None, None
            )
            _check(result, "tessellation")
            self._meshes[brep.address] = (result[1], tolerance)

    def face_triangulation(
        self, face: SmHandle
    ) -> tuple[list[Point], list[tuple[int, int, int]]] | None:
        brep = face.entity.GetBrep()
        known = self._meshes.get(brep.address)
        if known is None:
            return None
        poly = known[0]
        points: list[Point] = []
        index: dict[int, int] = {}
        triangles: list[tuple[int, int, int]] = []
        for poly_face in poly.GetPolyFaces([], face.entity):
            corners = []
            for vertex in poly_face.GetPolyVertices([]):
                if vertex.address not in index:
                    index[vertex.address] = len(points)
                    points.append(self._pt(vertex.GetPoint()))
                corners.append(index[vertex.address])
            for i in range(1, len(corners) - 1):  # a fan for anything beyond a triangle
                triangles.append((corners[0], corners[i], corners[i + 1]))
        if not triangles:
            return points, triangles
        # wind the triangles the way the face's normal points
        p0, p1, p2 = (points[i] for i in triangles[0])
        winding = _cross([b - a for a, b in zip(p0, p1)], [b - a for a, b in zip(p0, p2)])
        centroid = tuple((a + b + c) / 3 for a, b, c in zip(p0, p1, p2))
        view = SmSurfaceView(self, face)
        try:
            _, normal = view.normal(*view.closest_uv(centroid))
        except (RuntimeError, NotImplementedError):
            _, normal = view.normal(*view._mid())  # pylint: disable=protected-access
        if _dot(winding, normal) < 0:
            triangles = [(a, c, b) for a, b, c in triangles]
        return points, triangles

    # ---- edges from edges ---------------------------------------------------

    def edge_range(self, edge: SmHandle) -> tuple[float, float]:
        view = SmCurveView(self, edge)
        return view.first, view.last

    def vertex_param_on_edge(self, vertex: SmHandle, edge: SmHandle) -> float:
        view = SmCurveView(self, edge)
        point = self.vertex_point(vertex)
        start, end = view.value(view.first), view.value(view.last)
        return view.first if math.dist(point, start) <= math.dist(point, end) else view.last

    def reversed_geometry(self, edge: SmHandle) -> SmHandle:
        view = SmCurveView(self, edge)
        curve = self._copy_curve(view.curve)
        old = self.sc.SmExtent1d(view.first, view.last)
        new = self.sc.SmExtent1d()
        _check(curve.ReverseParameterization(old, new), "curve reverse")
        return self._free_edge(curve, new.GetMin(), new.GetMax())

    def trim_edge(self, edge: SmHandle, start: float, end: float) -> SmHandle:
        view = SmCurveView(self, edge)
        flip = False
        if start > end:
            start, end, flip = end, start, True
        return self._free_edge(self._copy_curve(view.curve), start, end, flip)

    def trim_edge_by_length(self, edge: SmHandle, start: float, length: float) -> SmHandle:
        view = SmCurveView(self, edge)
        end = view.param_at_length(length, start)
        return self.trim_edge(edge, start, end)
    approximate_edge = _unsupported("approximate_edge")
    wire_as_bspline_edge = _unsupported("wire_as_bspline_edge")

    def wire_edges(self, wire: SmHandle) -> list[SmHandle]:
        return list(wire.members)

    build_curves_3d = _unsupported("build_curves_3d")
    find_plane = _unsupported("find_plane")
    continuity_between = _unsupported("continuity_between")
    planar_intersections = _unsupported("planar_intersections")
    def nearest_edge(self, shape: SmHandle, point: Point) -> tuple[SmHandle, float] | None:
        best: tuple[float, SmHandle, float] | None = None
        for edge in self.sub_shapes(shape, TopoKind.EDGE):
            param, distance = SmCurveView(self, edge).drop_point(point)
            if best is None or distance < best[0]:
                best = (distance, edge, param)
        if best is None:
            return None
        return best[1], best[2]
    edge_uv_on_face = _unsupported("edge_uv_on_face")
    vertex_uv_on_face = _unsupported("vertex_uv_on_face")
    edge_uv_ends_on_face = _unsupported("edge_uv_ends_on_face")
    def frame_at(
        self, shape: SmHandle, param: float, corrected: bool
    ) -> tuple[Point, Point, Point, Point]:
        view = self.curve_of(shape)
        point, first, second = view._derivatives(param, 2)  # pylint: disable=protected-access
        tangent = _unit(first)
        along = _dot(second, tangent)
        normal = [c - along * t for c, t in zip(second, tangent)]
        if math.sqrt(_dot(normal, normal)) < 1e-9:
            # a straight stretch: any perpendicular will do, chosen stably
            helper = (0.0, 0.0, 1.0) if abs(tangent[2]) < 0.9 else (1.0, 0.0, 0.0)
            normal = list(_cross(helper, tangent))
        normal_unit = _unit(normal)
        return point, tangent, normal_unit, _cross(tangent, normal_unit)

    def make_polygon(self, points: Sequence[Point]) -> SmHandle:
        edges = [self.make_line(a, b) for a, b in zip(points, points[1:])]
        return SmHandle(TopoKind.WIRE, members=edges)

    def connect_edges_to_wires(
        self, edges: Sequence[SmHandle], tolerance: float
    ) -> list[SmHandle]:
        """Chain edges whose ends meet within ``tolerance`` into wires, each
        edge flipped as needed to run the wire's way."""

        def ends(edge: SmHandle) -> tuple[Point, Point]:
            start, end = (self.vertex_point(v) for v in self.edge_vertices(edge))
            return (end, start) if edge.flip else (start, end)

        def near(a: Point, b: Point) -> bool:
            return all(abs(x - y) <= tolerance for x, y in zip(a, b))

        remaining = list(edges)
        wires: list[SmHandle] = []
        while remaining:
            chain = [remaining.pop(0)]
            grown = True
            while grown and remaining:
                grown = False
                tail = ends(chain[-1])[1]
                head = ends(chain[0])[0]
                for candidate in list(remaining):
                    start, end = ends(candidate)
                    if near(start, tail):
                        chain.append(candidate)
                    elif near(end, tail):
                        chain.append(candidate.with_flip(not candidate.flip))
                    elif near(end, head):
                        chain.insert(0, candidate)
                    elif near(start, head):
                        chain.insert(0, candidate.with_flip(not candidate.flip))
                    else:
                        continue
                    remaining.remove(candidate)
                    grown = True
                    break
            wires.append(SmHandle(TopoKind.WIRE, members=chain))
        return wires

    def outer_wire(self, face: SmHandle) -> SmHandle:
        return self.sub_shape_uses(face, TopoKind.WIRE)[0]

    fillet_2d = _unsupported("fillet_2d")
    chamfer_2d = _unsupported("chamfer_2d")
    fix_small_edges = _unsupported("fix_small_edges")
    project = _unsupported("project")
    offset_2d = _unsupported("offset_2d")
    offset_face = _unsupported("offset_face")
    make_bezier_surface = _unsupported("make_bezier_surface")
    make_spline_surface_fit = _unsupported("make_spline_surface_fit")
    filling = _unsupported("filling")
    draft_prism = _unsupported("draft_prism")
    extrude_generated = _unsupported("extrude_generated")
    draft = _unsupported("draft")
    split_by_edges = _unsupported("split_by_edges")
    line_intersections = _unsupported("line_intersections")
    solid_from_shell = _unsupported("solid_from_shell")
    remove_wires = _unsupported("remove_wires")
    is_seam = _unsupported("is_seam")
    to_splines = _unsupported("to_splines")
    def edge_in_plane(self, edge: SmHandle, normal: Point, tolerance: float) -> bool:
        # the geometry is baked, so the normal given against the edge's
        # remembered placement is brought into the world
        world = gp_Dir(*normal).Transformed(edge.location.Transformation())
        axis = (world.X(), world.Y(), world.Z())
        points = SmCurveView(self, edge).positions_by_deflection(tolerance)
        origin = points[0]
        return all(
            abs(_dot([b - a for a, b in zip(origin, p)], axis)) <= tolerance
            for p in points
        )
    surface_axis = _unsupported("surface_axis")
    half_space = _unsupported("half_space")
    ruled_surface = _unsupported("ruled_surface")
