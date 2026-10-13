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
    gp_XYZ,
)
from OCP.Standard import Standard_Failure
from OCP.TopLoc import TopLoc_Location

from build123d.kernel.base import (
    AncestorMap,
    BoolOp,
    CurveView,
    DraftFailure,
    JoinKind,
    Kernel,
    Point,
    SurfaceView,
    TopoKind,
    WireStatus,
)

SM_SUCCESS = 1000
_TOLERANCE = 1e-6
_MASS_ACCURACY = 1e-10  # the integrator's accuracy; the areas and volumes are compared to 7 places
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


def _translation(vector: Sequence[float]) -> gp_Trsf:
    trsf = gp_Trsf()
    trsf.SetTranslation(gp_Vec(*vector))
    return trsf


def _restore_locations(moved: SmHandle, original: SmHandle) -> None:
    """Give a moved copy (and its members, one to one) the placements of the original."""
    moved.location = original.location
    if len(moved.members) == len(original.members):
        for new_member, old_member in zip(moved.members, original.members):
            _restore_locations(new_member, old_member)


def _point_key(point: Sequence[float], digits: int = 9) -> tuple[Any, ...]:
    """Identity of a free vertex: its position, rounded well below the kernel
    tolerance so that the ends of meeting edges agree."""
    return ("at",) + tuple(round(c, digits) + 0.0 for c in point)


def _perpendicular(direction: Sequence[float]) -> Point:
    """A unit vector at right angles to ``direction``, chosen the way
    OCCT's ``gp_Ax2`` chooses its X direction from a main direction."""
    a, b, c = direction
    aa, bb, cc = abs(a), abs(b), abs(c)
    if bb <= aa and bb <= cc:
        x = (-c, 0.0, a) if aa > cc else (c, 0.0, -a)
    elif aa <= bb and aa <= cc:
        x = (0.0, -c, b) if bb > cc else (0.0, c, -b)
    else:
        x = (-b, a, 0.0) if aa > bb else (b, -a, 0.0)
    return _unit(x)


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
_DROPPED = object()  # marks an offset edge used up at a corner


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
            # one NURBS running along the wire's edges
            self.curve = kernel._joined_curve(shape)  # pylint: disable=protected-access
            domain = self.curve.GetNaturalInterval()
            self.first, self.last = domain.GetMin(), domain.GetMax()
            self.geom_type = kernel.curve_type(self.curve)
            return
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
        if self.geom_type == "LINE":
            return math.dist(self._point(a), self._point(b))
        whole = t0 is None and t1 is None
        if whole:
            # the whole length is asked for again and again (every
            # position_at): remembered by the curve and its ends
            key = (self.curve.address, a, b, _point_key(self._point(a)), _point_key(self._point(b)))
            known = self.kernel._lengths.get(key)  # pylint: disable=protected-access
            if known is not None:
                return known
        status, length = self.curve.Length(self.sc.SmExtent1d(a, b), 1e-10)
        _check(status, "curve length")
        if whole:
            self.kernel._lengths[key] = length  # pylint: disable=protected-access
        return length

    def param_at_length(self, length: float, t0: float | None = None) -> float:
        start = self.first if t0 is None else t0
        if self.geom_type == "LINE":
            whole = self.length()
            return start if whole <= 0 else start + (self.last - self.first) * length / whole
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

    def samples(self, count: int) -> list[Point]:
        """``count + 1`` points spread evenly in parameter, ends included."""
        return [
            self._point(self.first + (self.last - self.first) * i / count)
            for i in range(count + 1)
        ]

    def _spans(self) -> list[float]:
        """The parameters where the curve's spans meet, ends included."""
        breaks = [self.first]
        get_knots = getattr(self.curve, "GetKnotsAll", None)
        if get_knots is not None:
            status, knots = get_knots([])
            if status == SM_SUCCESS:
                for knot in knots:
                    if self.first + 1e-9 < knot < self.last - 1e-9 and knot - breaks[-1] > 1e-9:
                        breaks.append(float(knot))
        breaks.append(self.last)
        return breaks

    def positions_by_deflection(self, deflection: float) -> list[Point]:
        """Points along the curve whose chords stay within ``deflection``."""
        if deflection <= 0:
            raise RuntimeError("Deflection calculation failed")
        points: list[Point] = [self._point(self.first)]

        def refine(t0: float, p0: Point, t1: float, p1: Point, depth: int) -> None:
            t_mid = 0.5 * (t0 + t1)
            p_mid = self._point(t_mid)
            chord = tuple(b - a for a, b in zip(p0, p1))
            offset = tuple(b - a for a, b in zip(p0, p_mid))
            chord_len2 = _dot(chord, chord)
            if chord_len2 < 1e-24:
                gap = math.sqrt(_dot(offset, offset))
            else:
                along = _dot(offset, chord) / chord_len2
                foot = tuple(a + along * c for a, c in zip(p0, chord))
                gap = math.dist(p_mid, foot)
            if gap > deflection and depth < 16:
                refine(t0, p0, t_mid, p_mid, depth + 1)
                refine(t_mid, p_mid, t1, p1, depth + 1)
            else:
                points.append(p1)

        breaks = self._spans()
        for t0, t1 in zip(breaks, breaks[1:]):
            p0, p1 = self._point(t0), self._point(t1)
            if math.dist(p0, p1) > _TOLERANCE:
                refine(t0, p0, t1, p1, 0)
                continue
            # a span closing on itself (a full circle in one span) would be
            # mistaken for a chord of no length: start from its quarters
            quarter = [t0 + (t1 - t0) * i / 4 for i in range(5)]
            samples = [self._point(t) for t in quarter]
            for (ta, pa), (tb, pb) in zip(
                zip(quarter, samples), zip(quarter[1:], samples[1:])
            ):
                refine(ta, pa, tb, pb, 0)
        return points

    _GAUSS = (
        (0.0, 0.5688888888888889),
        (-0.5384693101056831, 0.47862867049936647),
        (0.5384693101056831, 0.47862867049936647),
        (-0.906179845938664, 0.23692688505618908),
        (0.906179845938664, 0.23692688505618908),
    )

    def centroid(self) -> Point:
        """The centre of mass of the curve (by arc length)."""
        total = 0.0
        moment = [0.0, 0.0, 0.0]
        breaks = self._spans()
        for t0, t1 in zip(breaks, breaks[1:]):
            pieces = 4
            for i in range(pieces):
                a = t0 + (t1 - t0) * i / pieces
                b = t0 + (t1 - t0) * (i + 1) / pieces
                half = 0.5 * (b - a)
                mid = 0.5 * (a + b)
                for node, weight in self._GAUSS:
                    point, tangent = self.d1(mid + half * node)
                    speed = math.sqrt(_dot(tangent, tangent)) * half * weight
                    total += speed
                    for k in range(3):
                        moment[k] += point[k] * speed
        if total < 1e-300:
            return self._point(self.first)
        return (moment[0] / total, moment[1] / total, moment[2] / total)

    def _frame(self) -> tuple[Point, Point, Point, Point]:
        """Origin, x, y and z axes of an analytic curve's placement (for a
        NURBS running on a circle: the circle's centre, the direction of the
        curve's start, and the axis)."""
        position = getattr(self.curve, "GetPosition", None)
        if position is None:
            fit = self.kernel._circle_fit(self.curve, self.first, self.last)  # pylint: disable=protected-access
            if fit is None:
                raise ValueError("not an analytic curve")
            centre, _, axis = fit
            x_axis = _unit(tuple(p - c for p, c in zip(self._point(self.first), centre)))
            return centre, x_axis, _cross(axis, x_axis), axis
        placement = position()
        pt = self.kernel._pt  # pylint: disable=protected-access
        return (
            pt(placement.GetOrigin()),
            pt(placement.GetXAxis()),
            pt(placement.GetYAxis()),
            pt(placement.GetZAxis()),
        )

    def circle(self) -> tuple[Point, Point, float]:
        if self.geom_type != "CIRCLE":
            raise ValueError("not a circle")
        origin, _, _, axis = self._frame()
        if hasattr(self.curve, "GetRadius"):
            return origin, axis, float(self.curve.GetRadius())
        return origin, axis, math.dist(origin, self._point(self.first))

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
        origin, du, dv = self.d1(*self._mid())
        _, normal = self.normal(*self._mid())
        if self.geom_type != "PLANE":
            # a surface of another kind that is flat all the same (a ruled
            # or skinned patch between straight sections)
            u0, u1, v0, v1 = self._bounds
            scale = max(1.0, max(abs(c) for c in origin))
            for i in range(5):
                for j in range(5):
                    point = self.value(u0 + (u1 - u0) * i / 4, v0 + (v1 - v0) * j / 4)
                    gap = _dot(normal, tuple(p - o for p, o in zip(point, origin)))
                    if abs(gap) > _TOLERANCE * scale:
                        return None
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
        """Principal curvatures from the fundamental forms, with the
        surface's own normal (not the face's outward one)."""
        evaluate = getattr(self.surface, "EvaluateDerivatives", None)
        if evaluate is None:
            raise NotImplementedError("smlib: surface curvatures")
        status, rows = evaluate(self.sc.SmVector2d(u, v), 2, 2, False, False)
        _check(status, "surface evaluation")
        su, sv = rows[1][0], rows[0][1]
        suu, suv, svv = rows[2][0], rows[1][1], rows[0][2]
        su, sv = (su.x, su.y, su.z), (sv.x, sv.y, sv.z)
        cross = _cross(su, sv)
        size = math.sqrt(_dot(cross, cross))
        if size < 1e-12:
            return None
        normal = (cross[0] / size, cross[1] / size, cross[2] / size)
        e, f, g = _dot(su, su), _dot(su, sv), _dot(sv, sv)
        l = _dot((suu.x, suu.y, suu.z), normal)
        m = _dot((suv.x, suv.y, suv.z), normal)
        n = _dot((svv.x, svv.y, svv.z), normal)
        denominator = e * g - f * f
        if abs(denominator) < 1e-18:
            return None
        gaussian = (l * n - m * m) / denominator
        mean = (e * n - 2 * f * m + g * l) / (2 * denominator)
        root = math.sqrt(max(0.0, mean * mean - gaussian))
        return mean - root, mean + root, normal

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
    failure_types = (RuntimeError, Standard_Failure)

    def __init__(self) -> None:
        self.sc = _import_core()
        self._probe = self._new_brep()
        self.context = self._probe.GetContext()
        self._meshes: dict[int, tuple[Any, float]] = {}  # brep address -> (poly brep, tolerance)
        self._bezier: set[int] = set()  # curve addresses made by make_bezier
        self._sheets: dict[int, tuple[Any, Any]] = {}  # face address -> (face, one-face body)
        self._boxes: dict[int, tuple[Point, Point]] = {}  # brep address -> box, during a boolean
        self._lengths: dict[tuple[Any, ...], float] = {}  # whole curve lengths by curve and ends

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

    kernel_pt = _pt

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

    def _lumps(self, brep: Any) -> list[Any]:
        """The separate solid bodies of a brep, each copied into its own brep
        when there are several; the brep itself when there is one."""
        # a solid is a region that is not void (the infinite region and the
        # cavities are); its shells are the outer one and the cavity walls
        regions: dict[int, list[Any]] = {}
        for shell in brep.GetShells([]):
            region = shell.GetRegion()
            if region.IsVoid():
                continue
            faces = regions.setdefault(region.address, [])
            known = {f.address for f in faces}
            for faceuse in shell.GetFaceuses([]):
                face = faceuse.GetFace()
                if face.address not in known:
                    known.add(face.address)
                    faces.append(face)
        if len(regions) <= 1:
            return [brep]
        lumps = []
        for faces in regions.values():
            lump = self._new_brep()
            _check(brep.CopyFaces(faces, lump, None, 1, None), "lump copy")
            lumps.append(lump)
        return lumps

    def _bodies_handle(self, brep: Any) -> SmHandle:
        """A solid, or a compound of solids when a brep holds several bodies."""
        lumps = self._lumps(brep)
        if len(lumps) == 1:
            if brep.IsManifoldSolid():
                return self._solid_handle(brep)
            return self._sheet_handle(brep)  # a sheet: one face, or a shell
        return SmHandle(
            TopoKind.COMPOUND, members=[self._solid_handle(l) for l in lumps]
        )

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
                if label == "BSPLINE":
                    if curve.address in self._bezier:
                        return "BEZIER"
                    return self._plain_nurbs_type(curve)
                return label
        return "OTHER"

    def _plain_nurbs_type(self, curve: Any) -> str:
        """A NURBS that is a straight line or a circular arc in all but name
        (the kernel's chamfers and sweeps make such) reports as one."""
        if int(curve.GetDegree()) == 1:
            status, poles, _ = curve.GetControlPolygon([], [])
            if status == SM_SUCCESS and len(poles) == 2:
                return "LINE"
        domain = curve.GetNaturalInterval()
        if self._circle_fit(curve, domain.GetMin(), domain.GetMax()) is not None:
            return "CIRCLE"
        return "BSPLINE"

    def _circle_fit(
        self, curve: Any, first: float, last: float
    ) -> tuple[Point, float, Point] | None:
        """Centre, radius and axis of the circle a curve runs on, if it does."""
        point = self.sc.SmVector3d()

        def at(t: float) -> Point:
            _check(curve.EvaluatePoint(t, point), "curve evaluation")
            return (point.x, point.y, point.z)

        span = last - first
        a, b, c = at(first), at(first + span / 3), at(first + 2 * span / 3)
        ab = tuple(q - p for p, q in zip(a, b))
        ac = tuple(q - p for p, q in zip(a, c))
        normal = _cross(ab, ac)
        size2 = _dot(normal, normal)
        if size2 < 1e-24:
            return None
        # circumcentre of the three points
        ab2, ac2 = _dot(ab, ab), _dot(ac, ac)
        to_centre = tuple(
            (ac2 * x - ab2 * y) / (2 * size2)
            for x, y in zip(_cross(normal, ab), _cross(normal, ac))
        )
        centre = tuple(p + d for p, d in zip(a, to_centre))
        radius = math.dist(centre, a)
        if radius < _TOLERANCE:
            return None
        axis = _unit(normal)
        for i in range(1, 12):
            q = at(first + span * i / 12)
            rel = tuple(x - y for x, y in zip(q, centre))
            if abs(math.dist(q, centre) - radius) > 1e-9 * radius or abs(_dot(rel, axis)) > 1e-9 * radius:
                return None
        return centre, radius, axis  # type: ignore[return-value]

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
                if label == "BSPLINE" and self._flat_surface(surface):
                    return "PLANE"  # a tapered extrusion's side, flat in all but name
                return label
        return "OTHER"

    def _flat_surface(self, surface: Any) -> bool:
        """Whether a surface is a plane over its whole domain (sampled)."""
        domain = surface.GetNaturalUVDomain()
        lo, hi = domain.GetMin(), domain.GetMax()
        point, normal = self.sc.SmVector3d(), self.sc.SmVector3d()
        um, vm = 0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y)
        if surface.EvaluateNormal(self.sc.SmVector2d(um, vm), 0, 0, normal) != SM_SUCCESS:
            return False
        if surface.EvaluatePoint(self.sc.SmVector2d(um, vm), point) != SM_SUCCESS:
            return False
        origin = (point.x, point.y, point.z)
        axis = (normal.x, normal.y, normal.z)
        scale = max(1.0, max(abs(c) for c in origin))
        for i in range(5):
            for j in range(5):
                uv = self.sc.SmVector2d(lo.x + (hi.x - lo.x) * i / 4, lo.y + (hi.y - lo.y) * j / 4)
                if surface.EvaluatePoint(uv, point) != SM_SUCCESS:
                    return False
                gap = _dot(axis, (point.x - origin[0], point.y - origin[1], point.z - origin[2]))
                if abs(gap) > _TOLERANCE * scale:
                    return False
        return True

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
        try:
            return self._mass_by_faces(brep)
        except RuntimeError:
            pass  # a face the integrators refuse: the body-wide one decides
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

    _GAUSS = SmCurveView._GAUSS
    _GAUSS3 = ((0.0, 0.8888888888888888), (-0.7745966692414834, 0.5555555555555556), (0.7745966692414834, 0.5555555555555556))
    _GAUSS2 = ((-0.5773502691896257, 1.0), (0.5773502691896257, 1.0))

    def _mass_by_faces(self, brep: Any) -> tuple[float, float, Point]:
        """Area, volume and centroid summed over the faces (divergence
        theorem): faces covering their whole surface domain by Gauss
        quadrature on the knot spans, trimmed ones by the kernel's integrator.
        The body-wide integrator is far slower at the accuracy wanted."""
        area = volume = 0.0
        moment = [0.0, 0.0, 0.0]
        for face in self._entities(brep, TopoKind.FACE):
            handle = SmHandle(TopoKind.FACE, face)
            view = SmSurfaceView(self, handle)
            planar = self._planar_face_integrals(view)
            if planar is not None:
                a, v, m = planar[0], planar[2], planar[3]
            elif self._covers_its_surface(handle, view):
                a, v, m = self._flux_quadrature(view)
            else:
                # the kernel's per-face flux is not reliable (zero on some
                # trimmed walls): integrated over the trimmed region through
                # its uv loops, unless the surface collapses to a point on a
                # side (a sphere's pole), where that integral stays inexact
                # and the body-wide integrator takes the whole body
                if any(self._degenerate_ends(view, True)) or any(self._degenerate_ends(view, False)):
                    raise RuntimeError("trimmed face with a pole")
                a, v, m = self._trimmed_flux(view)
            area += a
            volume += v
            for i in range(3):
                moment[i] += m[i]
        divisor = volume if abs(volume) > 1e-300 else 1.0
        centre = tuple(round(m / divisor, 12) + 0.0 for m in moment)  # rounding noise off
        return area, volume, centre  # type: ignore[return-value]

    def _covers_its_surface(self, face: SmHandle, view: SmSurfaceView) -> bool:
        """Whether a face is a whole rectangle of its surface's parameter
        space (its uv domain): one loop of at most four edge uses, with
        every corner of the domain a vertex of the face."""
        loops = face.entity.GetLoops([], 0, None)
        if len(loops) != 1 or len(loops[0].GetEdgeuses([])) > 4:
            return False
        u0, u1, v0, v1 = view._bounds  # pylint: disable=protected-access
        corners = [self.vertex_point(v) for v in self.sub_shapes(face, TopoKind.VERTEX)]
        if not corners:
            return False
        scale = max(1.0, max(abs(c) for p in corners for c in p))
        for u, v in ((u0, v0), (u1, v0), (u0, v1), (u1, v1)):
            point = view.value(u, v)
            if min(math.dist(point, c) for c in corners) > _TOLERANCE * scale:
                return False
        return True

    def _uv_loops(self, face: Any) -> list[list[tuple[Any, bool]]]:
        """The loops of a face as lists of (uv trim curve, reversed) in loop
        order, taken from the face use that carries the uv curves. The uv
        curves must follow the edges exactly (a circle on a sphere gets an
        approximation, which would taint an integral): checked by sampling."""
        surface = face.GetSurface()
        point, uv_point = self.sc.SmVector3d(), self.sc.SmVector3d()
        for faceuse in face.GetFaceuses():
            loops = []
            for loopuse in faceuse.GetLoopuses([], 0, None):
                loop = []
                for edgeuse in loopuse.GetEdgeuses([]):
                    curve = edgeuse.GetUVTrimCurvePointer()
                    if curve is None:
                        break
                    edge = edgeuse.GetEdge()
                    edge_curve, interval = edge.GetCurve(), edge.GetInterval()
                    domain = curve.GetNaturalInterval()
                    t0, t1 = domain.GetMin(), domain.GetMax()
                    for fraction in (0.21, 0.5, 0.79):
                        _check(curve.EvaluatePoint(t0 + (t1 - t0) * fraction, uv_point), "uv curve evaluation")
                        _check(surface.EvaluatePoint(self.sc.SmVector2d(uv_point.x, uv_point.y), point), "surface evaluation")
                        drop = edge_curve.DropPoint(interval, point, None, _TOLERANCE, None)
                        scale = max(1.0, abs(point.x), abs(point.y), abs(point.z))
                        if drop[0] != SM_SUCCESS or drop[-1] > 1e-9 * scale:
                            raise RuntimeError("inexact uv trim curve")
                    loop.append((curve, edgeuse.GetOrientation() != self.sc.SM_OT_SAME))
                else:
                    loops.append(loop)
                    continue
                break
            else:
                if loops:
                    return loops
        raise RuntimeError("no uv trim curves")

    def _trimmed_flux(self, view: SmSurfaceView) -> tuple[float, float, Point]:
        """Area, flux volume and first moments of a trimmed face: Green's
        theorem in parameter space turns the integral over the region into
        one around its uv loops of an inner integral along v, both by Gauss
        quadrature. The loops are closed with straight uv segments where an
        edge use carries no curve (a pole)."""
        sc = self.sc
        u0, u1, v0, v1 = view._bounds  # pylint: disable=protected-access
        status, knots = view.surface.GetKnots(sc.SM_SP_V, [])
        v_breaks = sorted({k for k in knots if v0 + 1e-9 < k < v1 - 1e-9}) if status == SM_SUCCESS else []
        parts = 3 if len(v_breaks) <= 3 else 1
        pole_start, pole_end = self._degenerate_ends(view, True)
        point, du, dv = sc.SmVector3d(), sc.SmVector3d(), sc.SmVector3d()
        evaluate = view.surface.Evaluate1stDerivatives
        gauss = self._GAUSS

        def inner(u: float, v: float) -> tuple[float, float, float, float, float]:
            """Integrals from v0 to v of the five integrands at this u."""
            bounds = self._graded(
                [v0, *(k for k in v_breaks if k < v), v],
                parts,
                pole_start,
                pole_end and v >= v1 - 1e-9,
            )
            area = flux = mx = my = mz = 0.0
            for lo, hi in zip(bounds, bounds[1:]):
                if True:
                    half, mid = 0.5 * (hi - lo), 0.5 * (lo + hi)
                    for node, weight in gauss:
                        _check(evaluate(sc.SmVector2d(u, mid + half * node), 0, 0, point, du, dv), "surface derivatives")
                        nx = du.y * dv.z - du.z * dv.y
                        ny = du.z * dv.x - du.x * dv.z
                        nz = du.x * dv.y - du.y * dv.x
                        w = half * weight
                        area += math.sqrt(nx * nx + ny * ny + nz * nz) * w
                        flux += (point.x * nx + point.y * ny + point.z * nz) * w / 3.0
                        mx += point.x * point.x * nx * w / 2.0
                        my += point.y * point.y * ny * w / 2.0
                        mz += point.z * point.z * nz * w / 2.0
            return area, flux, mx, my, mz

        totals = [0.0] * 5
        uv_point = sc.SmVector3d()

        def along(u: float, v: float, du_dt: float, w: float) -> None:
            if abs(du_dt) < 1e-15:
                return
            values = inner(u, v)
            for i in range(5):
                totals[i] -= values[i] * du_dt * w

        def segment(a: tuple[float, float], b: tuple[float, float]) -> None:
            if abs(b[0] - a[0]) < 1e-12:
                return  # a vertical uv segment contributes nothing to a du integral
            panels = 8
            for i in range(panels):
                lo, hi = i / panels, (i + 1) / panels
                half, mid = 0.5 * (hi - lo), 0.5 * (lo + hi)
                for node, weight in gauss:
                    t = mid + half * node
                    along(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, b[0] - a[0], half * weight)

        for loop in self._uv_loops(view.face.entity):
            ends = []
            for curve, reversed_ in loop:
                domain = curve.GetNaturalInterval()
                t0, t1 = domain.GetMin(), domain.GetMax()
                status, flat = curve.GetKnotsAll([])
                breaks = [t0, *sorted({k for k in flat if t0 + 1e-9 < k < t1 - 1e-9})] if status == SM_SUCCESS else [t0]
                breaks.append(t1)
                direction = -1.0 if reversed_ else 1.0
                for a, b in zip(breaks, breaks[1:]):
                    for i in range(3):
                        lo, hi = a + (b - a) * i / 3, a + (b - a) * (i + 1) / 3
                        half, mid = 0.5 * (hi - lo), 0.5 * (lo + hi)
                        for node, weight in gauss:
                            status, vectors = curve.EvaluateDerivatives(mid + half * node, 1, False, False)
                            _check(status, "uv curve evaluation")
                            along(vectors[0].x, vectors[0].y, direction * vectors[1].x, half * weight)
                _check(curve.EvaluatePoint(t1 if reversed_ else t0, uv_point), "uv curve evaluation")
                start = (uv_point.x, uv_point.y)
                _check(curve.EvaluatePoint(t0 if reversed_ else t1, uv_point), "uv curve evaluation")
                ends.append((start, (uv_point.x, uv_point.y)))
            for (_, tail), (head, _) in zip(ends, ends[1:] + ends[:1]):
                if math.dist(tail, head) > 1e-9:
                    segment(tail, head)  # a gap in the uv loop: a pole or seam
        area, flux, mx, my, mz = totals
        if area < 0:  # the loops ran clockwise in uv
            area, flux, mx, my, mz = -area, -flux, -mx, -my, -mz
        # the outward normal may run against du x dv: settle the sign once
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        _, du_m, dv_m = view.d1(um, vm)
        _, outward = view.normal(um, vm)
        if _dot(_cross(du_m, dv_m), outward) < 0:
            flux, mx, my, mz = -flux, -mx, -my, -mz
        return area, flux, (mx, my, mz)

    def _degenerate_ends(self, view: SmSurfaceView, along_v: bool) -> tuple[bool, bool]:
        """Whether the surface collapses to a point at the start / end of the
        v (or u) range: there the area element has a square-root behaviour."""
        u0, u1, v0, v1 = view._bounds  # pylint: disable=protected-access
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        scale = max(1.0, max(abs(c) for c in view.value(um, vm)))
        flags = []
        for end in ((v0, v1) if along_v else (u0, u1)):
            _, du, dv = view.d1(um, end) if along_v else view.d1(end, vm)
            other = du if along_v else dv  # the derivative across the end
            flags.append(math.sqrt(_dot(other, other)) < 1e-7 * scale)
        return flags[0], flags[1]

    @staticmethod
    def _graded(bounds: list[float], parts: int, at_start: bool, at_end: bool) -> list[float]:
        """Panel bounds: each span split in ``parts``, the first and last
        further halved geometrically toward a degenerate end."""
        fine = []
        for a, b in zip(bounds, bounds[1:]):
            fine.extend(a + (b - a) * i / parts for i in range(parts))
        fine.append(bounds[-1])
        levels = 8
        if at_start:
            a, b = fine[0], fine[1]
            inserts = [a + (b - a) * 2.0 ** -k for k in range(levels, 0, -1)]
            fine = [a, *inserts, *fine[1:]] if len(fine) > 2 else [a, *inserts, b]
        if at_end:
            a, b = fine[-2], fine[-1]
            inserts = [b - (b - a) * 2.0 ** -k for k in range(1, levels + 1)]
            fine = [*fine[:-1], *inserts, b]
        return fine

    def _flux_quadrature(self, view: SmSurfaceView) -> tuple[float, float, Point]:
        """Area, flux volume (1/3 of the integral of x.n) and the volume's
        first moments (1/2 of the integral of x_i^2 n_i) over a whole face."""
        sc = self.sc
        u0, u1, v0, v1 = view._bounds  # pylint: disable=protected-access
        breaks = []
        for direction, lo, hi in ((sc.SM_SP_U, u0, u1), (sc.SM_SP_V, v0, v1)):
            status, knots = view.surface.GetKnots(direction, [])
            inner = sorted({k for k in knots if lo + 1e-9 < k < hi - 1e-9}) if status == SM_SUCCESS else []
            breaks.append([lo, *inner, hi])
        # the outward normal may run against du x dv: settle the sign once
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        _, du_m, dv_m = view.d1(um, vm)
        _, outward = view.normal(um, vm)
        sign = 1.0 if _dot(_cross(du_m, dv_m), outward) >= 0 else -1.0
        # a few spans are cut finer for the quadrature; a dense knot vector
        # (a swept or approximated surface) is fine as it is, with fewer nodes
        panels = []
        rules = []
        for lo_hi, along_v in ((breaks[0], False), (breaks[1], True)):
            spans = len(lo_hi) - 1
            parts = 3 if spans <= 4 else 1
            rules.append(self._GAUSS if spans <= 16 else self._GAUSS3 if spans <= 64 else self._GAUSS2)
            panels.append(self._graded(lo_hi, parts, *self._degenerate_ends(view, along_v)))
        point, du, dv = sc.SmVector3d(), sc.SmVector3d(), sc.SmVector3d()
        evaluate = view.surface.Evaluate1stDerivatives
        area = volume = 0.0
        mx = my = mz = 0.0
        gauss_u, gauss_v = rules
        for ua, ub in zip(panels[0], panels[0][1:]):
            hu, cu = 0.5 * (ub - ua), 0.5 * (ua + ub)
            for va, vb in zip(panels[1], panels[1][1:]):
                hv, cv = 0.5 * (vb - va), 0.5 * (va + vb)
                for nu, wu in gauss_u:
                    u = cu + hu * nu
                    for nv, wv in gauss_v:
                        _check(evaluate(sc.SmVector2d(u, cv + hv * nv), 0, 0, point, du, dv), "surface derivatives")
                        nx = du.y * dv.z - du.z * dv.y
                        ny = du.z * dv.x - du.x * dv.z
                        nz = du.x * dv.y - du.y * dv.x
                        w = wu * wv * hu * hv
                        area += math.sqrt(nx * nx + ny * ny + nz * nz) * w
                        w *= sign
                        volume += (point.x * nx + point.y * ny + point.z * nz) * w / 3.0
                        mx += point.x * point.x * nx * w / 2.0
                        my += point.y * point.y * ny * w / 2.0
                        mz += point.z * point.z * nz * w / 2.0
        return area, volume, (mx, my, mz)

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
        if a.kind is b.kind is TopoKind.VERTEX:
            # a free vertex, or one of another body, is the same as any vertex
            # at its place (OCCT shares vertices between the faces built from
            # a wire and the wire; SMLib bodies each own theirs)
            if (
                a.entity is None
                or b.entity is None
                or a.entity.GetBrep().address != b.entity.GetBrep().address
            ):
                return math.dist(self.vertex_point(a), self.vertex_point(b)) < _TOLERANCE
        if a.kind is b.kind and a.kind in (TopoKind.EDGE, TopoKind.FACE) and (
            (a.entity is None) != (b.entity is None)
            or (
                a.entity is not None
                and b.entity is not None
                and a.entity.GetBrep().address != b.entity.GetBrep().address
            )
        ):
            # likewise an edge of a wire and the edge of a face built on it,
            # or the untouched face of a body an operation rebuilt
            if a.kind is TopoKind.EDGE:
                return self._same_curve(a, b)
            return self._same_face(a, b)
        return False

    def _same_face(self, a: SmHandle, b: SmHandle) -> bool:
        """Whether two faces cover the same patch of the same surface."""
        if self.surface_type(a.entity.GetSurface()) != self.surface_type(b.entity.GetSurface()):
            return False
        corners_a = {_point_key(p) for p in self._corner_points(a)}
        corners_b = {_point_key(p) for p in self._corner_points(b)}
        if corners_a != corners_b:
            return False
        try:
            if abs(self.area(a) - self.area(b)) > _TOLERANCE:
                return False
        except RuntimeError:
            return False
        return math.dist(self.center_of_mass(a, TopoKind.FACE), self.center_of_mass(b, TopoKind.FACE)) <= _TOLERANCE

    def _corner_points(self, shape: SmHandle) -> list[Point]:
        return [self.vertex_point(v) for v in self.sub_shapes(shape, TopoKind.VERTEX)]

    def _same_curve(self, a: SmHandle, b: SmHandle) -> bool:
        """Whether two edges run over the same points (either way round)."""
        view_a, view_b = SmCurveView(self, a), SmCurveView(self, b)
        if abs(view_a.length() - view_b.length()) > _TOLERANCE:
            return False
        ends_a = (view_a.value(view_a.first), view_a.value(view_a.last))
        ends_b = (view_b.value(view_b.first), view_b.value(view_b.last))
        same_way = math.dist(ends_a[0], ends_b[0]) < _TOLERANCE and math.dist(ends_a[1], ends_b[1]) < _TOLERANCE
        other_way = math.dist(ends_a[0], ends_b[1]) < _TOLERANCE and math.dist(ends_a[1], ends_b[0]) < _TOLERANCE
        if not (same_way or other_way):
            return False
        quarter_a = view_a.value(view_a.first + 0.25 * (view_a.last - view_a.first))
        param_b, distance = view_b.drop_point(quarter_a)
        return distance < _TOLERANCE and view_b.first - 1e-9 <= param_b <= view_b.last + 1e-9

    def is_equal(self, a: SmHandle, b: SmHandle) -> bool:
        return a.key() == b.key() and a.flip == b.flip

    def is_partner(self, a: SmHandle, b: SmHandle) -> bool:
        return a.key() == b.key() or (a.partner == b.partner and a.kind == b.kind)

    def hash_key(self, shape: SmHandle) -> Hashable:
        # vertices, edges and faces are the same across bodies when they lie
        # at the same place (see is_same), so their hashes are positional
        if shape.kind is TopoKind.VERTEX:
            return _point_key(self.vertex_point(shape), 5)
        if shape.entity is None:
            return shape.key()  # free edges stay apart, like OCCT's
        try:
            if shape.kind is TopoKind.EDGE:
                view = SmCurveView(self, shape)
                ends = frozenset(
                    _point_key(view.value(t), 5) for t in (view.first, view.last)
                )
                return (TopoKind.EDGE, ends, round(view.length(), 4))
            if shape.kind is TopoKind.FACE:
                # cheap: the kind of surface and where the corners are (the
                # area would need an integration, costly on blend faces)
                return (
                    TopoKind.FACE,
                    self.surface_type(shape.entity.GetSurface()),
                    frozenset(_point_key(p, 5) for p in self._corner_points(shape)),
                )
        except RuntimeError:
            pass  # a face the kernel cannot measure keeps its own identity
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
            key = item.key()
            if kind is TopoKind.VERTEX and shape.kind in (TopoKind.WIRE, TopoKind.COMPOUND):
                # the vertices of chained edges of mixed origin meet by position
                key = _point_key(self.vertex_point(item))
            found.setdefault(key, item)
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
            if kind is TopoKind.WIRE:  # the loops of every face
                return [
                    w
                    for f in self._entities(shape.entity, TopoKind.FACE)
                    for w in self.sub_shape_uses(SmHandle(TopoKind.FACE, f), TopoKind.WIRE)
                ]
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
        # the loop's edge uses run counter-clockwise about the surface normal;
        # the wire runs counter-clockwise about the face's outward normal
        outward_flipped = self.face_outward_is_flipped(face)
        for edgeuse in loop.GetEdgeuses([]):
            edge = edgeuse.GetEdge()
            reversed_use = edgeuse.GetOrientation() == self.sc.SM_OT_OPPOSITE
            flip = reversed_use != face.flip
            edges.append(SmHandle(TopoKind.EDGE, edge, flip=flip != outward_flipped))
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
        planar = self._planar_face_properties(face)
        if planar is not None:
            return planar
        status, area, _, moments = face.entity.ComputePreciseProperties(
            self.sc.SM_OT_SAME, _MASS_ACCURACY, self._vec((0.0, 0.0, 0.0)), 0.0, 0.0, []
        )
        if status != SM_SUCCESS:
            return self._face_properties_by_mesh(face)
        first = moments[0]
        divisor = area if area > 1e-300 else 1.0
        return area, (first.x / divisor, first.y / divisor, first.z / divisor)

    def _planar_face_properties(self, face: SmHandle) -> tuple[float, Point] | None:
        """Area and centroid of a planar face (None for a curved one)."""
        integrals = self._planar_face_integrals(SmSurfaceView(self, face))
        return None if integrals is None else (integrals[0], integrals[1])

    def _planar_face_integrals(
        self, view: SmSurfaceView
    ) -> tuple[float, Point, float, Point] | None:
        """Area, centroid, flux volume (1/3 of the integral of x.n) and the
        volume's first moments (1/2 of the integral of x_i^2 n_i) of a planar
        face, by integrating around its loops (Green's theorem, Gauss
        quadrature on each edge): exact to rounding where the kernel's
        integrator is not. None for a face that is not planar."""
        plane = view.plane()
        if plane is None:
            return None
        origin, normal, e1 = plane
        _, outward = view.normal(*view._mid())  # pylint: disable=protected-access
        e2 = _cross(normal, e1)
        area = ia = ib = iaa = ibb = iab = 0.0
        for wire in self.sub_shape_uses(view.face, TopoKind.WIRE):
            for curve in self._loop_curves(wire):
                domain = curve.GetNaturalInterval()
                edge = SmCurveView(self, self._free_edge(curve, domain.GetMin(), domain.GetMax()))
                if edge.geom_type == "LINE":
                    # a straight edge integrates in closed form
                    p0, p1 = edge.value(edge.first), edge.value(edge.last)
                    r0 = (p0[0] - origin[0], p0[1] - origin[1], p0[2] - origin[2])
                    r1 = (p1[0] - origin[0], p1[1] - origin[1], p1[2] - origin[2])
                    a0, b0, a1, b1 = _dot(r0, e1), _dot(r0, e2), _dot(r1, e1), _dot(r1, e2)
                    da, db = a1 - a0, b1 - b0
                    area += 0.5 * (a0 + a1) * db
                    ia += (a0 * a0 + a0 * a1 + a1 * a1) / 6.0 * db
                    ib -= (b0 * b0 + b0 * b1 + b1 * b1) / 6.0 * da
                    iaa += (a0 * a0 * a0 + a0 * a0 * a1 + a0 * a1 * a1 + a1 * a1 * a1) / 12.0 * db
                    ibb -= (b0 * b0 * b0 + b0 * b0 * b1 + b0 * b1 * b1 + b1 * b1 * b1) / 12.0 * da
                    iab += 0.5 * db * (
                        a0 * a0 * b0
                        + 0.5 * (a0 * a0 * db + 2 * a0 * da * b0)
                        + (2 * a0 * da * db + da * da * b0) / 3.0
                        + da * da * db / 4.0
                    )
                    continue
                breaks = edge._spans()  # pylint: disable=protected-access
                for t0, t1 in zip(breaks, breaks[1:]):
                    pieces = 4
                    for i in range(pieces):
                        lo = t0 + (t1 - t0) * i / pieces
                        hi = t0 + (t1 - t0) * (i + 1) / pieces
                        half, mid = 0.5 * (hi - lo), 0.5 * (lo + hi)
                        for node, weight in self._GAUSS:
                            point, tangent = edge.d1(mid + half * node)
                            rel = (point[0] - origin[0], point[1] - origin[1], point[2] - origin[2])
                            a, b = _dot(rel, e1), _dot(rel, e2)
                            da, db = _dot(tangent, e1), _dot(tangent, e2)
                            w = half * weight
                            area += a * db * w
                            ia += 0.5 * a * a * db * w
                            ib -= 0.5 * b * b * da * w
                            iaa += a * a * a / 3.0 * db * w
                            ibb -= b * b * b / 3.0 * da * w
                            iab += 0.5 * a * a * b * db * w
        if abs(area) < 1e-300:
            return None
        if area < 0:  # the loops run clockwise about the plane's normal
            area, ia, ib, iaa, ibb, iab = -area, -ia, -ib, -iaa, -ibb, -iab
        # the quadrature leaves rounding noise where OCCT is exact (a box
        # face's centre at -5.000000000000001): trimmed to 12 decimals
        centre = tuple(
            round(o + (ia / area) * x + (ib / area) * y, 12) + 0.0
            for o, x, y in zip(origin, e1, e2)
        )
        flux = _dot(origin, outward) * area / 3.0
        moments = []
        for i in range(3):
            square = (
                origin[i] * origin[i] * area
                + 2 * origin[i] * (e1[i] * ia + e2[i] * ib)
                + e1[i] * e1[i] * iaa
                + 2 * e1[i] * e2[i] * iab
                + e2[i] * e2[i] * ibb
            )
            moments.append(0.5 * square * outward[i])
        return area, centre, flux, (moments[0], moments[1], moments[2])  # type: ignore[return-value]

    def _face_properties_by_mesh(self, face: SmHandle) -> tuple[float, Point]:
        """Area and centroid summed over a fine triangulation, for the faces
        whose loops the kernel's integrator does not accept."""
        self.mesh(face, 1e-3, math.radians(5))
        triangulation = self.face_triangulation(face)
        if triangulation is None:
            raise RuntimeError("SMLib face properties failed")
        points, triangles = triangulation
        area = 0.0
        moment = [0.0, 0.0, 0.0]
        for i, j, k in triangles:
            a, b, c = points[i], points[j], points[k]
            n = _cross(tuple(q - p for p, q in zip(a, b)), tuple(q - p for p, q in zip(a, c)))
            piece = 0.5 * math.sqrt(_dot(n, n))
            area += piece
            for axis in range(3):
                moment[axis] += piece * (a[axis] + b[axis] + c[axis]) / 3
        divisor = area if area > 1e-300 else 1.0
        return area, (moment[0] / divisor, moment[1] / divisor, moment[2] / divisor)

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
            return SmCurveView(self, shape).centroid()
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
            view = SmCurveView(self, shape)
            extent = self.sc.SmExtent3d()
            _check(
                view.curve.CalculateTightBoundingBox(
                    self.sc.SmExtent1d(view.first, view.last), extent
                ),
                "bounding box",
            )
            boxes.append(
                tuple(extent.GetMin(i) for i in range(3))
                + tuple(extent.GetMax(i) for i in range(3))
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
        if shape.kind is TopoKind.FACE:
            # a face is worked on through its sheet body
            sheet = SmHandle(TopoKind.SHELL, shape.entity.GetBrep(), partner=shape.partner)
            inner = self._tracked_copy(sheet, record)
            record.modified[shape.key()] = record.modified.get(shape.key(), record.modified[sheet.key()])
            return inner
        if shape.kind is TopoKind.SHELL and shape.entity.IsManifoldSolid():
            # the shell of a solid, asked for as a surface: a sheet copy of
            # its faces, which bounds no region
            sheet = self._new_brep()
            faces = self._entities(shape.entity, TopoKind.FACE)
            _check(shape.entity.CopyFaces(faces, sheet, None, 0, None), "sheet copy")
            record.modified[shape.key()] = [SmHandle(TopoKind.SHELL, sheet, partner=shape.partner)]
            return sheet
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
        results = result if isinstance(result, list) else [result]
        alive = {
            e.address
            for r in results
            for kind in (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE)
            for e in self._entities(r, kind)
        }
        alive.update(r.address for r in results)
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
        if op is BoolOp.SECTION and all(self._is_one_dimensional(a) for a in args):
            return self._section_points(args, tools), _Record()
        if op is BoolOp.SECTION and all(self._is_one_dimensional(t) for t in tools):
            return self._section_points(tools, args), _Record()
        if op is BoolOp.SECTION:
            return self._section_edges(args, tools), _Record()
        if op not in ops:
            raise NotImplementedError(f"smlib: boolean {op.value}")
        record = _Record()
        if (
            op is BoolOp.COMMON
            and all(self._is_one_dimensional(a) for a in args)
            and not any(self._is_one_dimensional(t) for t in tools)
        ):
            return self._edges_inside(args, tools), record
        if (
            op is BoolOp.COMMON
            and all(self._is_one_dimensional(t) for t in tools)
            and not any(self._is_one_dimensional(a) for a in args)
        ):
            return self._edges_inside(tools, args), record
        if op is BoolOp.CUT and all(self._is_one_dimensional(a) for a in args):
            return self._edge_pieces(args, tools, False), record
        if all(self._is_one_dimensional(s) for s in (*args, *tools)):
            if op is BoolOp.COMMON:
                return self._edge_pieces(args, tools, True), record
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
        # the merges consume their operands, so every body's box is taken
        # before it goes in (and a merge result's as soon as it is made)
        self._boxes = {}
        for brep in (*arg_bodies, *tool_bodies):
            self._box_of(brep, fresh=True)
        results = self._fuse_bodies([], arg_bodies)
        if op is BoolOp.FUSE:
            results = self._fuse_bodies(results, tool_bodies)
        else:
            # a merge consumes both bodies: the tracked copies go in as they
            # are the first time (so the record stays true) and as fresh
            # copies when they are needed again
            used: set[int] = set()

            def operand(brep: Any) -> Any:
                if brep.address in used:
                    copy = self._copy_body(brep)[0]
                    self._boxes[copy.address] = self._boxes[brep.address]
                    return copy
                used.add(brep.address)
                return brep

            if op is BoolOp.CUT:
                for tool in tool_bodies:
                    cut = []
                    for body in results:
                        if self._boxes_touch(body, tool):
                            body = self._merge(body, operand(tool), self.sc.SM_BO_DIFFERENCE)
                        cut.append(body)
                    results = cut
            else:  # COMMON: what each body shares with each tool
                common = []
                for body in results:
                    for tool in tool_bodies:
                        if self._boxes_touch(body, tool):
                            common.append(
                                self._merge(operand(body), operand(tool), self.sc.SM_BO_INTERSECTION)
                            )
                results = common
        results = [r for r in results if self._entities(r, TopoKind.FACE)]
        self._finish_record(record, results)
        self._record_by_geometry(record, [*args, *tools], results)
        if all(s.kind is TopoKind.FACE for s in (*args, *tools)):
            # faces combine into faces; a sheet of one face is that face
            faces = [f for r in results for f in self._entities(r, TopoKind.FACE)]
            if len(faces) == 1:
                return SmHandle(TopoKind.FACE, faces[0]), record
            return (
                SmHandle(
                    TopoKind.COMPOUND,
                    members=[SmHandle(TopoKind.FACE, f) for f in faces],
                ),
                record,
            )
        handles = [self._bodies_handle(r) for r in results]
        if len(handles) == 1:
            return handles[0], record
        members = [m for h in handles for m in (h.members if h.kind is TopoKind.COMPOUND else [h])]
        return SmHandle(TopoKind.COMPOUND, members=members), record

    def _record_by_geometry(
        self, record: _Record, inputs: Sequence[SmHandle], results: Sequence[Any]
    ) -> None:
        """Record what each input face, edge and vertex became by geometry:
        the merge builds its result from new entities, so a result entity
        that lies on an input's surface, curve or point is what it became."""
        faces = [f for r in results for f in self._entities(r, TopoKind.FACE)]
        edges = [e for r in results for e in self._entities(r, TopoKind.EDGE)]
        vertices = [v for r in results for v in self._entities(r, TopoKind.VERTEX)]
        face_points = {}
        for f in faces:
            handle = SmHandle(TopoKind.FACE, f)
            view = SmSurfaceView(self, handle)
            face_points[f.address] = [self.vertex_point(v) for v in self.sub_shapes(handle, TopoKind.VERTEX)] + [view.value(*view._mid())]  # pylint: disable=protected-access
        edge_points = {}
        for e in edges:
            view = SmCurveView(self, SmHandle(TopoKind.EDGE, e))
            edge_points[e.address] = [view.value(view.first), view.value(0.5 * (view.first + view.last)), view.value(view.last)]
        vertex_points = {v.address: self._pt(v.GetPoint()) for v in vertices}

        def within(points: Sequence[Point], box: tuple[Point, Point]) -> bool:
            low, high = box
            return all(
                low[i] - _TOLERANCE <= p[i] <= high[i] + _TOLERANCE for p in points for i in range(3)
            )

        def box_of(points: Sequence[Point]) -> tuple[Point, Point]:
            return (
                tuple(min(p[i] for p in points) for i in range(3)),  # type: ignore[return-value]
                tuple(max(p[i] for p in points) for i in range(3)),  # type: ignore[return-value]
            )

        for shape in inputs:
            for face in self.sub_shapes(shape, TopoKind.FACE):
                view = SmSurfaceView(self, face)
                corners = [self.vertex_point(v) for v in self.sub_shapes(face, TopoKind.VERTEX)]
                samples = [view.value(*view._mid())]  # pylint: disable=protected-access
                box = box_of(corners + samples) if corners else None
                became = []
                for f in faces:
                    points = face_points[f.address]
                    if box is not None and view.geom_type == "PLANE" and not within(points, box):
                        continue
                    if all(math.dist(view.value(*view.closest_uv(p)), p) <= _TOLERANCE for p in points):
                        became.append(SmHandle(TopoKind.FACE, f))
                self._note(record, face, became)
            for edge in self.sub_shapes(shape, TopoKind.EDGE):
                view = SmCurveView(self, edge)
                became = []
                for e in edges:
                    if all(view.drop_point(p)[1] <= _TOLERANCE for p in edge_points[e.address]):
                        became.append(SmHandle(TopoKind.EDGE, e))
                self._note(record, edge, became)
            for vertex in self.sub_shapes(shape, TopoKind.VERTEX):
                point = self.vertex_point(vertex)
                became = [
                    SmHandle(TopoKind.VERTEX, v)
                    for v in vertices
                    if math.dist(vertex_points[v.address], point) <= _TOLERANCE
                ]
                self._note(record, vertex, became)

    @staticmethod
    def _note(record: _Record, sub: SmHandle, became: list[SmHandle]) -> None:
        if became:
            record.modified[sub.key()] = became
            record.removed.discard(sub.key())
        else:
            record.modified.pop(sub.key(), None)
            record.removed.add(sub.key())

    def _box_of(self, brep: Any, fresh: bool = False) -> tuple[Point, Point]:
        """The box of a body, cached by address during a boolean; a body
        just made is measured afresh as its address may be a dead body's."""
        known = None if fresh else self._boxes.get(brep.address)
        if known is not None:
            return known
        extent = self.sc.SmExtent3d()
        _check(brep.CalculateTightBoundingBox(extent), "bounding box")
        box = (
            tuple(extent.GetMin(i) for i in range(3)),
            tuple(extent.GetMax(i) for i in range(3)),
        )
        self._boxes[brep.address] = box
        return box  # type: ignore[return-value]

    def _boxes_touch(self, a: Any, b: Any) -> bool:
        (min_a, max_a), (min_b, max_b) = self._box_of(a), self._box_of(b)
        return all(
            min_a[i] <= max_b[i] + _TOLERANCE and min_b[i] <= max_a[i] + _TOLERANCE
            for i in range(3)
        )

    def _fuse_bodies(self, results: list[Any], bodies: Sequence[Any]) -> list[Any]:
        """Union ``bodies`` into ``results``, merging only what touches so
        that the bodies apart stay separate breps."""
        results = list(results)
        for body in bodies:
            touching = [r for r in results if self._boxes_touch(r, body)]
            merged = body
            for other in touching:
                merged = self._merge(merged, other, self.sc.SM_BO_UNION)
            results = [r for r in results if r not in touching] + [merged]
        return results

    def _split(
        self, args: Sequence[SmHandle], tools: Sequence[SmHandle]
    ) -> tuple[SmHandle, Any]:
        """Split bodies by planar tools: each side of a tool's plane is kept
        as the part of the body inside a large box on that side."""
        record = _Record()
        if all(a.kind is TopoKind.EDGE for a in args) and all(
            t.kind is TopoKind.VERTEX for t in tools
        ):
            return self._split_edges_at_points(args, tools), record
        if all(self._is_one_dimensional(a) for a in args):
            # edges cut where they cross the tools, every piece kept
            inside = self._edge_pieces(args, tools, True)
            outside = self._edge_pieces(args, tools, False)
            return SmHandle(TopoKind.COMPOUND, members=[*inside.members, *outside.members]), record
        pieces: list[SmHandle] = [
            SmHandle(b.kind, b.entity, partner=b.partner) for a in args for b in self._bodies(a)
        ]
        for tool in tools:
            faces = self.sub_shapes(tool, TopoKind.FACE)
            if len(faces) != 1 or self.planar_surface(faces[0]) is None:
                pieces = self._split_by_sheet(pieces, tool, record)
                continue
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
                    if self.is_null(part) or self._extent(part) <= _TOLERANCE:
                        continue
                    for key, became in part_record.modified.items():
                        record.modified.setdefault(key, []).extend(became)
                    next_pieces.extend(
                        part.members if part.kind is TopoKind.COMPOUND else [part]
                    )
            pieces = next_pieces
        if len(pieces) == 1:
            return pieces[0], record
        return SmHandle(TopoKind.COMPOUND, members=pieces), record

    def is_inside_out(self, solid: SmHandle) -> bool:
        return False  # the kernel's regions keep the shells facing out

    def same_side(self, piece: SmHandle, tool: SmHandle, point: Point, tolerance: float) -> bool:
        """Whether a piece's centre of mass is on the side of the sheet that
        ``point`` is on (the sign of the normal at the nearest point)."""
        centre = self.center_of_mass(piece, piece.kind)
        return self._sheet_side(tool, centre) == self._sheet_side(tool, point)

    def _sheet_side(self, tool: SmHandle, point: Point) -> bool:
        best: tuple[float, bool] | None = None
        for face in (
            [tool] if tool.kind is TopoKind.FACE else self.sub_shapes(tool, TopoKind.FACE)
        ):
            view = SmSurfaceView(self, face)
            uv = view.closest_uv(point)
            nearest, normal = view.normal(*uv)
            distance = math.dist(nearest, point)
            if best is None or distance < best[0]:
                best = (distance, _dot(normal, tuple(p - n for p, n in zip(point, nearest))) >= 0)
        return best is not None and best[1]

    def _extent(self, shape: SmHandle) -> float:
        """Volume of a solid, area of a sheet: what tells a piece from nothing."""
        if shape.kind is TopoKind.COMPOUND:
            return sum(self._extent(m) for m in shape.members)
        if shape.kind is TopoKind.SOLID:
            return self.volume(shape)
        return self.area(shape)

    def _split_by_sheet(
        self, pieces: Sequence[SmHandle], tool: SmHandle, record: _Record
    ) -> list[SmHandle]:
        """Split solids by a sheet of any shape: merged into a body without
        joining regions, the sheet walls off the regions it closes, and each
        solid region is copied out as a piece."""
        tool_body = self._body(tool)
        result: list[SmHandle] = []
        for piece in pieces:
            if piece.kind is not TopoKind.SOLID:
                raise NotImplementedError("smlib: splitting a sheet by a curved sheet")
            copy, _ = self._copy_body(piece.entity)
            sheet, _ = self._copy_body(tool_body)
            merge = self.sc.SmMerge(self.context, copy, sheet, 0.0, 0.0, None)
            status, merged = merge.NonManifoldBoolean(self.sc.SM_BO_MERGE)
            _check(status, "split")
            if merged is None:
                raise RuntimeError("SMLib split produced no body")
            seen: set[tuple[Any, ...]] = set()
            for lump in self._lumps(merged):
                handle = SmHandle(TopoKind.SOLID, lump)
                if not lump.IsManifoldSolid():
                    continue
                _, volume, centre = self._mass(lump)
                key = (round(volume, 6),) + _point_key(centre, 6)
                if key in seen or volume <= _TOLERANCE:
                    continue  # the same region walled off twice
                seen.add(key)
                result.append(handle)
        self._record_by_geometry(record, [tool, *pieces], [p.entity for p in result])
        return result

    def _crossings(self, curve: Any, interval: Any, other: tuple[str, Any]) -> list[Point]:
        """Where a curve meets another curve, a sheet or a body."""
        sc = self.sc
        solutions = sc.SmSolutionArray()
        if other[0] == "curve":
            other_curve, other_interval = other[1]
            # the intersector, unlike the general solver, finds crossings at
            # the seam of a closed curve
            _check(
                curve.GlobalCurveIntersect(
                    interval, other_curve, other_interval, _TOLERANCE, solutions, 0
                ),
                "curve crossing",
            )
        elif other[0] == "body":
            _check(
                sc.SmTopologySolver.BrepCurveSolve(
                    other[1], curve, interval, sc.SM_SO_INTERSECT, sc.SM_SR_ALL,
                    _TOLERANCE, _HUGE, None, solutions,
                ),
                "curve to body crossing",
            )
        elif other[0] == "point":
            solutions = None
            view_point = other[1]
            drop = curve.DropPoint(interval, self._vec(view_point), None, _TOLERANCE, None)
            return [view_point] if drop[0] == SM_SUCCESS and drop[-1] <= _TOLERANCE else []
        else:
            raise NotImplementedError("smlib: section with a face of a body")
        return [found[1] for found in self._solutions(solutions, 1)]

    def _section_points(
        self, args: Sequence[SmHandle], tools: Sequence[SmHandle]
    ) -> SmHandle:
        """The vertices where edges and wires cross the tools."""
        points: dict[tuple[Any, ...], Point] = {}
        for arg in args:
            for kind, data in self._atoms(arg):
                if kind != "curve":
                    continue
                curve, interval = data
                for tool in tools:
                    for other in self._atoms(tool):
                        for point in self._crossings(curve, interval, other):
                            points.setdefault(_point_key(point), point)
        return SmHandle(
            TopoKind.COMPOUND,
            members=[self.make_vertex(p) for p in points.values()],
        )

    def _edge_pieces(
        self, args: Sequence[SmHandle], tools: Sequence[SmHandle], keep_inside: bool
    ) -> SmHandle:
        """The parts of edges inside (or outside) the tools: the edges are
        cut where they cross the tools' faces, or where they meet the ends of
        tool edges, and each piece is kept by where its middle lies."""
        bodies = [b for t in tools for b in self._bodies(t)] if not all(
            self._is_one_dimensional(t) for t in tools
        ) else []
        tool_views = (
            [SmCurveView(self, e) for t in tools for e in self.sub_shapes(t, TopoKind.EDGE)]
            if not bodies
            else []
        )

        def inside(point: Point) -> bool:
            if bodies:
                return any(self.point_in_solid(b, point, _TOLERANCE) for b in bodies)
            return any(v.drop_point(point)[1] <= _TOLERANCE for v in tool_views)

        pieces = []
        for arg in args:
            for edge in self.sub_shapes(arg, TopoKind.EDGE):
                view = SmCurveView(self, edge)
                interval = self.sc.SmExtent1d(view.first, view.last)
                crossings: list[Point] = []
                for body in bodies:
                    crossings.extend(self._crossings(view.curve, interval, ("body", body.entity)))
                    if body.kind is TopoKind.SHELL:
                        # an edge lying in a sheet is cut where it leaves it
                        for rim in self._entities(body.entity, TopoKind.EDGE):
                            rim_interval = rim.GetInterval()
                            crossings.extend(
                                self._crossings(
                                    view.curve,
                                    interval,
                                    ("curve", (rim.GetCurve(), self.sc.SmExtent1d(rim_interval.GetMin(), rim_interval.GetMax()))),
                                )
                            )
                for other in tool_views:
                    crossings.extend(
                        self._crossings(
                            view.curve,
                            interval,
                            ("curve", (other.curve, self.sc.SmExtent1d(other.first, other.last))),
                        )
                    )
                    crossings.extend(other.value(t) for t in (other.first, other.last))
                cuts = set()
                for point in crossings:
                    param, distance = view.drop_point(point)
                    if distance <= _TOLERANCE:
                        cuts.add(round(param, 9))
                bounds = sorted({view.first, view.last, *cuts})
                for t0, t1 in zip(bounds, bounds[1:]):
                    if t1 - t0 <= _TOLERANCE:
                        continue
                    middle = view.value((t0 + t1) / 2)
                    if inside(middle) == keep_inside:
                        piece = self.trim_edge(edge, t0, t1)
                        piece.flip = edge.flip
                        pieces.append(piece)
        return SmHandle(TopoKind.COMPOUND, members=pieces)

    def _edges_inside(
        self, args: Sequence[SmHandle], tools: Sequence[SmHandle]
    ) -> SmHandle:
        return self._edge_pieces(args, tools, True)

    def _section_edges(
        self, args: Sequence[SmHandle], tools: Sequence[SmHandle]
    ) -> SmHandle:
        """The edges where faces, shells and solids cross each other: the
        intersection edges the kernel's merge computes, copied out as free edges."""
        edges: list[SmHandle] = []
        for arg in args:
            for tool in tools:
                for a in self._bodies(arg):
                    for b in self._bodies(tool):
                        copy_a, _ = self._copy_body(a.entity)
                        copy_b, _ = self._copy_body(b.entity)
                        merge = self.sc.SmMerge(self.context, copy_a, copy_b, 0.0, 0.0, None)
                        status, _result = merge.ManifoldBoolean(
                            self.sc.SM_BO_INTERSECTION, None
                        )
                        if status != SM_SUCCESS:
                            continue
                        if (
                            a.kind is TopoKind.SHELL
                            and b.kind is TopoKind.SHELL
                            and _result is not None
                            and self._entities(_result, TopoKind.FACE)
                        ):
                            # sheets overlapping in their plane share an
                            # area, not edges (the kernel's edge list is
                            # not valid then)
                            continue
                        for edge in merge.GetIntersectionEdges([]):
                            interval = edge.GetInterval()
                            edges.append(
                                self._free_edge(
                                    self._copy_curve(edge.GetCurve()),
                                    interval.GetMin(),
                                    interval.GetMax(),
                                )
                            )
        return SmHandle(TopoKind.COMPOUND, members=edges)

    def _split_edges_at_points(
        self, edges: Sequence[SmHandle], vertices: Sequence[SmHandle]
    ) -> SmHandle:
        """Edges cut at the vertices that lie on them."""
        pieces = []
        for edge in edges:
            view = SmCurveView(self, edge)
            cuts = sorted(
                param
                for param, distance in (
                    view.drop_point(self.vertex_point(v)) for v in vertices
                )
                if distance <= _TOLERANCE
                and view.first + _TOLERANCE < param < view.last - _TOLERANCE
            )
            bounds = [view.first, *cuts, view.last]
            if len(bounds) == 2:
                pieces.append(edge)
                continue
            for t0, t1 in zip(bounds, bounds[1:]):
                piece = self.trim_edge(edge, t0, t1)
                piece.flip = edge.flip
                pieces.append(piece)
        return SmHandle(TopoKind.COMPOUND, members=pieces)

    def _merge(self, a: Any, b: Any, op: Any) -> Any:
        merge = self.sc.SmMerge(self.context, a, b, 0.0, 0.0, None)
        status, result = merge.ManifoldBoolean(op, None)
        _check(status, "boolean")
        if result is None:
            raise RuntimeError("SMLib boolean produced no body")
        if self._entities(result, TopoKind.FACE):
            self._box_of(result, fresh=True)
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
        return self.boolean(BoolOp.SECTION, [a], [b]), _Record()

    def unify_same_domain(self, shape: SmHandle) -> tuple[SmHandle, Any]:
        return shape, _Record()

    # ---- placement ----------------------------------------------------------

    def identity_location(self) -> TopLoc_Location:
        return TopLoc_Location()

    def location_of(self, shape: SmHandle) -> TopLoc_Location:
        return shape.location

    def _affine_parts(
        self, matrix: Any, translation: Any
    ) -> tuple[Any, Any, bool]:
        """Split an affine map ``x -> M x + t`` into what SMLib's ``Transform``
        applies: a mirror in the XY plane (first), a scaling along the world
        axes, then a rotation and move given as a placement.

        Only maps of the form ``R diag(s)`` (with ``R`` possibly improper) are
        representable; anything else raises ``NotImplementedError``."""
        columns = [
            [matrix.Value(row, col) for row in (1, 2, 3)] for col in (1, 2, 3)
        ]
        scales = [math.sqrt(_dot(c, c)) for c in columns]
        if min(scales) < 1e-12:
            raise NotImplementedError("smlib: degenerate (collapsing) transform")
        axes = [[v / scale for v in c] for c, scale in zip(columns, scales)]
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(_dot(axes[i], axes[j])) > 1e-9:
                    raise NotImplementedError(
                        "smlib: shear or scaling along rotated axes"
                    )
        mirror = _dot(_cross(axes[0], axes[1]), axes[2]) < 0
        if mirror:
            # R = R' diag(1, 1, -1): mirror the geometry in XY first, then
            # apply the proper rotation R'
            axes[2] = [-v for v in axes[2]]
        origin = (translation.X(), translation.Y(), translation.Z())
        placement = self.sc.SmAxis2Placement(
            self._vec(origin), self._vec(axes[0]), self._vec(axes[1])
        )
        uniform = all(abs(scale - 1.0) < 1e-12 for scale in scales)
        return placement, (None if uniform else self._vec(scales)), mirror

    def _xy_mirror(self) -> Any:
        return self.sc.SmAxis2Placement(
            self._vec((0, 0, 0)), self._vec((1, 0, 0)), self._vec((0, 1, 0))
        )

    def _bake(
        self,
        shape: SmHandle,
        trsf: gp_Trsf,
        copies: dict[int, tuple[Any, Any]] | None = None,
        general: Any | None = None,
    ) -> SmHandle:
        """A moved copy of ``shape``: the geometry transformed, the placement
        remembered. ``copies`` maps a body's address to its moved copy and
        forward map, so that the members of one container share one copy.
        ``general`` is a ``gp_GTrsf`` applied instead of ``trsf`` (its
        placement is then not remembered, as the geometry absorbs it)."""
        if general is None:
            if trsf.Form() == 0 and trsf.TranslationPart().Modulus() == 0:
                return shape  # gp_Identity
            parts = self._affine_parts(trsf.VectorialPart(), trsf.TranslationPart())
            rigid = parts[1] is None and not parts[2]
        else:
            parts = self._affine_parts(general.VectorialPart(), general.TranslationPart())
            rigid = False
        if copies is None:
            copies = {}
        placement, scale, mirror = parts
        new_location = (
            TopLoc_Location(trsf).Multiplied(shape.location) if rigid else shape.location
        )
        if shape.kind in (TopoKind.WIRE, TopoKind.COMPOUND):
            return SmHandle(
                shape.kind,
                members=[self._bake(m, trsf, copies, general) for m in shape.members],
                flip=shape.flip,
                location=new_location,
                partner=shape.partner,
            )
        if shape.point is not None:
            if general is None:
                moved = gp_Pnt(*shape.point).Transformed(trsf)
            else:
                xyz = gp_XYZ(*shape.point)
                general.Transforms(xyz)
                moved = gp_Pnt(xyz)
            return SmHandle(
                TopoKind.VERTEX,
                point=(moved.X(), moved.Y(), moved.Z()),
                location=new_location,
                partner=shape.partner,
            )
        if shape.curve is not None:
            curve = self._copy_curve(shape.curve)
            if mirror:
                status, curve = curve.CreateMirrorCurve(self.context, self._xy_mirror())
                _check(status, "curve mirror")
            _check(curve.Transform(placement, scale), "curve transform")
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
            if mirror:
                _check(copy.Mirror(self._xy_mirror()), "mirror")
            _check(copy.Transform(placement, scale, 1), "transform")
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

    @staticmethod
    def _take_over(shape: SmHandle, moved: SmHandle) -> None:
        """Make ``shape`` hold what ``moved`` holds (an in-place move)."""
        shape.entity = moved.entity
        shape.members = moved.members
        shape.location = moved.location
        shape.point = moved.point
        shape.curve = moved.curve
        shape.interval = moved.interval

    def move(self, shape: SmHandle, location: TopLoc_Location) -> None:
        self._take_over(shape, self._bake(shape, location.Transformation()))

    def set_location(self, shape: SmHandle, location: TopLoc_Location) -> None:
        self._take_over(shape, self.located(shape, location))

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
        sphere = self._primitive(
            lambda c: c.CreateSphere(radius, 0.0, degrees, placement)
        )
        low, high = math.sin(angle1) * radius, math.sin(angle2) * radius
        if low <= -radius + _TOLERANCE and high >= radius - _TOLERANCE:
            return sphere
        # the kernel's sphere spans the poles: the part between the two
        # latitudes is what is inside a box of that height about the axis
        axis = frame.Axis()
        origin, direction = axis.Location(), axis.Direction()
        bottom = gp_Pnt(
            origin.X() + direction.X() * (low - _INFINITE),
            origin.Y() + direction.Y() * (low - _INFINITE),
            origin.Z() + direction.Z() * (low - _INFINITE),
        )
        slab_frame = gp_Ax2(bottom, direction, frame.XDirection())
        slab = self.make_box(slab_frame, 2 * _INFINITE, 2 * _INFINITE, high - low + _INFINITE)
        # the box is placed by its corner: shift it to be centred on the axis
        shift = gp_Trsf()
        shift.SetTranslation(
            gp_Vec(frame.XDirection()).Multiplied(-_INFINITE).Added(
                gp_Vec(frame.YDirection()).Multiplied(-_INFINITE)
            )
        )
        slab = self._bake(slab, shift)
        result, _ = self.boolean(BoolOp.COMMON, [sphere], [slab])
        return result

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

    def make_wedge(
        self,
        frame: Any,
        delta_x: float,
        delta_y: float,
        delta_z: float,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
    ) -> SmHandle:
        """A wedge as a ruled loft from its base rectangle to its top one."""
        origin, x_dir, y_dir, z_dir = (
            frame.Location(), frame.XDirection(), frame.YDirection(), frame.Direction()
        )

        def at(x: float, y: float, z: float) -> Point:
            return (
                origin.X() + x * x_dir.X() + y * y_dir.X() + z * z_dir.X(),
                origin.Y() + x * x_dir.Y() + y * y_dir.Y() + z * z_dir.Y(),
                origin.Z() + x * x_dir.Z() + y * y_dir.Z() + z * z_dir.Z(),
            )

        if max_x - min_x <= _TOLERANCE or max_z - min_z <= _TOLERANCE:
            raise NotImplementedError("smlib: wedges tapering to an edge or a point")
        base = self.make_polygon(
            [at(0, 0, 0), at(delta_x, 0, 0), at(delta_x, 0, delta_z), at(0, 0, delta_z), at(0, 0, 0)]
        )
        top = self.make_polygon(
            [
                at(min_x, delta_y, min_z),
                at(max_x, delta_y, min_z),
                at(max_x, delta_y, max_z),
                at(min_x, delta_y, max_z),
                at(min_x, delta_y, min_z),
            ]
        )
        return self.loft([base, top], True, True)

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
            if self.planar_surface(shape) is None:
                # a curved face sweeps into the sheet of its boundary, closed
                # by the face itself and a copy of it moved to the far end
                walls = self._primitive(
                    lambda c: c.CreateLinearSweep(curves, unit, distance, 1, 0, 1)
                )
                far = self._bake(shape, _translation(direction))
                faces = [*self.sub_shapes(walls, TopoKind.FACE), shape, far]
                _, record = self.sew(faces)
                result = record.result
                assert result is not None
                return self.solid_from_shell(result)
            return self._primitive(
                lambda c: c.CreateLinearSweep(curves, unit, distance, 1, 1, 1)
            )
        if shape.kind in (TopoKind.WIRE, TopoKind.EDGE):
            wire = shape if shape.kind is TopoKind.WIRE else SmHandle(TopoKind.WIRE, members=[shape])
            curves = self._loop_curves(wire)
            sheet = self._primitive(
                lambda c: c.CreateLinearSweep(curves, unit, distance, 1, 0, 1)
            )
            faces = self._entities(sheet.entity, TopoKind.FACE)
            for face in faces:
                # the sweep's normal is tangent x direction; OCCT's sheets
                # face the other way (direction x tangent)
                _check(face.SwapUV(), "face orientation")
            if shape.kind is TopoKind.EDGE and len(faces) == 1:
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
        sign = -1.0 if angle < 0 else 1.0  # a negative angle turns the other way
        angle = abs(angle)
        axis_dir = self._vec((sign * direction.X(), sign * direction.Y(), sign * direction.Z()))
        if shape.kind is TopoKind.FACE:
            curves = [
                c
                for w in self.sub_shape_uses(shape, TopoKind.WIRE)
                for c in self._loop_curves(w)
            ]
        elif shape.kind is TopoKind.WIRE and self.is_closed(shape):
            curves = self._loop_curves(shape)
        elif shape.kind in (TopoKind.EDGE, TopoKind.WIRE):
            # an open profile sweeps a sheet
            wire = shape if shape.kind is TopoKind.WIRE else SmHandle(TopoKind.WIRE, members=[shape])
            faces: list[SmHandle] = []
            for curve in self._loop_curves(wire):
                # the kernel sweeps one open curve at a time into a sheet
                sheet = self._primitive(
                    lambda c, curve=curve: c.CreateRotationalSweep(
                        [curve], base, axis_dir, math.degrees(angle), 1, 0, 1
                    )
                )
                faces.extend(self.sub_shapes(sheet, TopoKind.FACE))
            if len(faces) == 1:
                return faces[0]
            _, record = self.sew(faces)
            assert record.result is not None
            return record.result
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
        kinds = [section.kind for section in sections]
        if TopoKind.VERTEX in kinds:
            return self._loft_to_points(sections, solid, ruled)
        counts, curves = [], []
        for section in sections:
            loop = self._loop_curves(self._as_wire(section))
            counts.append(len(loop))
            curves.extend(loop)
        return self._primitive(
            lambda c: c.CreateSkinPrimitive(
                counts, curves, 0.0, 1 if ruled else 3, 1 if solid else 0, [], [], []
            )
        )

    @staticmethod
    def _as_wire(section: SmHandle) -> SmHandle:
        return (
            section
            if section.kind is TopoKind.WIRE
            else SmHandle(TopoKind.WIRE, members=[section])
        )

    def _apex_faces(self, wire: SmHandle, apex: Point) -> list[SmHandle]:
        """Faces ruled from each edge of a loop to one point."""
        body = self._new_brep()
        tip = self._nurbs([apex, apex], [0.0, 0.0, 1.0, 1.0], 1, None)
        for curve in self._loop_curves(wire):
            status, surface = self.sc.SmBSplineSurface.CreateRuledSurface(
                self.context, self._as_nurbs(curve), tip, self.sc.SM_SP_V
            )
            _check(status, "ruled surface")
            status, _ = body.CreateFaceFromSurface(surface, surface.GetNaturalUVDomain())
            _check(status, "face from surface")
        return [SmHandle(TopoKind.FACE, f) for f in self._entities(body, TopoKind.FACE)]

    def _loft_to_points(
        self, sections: Sequence[SmHandle], solid: bool, ruled: bool
    ) -> SmHandle:
        """A loft whose first and/or last section is a vertex: the loops are
        skinned, the ends ruled to the points, and all of it sewn."""
        if any(s.kind is TopoKind.VERTEX for s in sections[1:-1]) or all(
            s.kind is TopoKind.VERTEX for s in sections
        ):
            raise NotImplementedError("smlib: loft through a vertex")
        faces: list[SmHandle] = []
        wires = [self._as_wire(s) for s in sections if s.kind is not TopoKind.VERTEX]
        if len(wires) > 1:
            counts, curves = [], []
            for wire in wires:
                loop = self._loop_curves(wire)
                counts.append(len(loop))
                curves.extend(loop)
            skin = self._primitive(
                lambda c: c.CreateSkinPrimitive(
                    counts, curves, 0.0, 1 if ruled else 3, 0, [], [], []
                )
            )
            faces.extend(self.sub_shapes(skin, TopoKind.FACE))
        for end, wire in ((sections[0], wires[0]), (sections[-1], wires[-1])):
            if end.kind is TopoKind.VERTEX:
                faces.extend(self._apex_faces(wire, self.vertex_point(end)))
            elif solid:
                faces.append(self.make_planar_face(wire, []))
        _, record = self.sew(faces)
        result = record.result
        assert result is not None
        if solid and result.kind is TopoKind.SHELL:
            return self.solid_from_shell(result)
        return result

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
        # the kernel's chamfer is set by the distance between its two rails;
        # OCCT's by the setback along each face, which the angle between the
        # faces at the edge converts (measured before the copy is worked on)
        distances = []
        for edge, face, d1, d2 in specs:
            angle = self._dihedral_angle(edge)
            distances.append(math.sqrt(max(0.0, d1 * d1 + d2 * d2 - 2 * d1 * d2 * math.cos(angle))))
        if any(d1 != d2 for _, _, d1, d2 in specs):
            raise NotImplementedError("smlib: asymmetric chamfers")
        body = self._tracked_copy(shape, record)
        working_edges = [record.modified[e.key()][0].entity for e, _, _, _ in specs]
        executive = self.sc.SmFilletExecutive(self.context, body)
        executive.SetDoGlobalMerge(1)
        _check(
            executive.SetFilletParameters(
                body,
                working_edges,
                [self.sc.SM_FSG_LINEAR] * len(working_edges),
                [self.sc.SM_FS_CONST_DIST] * len(working_edges),
                distances,
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

    def _dihedral_angle(self, edge: SmHandle) -> float:
        """The angle inside the body between the two faces along an edge,
        measured at its middle (pi/2 for the edge of a box)."""
        faces = [SmHandle(TopoKind.FACE, f) for f in edge.entity.GetFaces([], None)]
        if len(faces) != 2:
            return math.pi / 2
        view = SmCurveView(self, edge)
        middle, tangent = view.d1(0.5 * (view.first + view.last))
        tangent = _unit(tangent)
        step = max(_TOLERANCE * 100, view.length() * 1e-3)
        inward = []
        for face in faces:
            surface = SmSurfaceView(self, face)
            _, normal = surface.normal(*surface.closest_uv(middle))
            across = _unit(_cross(normal, tangent))
            probe = tuple(m + step * a for m, a in zip(middle, across))
            if not self.point_on_face(face, probe, step * 0.5):
                across = tuple(-a for a in across)  # type: ignore[assignment]
            inward.append(across)
        cosine = max(-1.0, min(1.0, _dot(inward[0], inward[1])))
        return math.acos(cosine)

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
        if not openings:
            # with nothing opened the thick solid is the offset solid itself
            status, result = self.sc.SmOffsetExecutive.OffsetBrep(
                self.context, body, thickness, 1 if join is JoinKind.INTERSECTION else 0, 1
            )
            _check(status, "offset")
            return self._solid_handle(result)
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
            # other sheets by the kernel's shelling with nothing opened,
            # which offsets the faces and walls up their rims
            if shape.kind is TopoKind.FACE and self.planar_surface(shape) is not None:
                _, normal = SmSurfaceView(self, shape).normal(
                    *SmSurfaceView(self, shape)._mid()  # pylint: disable=protected-access
                )
                return self.extrude(shape, tuple(amount * c for c in normal))  # type: ignore[arg-type]
            if shape.kind is TopoKind.FACE:
                sheet, _ = self._copy_body(self._sheet_of(shape))
            else:
                sheet, _ = self._copy_body(self._body(shape))
            status, result = self.sc.SmOffsetExecutive.ShellBrep(
                self.context, sheet, amount, 1, 1, 1, []
            )
            _check(status, "thicken")
            return self._solid_handle(result)
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
        """The degree domain of an arc given in radians: the counter-clockwise
        sweep from ``start`` to ``end`` either way (OCCT's arc of the other
        sense is this one run backwards, which the caller's flip does)."""
        del sense
        a1, a2 = math.degrees(start), math.degrees(end)
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
        return self._arc_edge(self._circle(frame, radius, a1, a2), sense)

    def _arc_edge(self, curve: Any, sense: bool) -> SmHandle:
        """A forward edge over a whole arc; the arc of the other sense is the
        curve itself run backwards (as OCCT's), not a reversed edge."""
        if sense:
            return self._edge_on(curve)
        domain = curve.GetNaturalInterval()
        reversed_domain = self.sc.SmExtent1d()
        _check(curve.ReverseParameterization(domain, reversed_domain), "curve reverse")
        return self._free_edge(curve, reversed_domain.GetMin(), reversed_domain.GetMax())

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
        return self._arc_edge(self._ellipse(frame, major, minor, rotation, a1, a2), sense)

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
            # tangents at every point; the ones not given follow the chords
            tangents = []
            for i, given in enumerate(point_tangents):
                if given is not None:
                    tangents.append(self._vec(given))
                    continue
                before = points[i - 1] if i > 0 else points[i]
                after = points[i + 1] if i + 1 < len(points) else points[i]
                tangents.append(self._vec(_unit([b - a for a, b in zip(before, after)])))
            status, curve = self.sc.SmBSplineCurve.CreateInterpolatingCurve(
                self.context,
                self.sc.SM_CP_CHORDLENGTH,
                3,
                min(3, max(1, len(points) - 1)) if len(points) < 4 else 3,
                [self._vec(p) for p in points],
                tangents,
                None,
                1 if scale else 0,
            )
            _check(status, "interpolation with tangents")
            return self._edge_on(curve)
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

    def make_helix(
        self,
        center: Point,
        normal: Point,
        radius: float,
        pitch: float,
        height: float,
        cone_angle: float,
        lefthand: bool,
    ) -> SmHandle:
        frame = gp_Ax3(gp_Pnt(*center), gp_Dir(*normal))
        placement = self._placement(frame.Ax2())
        turns = height / pitch
        end_radius = radius + height * math.tan(cone_angle)
        status, helix = self.sc.SmBSplineCurve.CreateHelixSegment(
            self.context,
            placement,
            height,
            radius,
            end_radius,
            turns,
            0 if lefthand else 1,
            self.sc.SmApproxTol3d(_TOLERANCE / 10),
        )
        _check(status, "helix")
        return self._edge_on(helix)

    def make_wire(
        self, shapes: Sequence[SmHandle]
    ) -> tuple[SmHandle | None, WireStatus]:
        edges = [
            e for s in shapes for e in (s.members if s.kind is TopoKind.WIRE else [s])
        ]
        if not edges:
            return None, WireStatus.EMPTY
        # like OCCT's wire builder, the edges are chained end to end whatever
        # order and way round they were given in
        chained = self.connect_edges_to_wires(edges, _TOLERANCE)
        if len(chained) != 1:
            return None, WireStatus.DISCONNECTED
        return chained[0], WireStatus.OK

    def make_wire_placed(
        self, edges: Sequence[SmHandle]
    ) -> tuple[SmHandle, list[SmHandle]]:
        wire, _ = self.make_wire(edges)
        assert wire is not None
        return wire, list(wire.members)

    def make_planar_face(self, outer: SmHandle, inners: Sequence[SmHandle]) -> SmHandle:
        if not self.is_closed(outer):
            raise ValueError("outer wire is not closed")
        for inner in inners:
            if not self.is_closed(inner):
                raise ValueError("inner wire is not closed")
        curves = self._loop_curves(outer)
        for inner in inners:
            curves.extend(self._loop_curves(inner))
        brep = self._new_brep()
        status, faces = brep.CreatePlanarFacesWith3DCurves(
            brep.GetInfiniteRegion(), curves, _TOLERANCE, []
        )
        if status != SM_SUCCESS:
            raise ValueError(f"Cannot build face(s): SMLib planar face failed with status {status}")
        if len(faces) != 1:
            raise ValueError(f"Cannot build face(s): the wires gave {len(faces)} faces")
        return SmHandle(TopoKind.FACE, faces[0])

    def add_holes(self, face: SmHandle, wires: Sequence[SmHandle]) -> SmHandle:
        """A planar face rebuilt with extra inner loops."""
        if self.planar_surface(face) is None:
            raise NotImplementedError("smlib: holes in a curved face")
        loops = self.sub_shape_uses(face, TopoKind.WIRE)
        return self.make_planar_face(loops[0], [*loops[1:], *wires])

    def face_of_plane(
        self, plane: Any, bounds: tuple[float, float, float, float] | None = None
    ) -> SmHandle:
        if bounds is None:
            # SMLib has no unbounded faces; a very large one stands in
            bounds = (-_INFINITE, _INFINITE, -_INFINITE, _INFINITE)
        u0, u1, v0, v1 = bounds
        if u1 - u0 < _TOLERANCE:  # a face of no extent (OCCT allows one) gets some
            u0, u1 = u0 - 0.5, u1 + 0.5
        if v1 - v0 < _TOLERANCE:
            v0, v1 = v0 - 0.5, v1 + 0.5
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
        for edge in self._entities(target, TopoKind.EDGE):
            if edge.GetEdgeuseCount() > 4:
                raise ValueError("more than two faces meet on an edge")
        pieces = self._components(target)
        if len(pieces) == 1:
            record.result = self._sheet_handle(target)
        else:
            parts = []
            for faces in pieces:
                piece = self._new_brep()
                _check(target.CopyFaces(faces, piece, None, 0, None), "sheet copy")
                parts.append(self._sheet_handle(piece))
            record.result = SmHandle(TopoKind.COMPOUND, members=parts)
        return record.result, record

    def _sheet_handle(self, brep: Any) -> SmHandle:
        """A face handle for a one-face sheet, else a shell handle."""
        faces = self._entities(brep, TopoKind.FACE)
        if len(faces) == 1:
            return SmHandle(TopoKind.FACE, faces[0])
        return SmHandle(TopoKind.SHELL, brep)

    def _components(self, brep: Any) -> list[list[Any]]:
        """The faces of a brep grouped by connection through its shells."""
        faces = self._entities(brep, TopoKind.FACE)
        index = {f.address: i for i, f in enumerate(faces)}
        parent = list(range(len(faces)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # the kernel's shells may hold faces that share no edge, so the
        # grouping follows the edges themselves
        for edge in self._entities(brep, TopoKind.EDGE):
            members = [index[f.address] for f in edge.GetFaces([], None) if f.address in index]
            for i in members[1:]:
                a, b = find(members[0]), find(i)
                if a != b:
                    parent[b] = a
        groups: dict[int, list[Any]] = {}
        for i, face in enumerate(faces):
            groups.setdefault(find(i), []).append(face)
        return list(groups.values())

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
        if copy:
            # like OCCT's copying transform, the geometry moves and the
            # placement the shape remembers is left as it was
            _restore_locations(moved, shape)
        return moved, record

    def transformed_general(self, shape: SmHandle, transform: Any) -> SmHandle:
        moved = self._bake(shape, gp_Trsf(), general=transform)
        _restore_locations(moved, shape)
        return moved

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
        if solid.kind is TopoKind.COMPOUND:
            return any(self.point_in_solid(m, point, tolerance) for m in solid.members)
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

    def _sheet_of(self, face: SmHandle) -> Any:
        """A body holding a copy of this one face (cached per face)."""
        key = face.entity.address
        sheet = self._sheets.get(key)
        if sheet is None or sheet[0] is not face.entity:
            body = self._new_brep()
            _check(
                face.entity.GetBrep().CopyFaces([face.entity], body, None, 0, None),
                "face copy",
            )
            sheet = (face.entity, body)
            self._sheets[key] = sheet
        return sheet[1]

    def point_on_face(self, face: SmHandle, point: Point, tolerance: float) -> bool:
        zone = self.sc.SmZoneTol3d(tolerance)
        classification = self.sc.SmPointClassification(zone, None)
        _check(
            self._sheet_of(face).Point3DClassify(self._vec(point), zone, 1, classification),
            "point classification",
        )
        return classification.GetPointClass() != self.sc.SM_PC_REGION

    def line_intersections(
        self, shape: SmHandle, origin: Point, direction: Point, tolerance: float
    ) -> list[tuple[SmHandle, Point]]:
        line = self.make_infinite_line(origin, direction)
        interval = self.sc.SmExtent1d(*line.interval)
        found: list[tuple[SmHandle, Point]] = []
        faces = (
            [shape] if shape.kind is TopoKind.FACE else self.sub_shapes(shape, TopoKind.FACE)
        )
        for face in faces:
            for point in self._crossings(line.curve, interval, ("body", self._sheet_of(face))):
                found.append((face, point))
        return found

    def find_plane(self, shape: SmHandle) -> tuple[Point, Point, Point] | None:
        """The plane all of a shape's edges lie in, if there is one."""
        edges = (
            [shape] if shape.kind is TopoKind.EDGE else self.sub_shapes(shape, TopoKind.EDGE)
        )
        points: list[Point] = []
        x_dir: Point | None = None
        for edge in edges:
            view = SmCurveView(self, edge)
            if x_dir is None:
                x_dir = _unit(view.d1(0.5 * (view.first + view.last))[1])
            points.extend(
                view.value(view.first + (view.last - view.first) * i / 6) for i in range(7)
            )
        if not points:
            return None
        origin = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
        first = max(points, key=lambda p: math.dist(p, origin))
        a = tuple(c - o for c, o in zip(first, origin))
        if math.sqrt(_dot(a, a)) < _TOLERANCE:
            return None
        # the normal follows the winding of the points (Newell), the way
        # OCCT's plane finder orients it; a chain that encloses no area is
        # given the normal its spread defines
        nx = ny = nz = 0.0
        for p, q in zip(points, points[1:] + points[:1]):
            nx += (p[1] - q[1]) * (p[2] + q[2])
            ny += (p[2] - q[2]) * (p[0] + q[0])
            nz += (p[0] - q[0]) * (p[1] + q[1])
        normal: Point | None = (nx, ny, nz)
        best = math.sqrt(_dot(normal, normal))
        if best < _TOLERANCE * _dot(a, a):
            normal, best = None, 0.0
            for p in points:
                b = tuple(c - o for c, o in zip(p, origin))
                n = _cross(a, b)
                size = math.sqrt(_dot(n, n))
                if size > best:
                    normal, best = n, size
            if normal is None or best < _TOLERANCE * math.sqrt(_dot(a, a)):
                return None  # all the points are in a line
        normal = _unit(normal)
        scale = max(1.0, max(math.dist(p, origin) for p in points))
        for p in points:
            if abs(_dot(tuple(c - o for c, o in zip(p, origin)), normal)) > _TOLERANCE * scale:
                return None
        assert x_dir is not None
        tangent = tuple(c - _dot(x_dir, normal) * n for c, n in zip(x_dir, normal))
        if math.sqrt(_dot(tangent, tangent)) < _TOLERANCE:
            tangent = _unit(a)
        return origin, normal, _unit(tangent)  # type: ignore[return-value]

    def continuity_between(
        self, edge1: SmHandle, edge2: SmHandle, u1: float, u2: float, tolerance: float
    ) -> str:
        a = SmCurveView(self, edge1)._derivatives(u1, 2)  # pylint: disable=protected-access
        b = SmCurveView(self, edge2)._derivatives(u2, 2)  # pylint: disable=protected-access
        if math.dist(a[0], b[0]) > tolerance:
            return "C0"
        ta, tb = a[1], b[1]
        la, lb = math.sqrt(_dot(ta, ta)), math.sqrt(_dot(tb, tb))
        if la < tolerance or lb < tolerance:
            return "C0"
        if math.dist(_unit(ta), _unit(tb)) > tolerance:
            return "C0"
        label = "G1"
        if math.dist(ta, tb) <= tolerance:
            label = "C1"
            if math.dist(a[2], b[2]) <= tolerance:
                return "C2"
        # the same curvature vector makes the join G2
        def curvature(d1: Point, d2: Point) -> Point:
            speed2 = _dot(d1, d1)
            along = _dot(d1, d2) / speed2
            return tuple((c2 - along * c1) / speed2 for c1, c2 in zip(d1, d2))  # type: ignore[return-value]
        if math.dist(curvature(ta, a[2]), curvature(tb, b[2])) <= tolerance:
            return "G2" if label == "G1" else "C1"
        return label

    def wire_as_bspline_edge(self, wire: SmHandle, tolerance: float) -> SmHandle:
        return self._edge_on(self._joined_curve(wire))

    def planar_intersections(
        self,
        edge: SmHandle,
        bounds: tuple[float, float],
        other: SmHandle | None,
        other_bounds: tuple[float, float] | None,
        plane: gp_Pln,
        tolerance: float,
    ) -> list[tuple[float, float]]:
        if other is None:
            return []  # a curve crossing itself is not looked for
        view, other_view = SmCurveView(self, edge), SmCurveView(self, other)
        assert other_bounds is not None
        points = self._crossings(
            view.curve,
            self.sc.SmExtent1d(*bounds),
            ("curve", (other_view.curve, self.sc.SmExtent1d(*other_bounds))),
        )
        origin = plane.Location()
        x_axis, y_axis = plane.XAxis().Direction(), plane.YAxis().Direction()
        found = []
        for p in points:
            d = (p[0] - origin.X(), p[1] - origin.Y(), p[2] - origin.Z())
            found.append(
                (
                    d[0] * x_axis.X() + d[1] * x_axis.Y() + d[2] * x_axis.Z(),
                    d[0] * y_axis.X() + d[1] * y_axis.Y() + d[2] * y_axis.Z(),
                )
            )
        return found
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

    def wire_edges(self, wire: SmHandle) -> list[SmHandle]:
        return list(wire.members)

    build_curves_3d = _unsupported("build_curves_3d")
    def nearest_edge(self, shape: SmHandle, point: Point) -> tuple[SmHandle, float] | None:
        best: tuple[float, SmHandle, float] | None = None
        for edge in self.sub_shapes(shape, TopoKind.EDGE):
            param, distance = SmCurveView(self, edge).drop_point(point)
            if best is None or distance < best[0]:
                best = (distance, edge, param)
        if best is None:
            return None
        return best[1], best[2]
    def edge_uv_on_face(
        self, edge: SmHandle, face: SmHandle, params: Sequence[float]
    ) -> list[tuple[float, float]]:
        curve = SmCurveView(self, edge)
        surface = SmSurfaceView(self, face)
        return [surface.closest_uv(curve.value(t)) for t in params]

    def vertex_uv_on_face(self, vertex: SmHandle, face: SmHandle) -> tuple[float, float]:
        return SmSurfaceView(self, face).closest_uv(self.vertex_point(vertex))
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
            # a straight stretch: the perpendicular OCCT's frame picks there
            normal = list(_perpendicular(tangent))
        normal_unit = _unit(normal)
        # the analytic curves evaluate with rounding noise where OCCT's are
        # exact; trimmed off so that the frame's Euler angles come out the same
        tangent = _unit(tuple(0.0 if abs(c) < 1e-12 else c for c in tangent))
        normal_unit = _unit(tuple(0.0 if abs(c) < 1e-12 else c for c in normal_unit))
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
    def chamfer_2d(
        self, face: SmHandle, corners: Sequence[tuple[SmHandle, SmHandle, float, float]]
    ) -> tuple[SmHandle, Any]:
        """The planar face with corners cut off: each loop is rebuilt from
        free copies of its edges, shortened at the corners and bridged."""
        loops = []
        wire_given = face.kind is TopoKind.WIRE
        source_loops = [face] if wire_given else self.sub_shape_uses(face, TopoKind.WIRE)
        for loop in source_loops:
            originals = list(loop.members)
            edges: list[SmHandle | None] = []
            for member in originals:
                view = SmCurveView(self, member)
                edges.append(
                    self._free_edge(
                        self._copy_curve(view.curve), view.first, view.last, member.flip
                    )
                )
            bridges: dict[int, SmHandle] = {}
            count = len(originals)
            for first, second, d1, d2 in corners:
                indices = [
                    k for k, member in enumerate(originals)
                    if self.is_same(member, first) or self.is_same(member, second)
                ]
                if len(indices) != 2:
                    continue
                i, j = indices
                closing = i == 0 and j == count - 1 and count > 2 and self.is_closed(loop)
                if not (j == i + 1 or closing):
                    continue
                if closing:
                    i, j = j, i  # the loop closes from the last edge to the first
                along = {}
                for k, member in ((i, originals[i]), (j, originals[j])):
                    along[k] = d1 if self.is_same(member, first) else d2
                if edges[i] is None or edges[j] is None:
                    continue
                shortened_i = self._shorten(edges[i], along[i], at_end=True)
                shortened_j = self._shorten(edges[j], along[j], at_end=False)
                if shortened_i is None or shortened_j is None:
                    continue
                edges[i], edges[j] = shortened_i, shortened_j
                _, p1, _, _ = self._oriented_ends(shortened_i)
                p2, _, _, _ = self._oriented_ends(shortened_j)
                bridges[i] = self.make_line(p1, p2)
            rebuilt: list[SmHandle] = []
            for k, edge in enumerate(edges):
                if edge is not None:
                    rebuilt.append(edge)
                if k in bridges:
                    rebuilt.append(bridges[k])
            loops.append(SmHandle(TopoKind.WIRE, members=rebuilt))
        if wire_given:
            return loops[0], _Record()
        return self.make_planar_face(loops[0], loops[1:]), _Record()

    def _shorten(self, edge: SmHandle, length: float, at_end: bool) -> SmHandle | None:
        """An edge with ``length`` taken off its oriented end (or start)."""
        view = SmCurveView(self, edge)
        total = view.length()
        if length <= 0 or length >= total - _TOLERANCE:
            return None
        # the distance from the curve's own first parameter to the cut
        from_first = (total - length) if (at_end != edge.flip) else length
        param = view.param_at_length(from_first, view.first)
        return self._cut_keeping(edge, view.value(param), keep_start=at_end)
    fix_small_edges = _unsupported("fix_small_edges")
    def project(
        self,
        shape: SmHandle,
        target: SmHandle,
        *,
        direction: Point | None = None,
        center: Point | None = None,
    ) -> list[SmHandle]:
        """Wires where the edges of ``shape``, swept along ``direction`` both
        ways, cut the target's faces (the projection onto those faces)."""
        if direction is None:
            raise NotImplementedError("smlib: projection from a point")
        unit = _unit(direction)
        boxes = [b for b in (self.bounding_box(shape, False), self.bounding_box(target, False)) if b is not None]
        reach = max(
            math.dist(b[:3], b[3:]) for b in boxes
        ) + max(math.dist(a[:3], b[3:]) for a in boxes for b in boxes)
        edges = (
            [shape] if shape.kind is TopoKind.EDGE else self.sub_shapes(shape, TopoKind.EDGE)
        )
        back = _translation(tuple(-reach * c for c in unit))
        found: list[SmHandle] = []
        faces = (
            [target] if target.kind is TopoKind.FACE else self.sub_shapes(target, TopoKind.FACE)
        )
        for edge in edges:
            moved = self._bake(edge, back)
            sheet = self.extrude(moved, tuple(2 * reach * c for c in unit))  # type: ignore[arg-type]
            sheet_body = SmHandle(TopoKind.SHELL, self._body(sheet))
            for face in faces:
                one_face = SmHandle(TopoKind.SHELL, self._sheet_of(face))
                found.extend(self._section_edges([sheet_body], [one_face]).members)
        return self.connect_edges_to_wires(found, _TOLERANCE)
    # ---- 2D offsets -----------------------------------------------------------

    def _wire_normal(self, edges: Sequence[SmHandle]) -> Point:
        """The normal of the plane a wire lies in (Newell's method over sampled points)."""
        points: list[Point] = []
        for edge in edges:
            points.extend(SmCurveView(self, edge).samples(32))
        nx = ny = nz = 0.0
        for a, b in zip(points, points[1:] + points[:1]):
            nx += (a[1] - b[1]) * (a[2] + b[2])
            ny += (a[2] - b[2]) * (a[0] + b[0])
            nz += (a[0] - b[0]) * (a[1] + b[1])
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-12:
            raise ValueError("the wire has no plane to be offset in")
        return (nx / length, ny / length, nz / length)

    def _oriented_ends(self, edge: SmHandle) -> tuple[Point, Point, Point, Point]:
        """Start, end, start tangent and end tangent of an edge the way it runs."""
        view = SmCurveView(self, edge)
        p0, t0 = view.d1(view.first)
        p1, t1 = view.d1(view.last)
        if edge.flip:
            return p1, p0, tuple(-c for c in t1), tuple(-c for c in t0)  # type: ignore[return-value]
        return p0, p1, t0, t1

    def _offset_edge(self, edge: SmHandle, normal: Point, distance: float) -> SmHandle | None:
        """The edge offset by ``distance`` to its right (looking along it with
        ``normal`` up); None when the offset collapses it. Lines stay lines and
        circles stay circles."""
        view = SmCurveView(self, edge)
        # the offset side is to the right of the way the edge runs
        side = -distance if edge.flip else distance
        if view.geom_type == "LINE":
            p0, p1 = view.value(view.first), view.value(view.last)
            tangent = _unit([b - a for a, b in zip(p0, p1)])
            right = _cross(tangent, normal)
            shift = tuple(side * c for c in right)
            moved = self.make_line(
                tuple(a + b for a, b in zip(p0, shift)),  # type: ignore[arg-type]
                tuple(a + b for a, b in zip(p1, shift)),  # type: ignore[arg-type]
            )
            return moved.with_flip(edge.flip)
        if view.geom_type == "CIRCLE":
            centre, _, radius = view.circle()
            # the right of a circle running counter-clockwise about the normal
            # is its outside; the way it runs comes from its own parameter
            # direction, not from its frame
            start, tangent = view.d1(view.first)
            radial = [p - c for p, c in zip(start, centre)]
            counter_clockwise = _dot(_cross(radial, tangent), normal) > 0
            outward = side if counter_clockwise else -side
            new_radius = radius + outward
            if new_radius <= _TOLERANCE:
                return None
            origin, x_axis, y_axis, _ = view._frame()  # pylint: disable=protected-access

            def angle_at(t: float) -> float:
                """The angle on the circle of the point at parameter ``t``."""
                relative = [p - c for p, c in zip(view.value(t), centre)]
                return math.degrees(math.atan2(_dot(relative, y_axis), _dot(relative, x_axis)))

            a1, a2 = angle_at(view.first), angle_at(view.last)
            if view.is_closed():
                a2 = a1 + 360.0
            while a2 <= a1 + 1e-9:
                a2 += 360.0
            circle = self.sc.SmCircle(
                self._vec(origin),
                self._vec(x_axis),
                self._vec(y_axis),
                self.sc.SmExtent1d(a1, a2),
                new_radius,
                3,
                self.context,
                None,
                0,
            )
            return self._edge_on(circle, flip=edge.flip)
        offset = self.sc.SmOffsetCurve(
            3,
            self._copy_curve(view.curve),
            self.sc.SmExtent1d(view.first, view.last),
            self._vec(normal),
            side,
            1,
        )
        status, nurbs = offset.CreateNonRationalNurb(self.context, _TOLERANCE / 10)
        _check(status, "offset curve")
        return self._edge_on(nurbs, flip=edge.flip)

    def _line_crossing(self, a: SmHandle, b: SmHandle, normal: Point) -> Point | None:
        """Where the lines of two straight edges cross in their plane."""
        p, _, u, _ = self._oriented_ends(a)
        q, _, v, _ = self._oriented_ends(b)
        u, v = _unit(u), _unit(v)
        denominator = _dot(_cross(u, v), normal)
        if abs(denominator) < 1e-12:
            return None
        w = [b_ - a_ for a_, b_ in zip(p, q)]
        along = _dot(_cross(w, v), normal) / denominator
        return tuple(a_ + along * b_ for a_, b_ in zip(p, u))  # type: ignore[return-value]

    def _cut_keeping(self, edge: SmHandle, point: Point, keep_start: bool) -> SmHandle | None:
        """The part of an edge on the start side (or the end side) of ``point``."""
        view = SmCurveView(self, edge)
        param, _ = view.drop_point(point)
        if not (view.first < param < view.last):
            return None
        from_start = keep_start != edge.flip  # keep the curve's own first part
        t0, t1 = (view.first, param) if from_start else (param, view.last)
        if t1 - t0 <= _TOLERANCE:
            return None
        piece = self.trim_edge(edge, t0, t1)
        piece.flip = edge.flip
        return piece

    def _join_offsets(
        self,
        first: SmHandle,
        second: SmHandle,
        corner: Point,
        turn_opens: bool,
        join: JoinKind,
        normal: Point,
        distance: float,
    ) -> tuple[SmHandle | None, SmHandle | None, SmHandle | None]:
        """Trim or connect two consecutive offset edges at a corner; returns
        the adjusted first and second edges and an edge to put between them."""
        _, p1, _, _ = self._oriented_ends(first)
        p0, _, _, _ = self._oriented_ends(second)
        if math.dist(p1, p0) <= _TOLERANCE:
            return first, second, None
        straight = (
            SmCurveView(self, first).geom_type == "LINE"
            and SmCurveView(self, second).geom_type == "LINE"
        )
        if turn_opens and join is JoinKind.ARC:
            towards = _unit([(a + b) / 2 - c for a, b, c in zip(p1, p0, corner)])
            midway = tuple(c + abs(distance) * t for c, t in zip(corner, towards))
            return first, second, self.make_three_point_arc(p1, midway, p0)  # type: ignore[arg-type]
        if straight:
            crossing = self._line_crossing(first, second, normal)
            if crossing is None:
                return first, second, self.make_line(p1, p0)
            a0, a1, _, _ = self._oriented_ends(first)
            b0, b1, _, _ = self._oriented_ends(second)
            # where the crossing falls along each line: before the start of the
            # first or past the end of the second means that line is used up
            along_first = _dot([c - a for a, c in zip(a0, crossing)], _unit([b - a for a, b in zip(a0, a1)]))
            along_second = _dot([b - c for b, c in zip(b1, crossing)], _unit([b - a for a, b in zip(b0, b1)]))
            first_cut = (
                self.make_line(a0, crossing).with_flip(False) if along_first > _TOLERANCE else _DROPPED
            )
            second_cut = (
                self.make_line(crossing, b1).with_flip(False) if along_second > _TOLERANCE else _DROPPED
            )
            return first_cut, second_cut, None
        if not turn_opens:
            view_first = SmCurveView(self, first)
            crossings = self._crossings(
                view_first.curve,
                self.sc.SmExtent1d(view_first.first, view_first.last),
                self._atoms(second)[0],
            )
            if crossings:
                crossing = min(crossings, key=lambda c: math.dist(c, corner))
                cut_first = self._cut_keeping(first, crossing, True)
                cut_second = self._cut_keeping(second, crossing, False)
                if cut_first is not None and cut_second is not None:
                    return cut_first, cut_second, None
        # whatever could not be joined better is bridged
        return first, second, self.make_line(p1, p0)

    def _offset_run(
        self,
        edges: Sequence[SmHandle],
        normal: Point,
        distance: float,
        join: JoinKind,
        closed: bool,
    ) -> list[SmHandle]:
        """The edges offset to their right and joined at the corners."""
        offsets = []
        for edge in edges:
            moved = self._offset_edge(edge, normal, distance)
            if moved is not None:
                offsets.append((edge, moved))
        if not offsets:
            return []
        result: list[SmHandle | None] = [moved for _, moved in offsets]
        pairs = list(range(len(offsets) - 1))
        if closed and len(offsets) > 1:
            pairs.append(len(offsets) - 1)
        inserted: dict[int, SmHandle] = {}
        for i in pairs:
            j = (i + 1) % len(offsets)
            base_first, base_second = offsets[i][0], offsets[j][0]
            _, corner, _, t_end = self._oriented_ends(base_first)
            _, _, t_start, _ = self._oriented_ends(base_second)
            turn = _dot(_cross(t_end, t_start), normal)
            # a left turn opens a gap on the right side, the offset side
            turn_opens = turn > 0 if distance > 0 else turn < 0
            if result[i] is None or result[j] is None:
                continue
            first, second, between = self._join_offsets(
                result[i], result[j], corner, turn_opens, join, normal, distance
            )
            if first is not None:
                result[i] = None if first is _DROPPED else first
            if second is not None:
                result[j] = None if second is _DROPPED else second
            if between is not None:
                inserted[i] = between
        ordered: list[SmHandle] = []
        for i, edge in enumerate(result):
            if edge is not None:
                ordered.append(edge)
            if i in inserted:
                ordered.append(inserted[i])
        survivors = [e for e in ordered if self.length(e) > _TOLERANCE]
        bridged: list[SmHandle] = []
        for edge in survivors:
            if bridged:
                _, previous_end, _, _ = self._oriented_ends(bridged[-1])
                next_start, _, _, _ = self._oriented_ends(edge)
                if math.dist(previous_end, next_start) > _TOLERANCE:
                    bridged.append(self.make_line(previous_end, next_start))
            bridged.append(edge)
        return bridged

    def offset_2d(
        self,
        wire: SmHandle,
        distance: float,
        join: JoinKind,
        plane: Any = None,
    ) -> SmHandle | None:
        edges = list(wire.members) if wire.kind is TopoKind.WIRE else [wire]
        if plane is not None:
            direction = plane.Axis().Direction()
            normal: Point = (direction.X(), direction.Y(), direction.Z())
        else:
            normal = self._wire_normal(edges)
        if self.is_closed(wire):
            # outwards for a positive distance: the right of a loop running
            # counter-clockwise about the normal, the left of one running clockwise
            points = [
                p
                for e in edges
                for p in SmCurveView(self, e).samples(32)[:-1]
            ]
            if not edges[0].flip and len(edges) == 1:
                pass
            twice_area = 0.0
            for a, b in zip(points, points[1:] + points[:1]):
                twice_area += _dot(_cross(a, b), normal)
            # the sampled points follow each edge's own parameter direction;
            # the first edge's flip tells which way the wire runs
            runs_backwards = edges[0].flip
            counter_clockwise = (twice_area > 0) != runs_backwards
            signed = distance if counter_clockwise else -distance
            run = self._offset_run(edges, normal, signed, join, True)
            return SmHandle(TopoKind.WIRE, members=run) if run else None
        right = self._offset_run(edges, normal, abs(distance), join, False)
        backwards = [e.with_flip(not e.flip) for e in reversed(edges)]
        left = self._offset_run(backwards, normal, abs(distance), join, False)
        loop: list[SmHandle] = []
        start, _, _, _ = self._oriented_ends(edges[0])
        _, end, _, _ = self._oriented_ends(edges[-1])
        if right and left:
            loop.extend(right)
            loop.append(self._end_cap(right[-1], left[0], end, abs(distance)))
            loop.extend(left)
            loop.append(self._end_cap(left[-1], right[0], start, abs(distance)))
        elif right or left:
            # one side collapsed entirely (an inside offset beyond the bends):
            # half circles round each end meet, or are joined, on that side
            run = right or left
            _, _, forward_start, _ = self._oriented_ends(edges[0])
            _, _, _, forward_end = self._oriented_ends(edges[-1])
            _, p_end, _, _ = self._oriented_ends(run[-1])
            p_start, _, _, _ = self._oriented_ends(run[0])
            far_end = tuple(2 * c - p for c, p in zip(end, p_end))
            far_start = tuple(2 * c - p for c, p in zip(start, p_start))
            loop.extend(run)
            ahead = _unit(forward_end)
            loop.append(
                self.make_three_point_arc(
                    p_end, tuple(c + abs(distance) * a for c, a in zip(end, ahead)), far_end
                )
            )
            if math.dist(far_end, far_start) > _TOLERANCE:
                loop.append(self.make_line(far_end, far_start))
            behind = _unit(forward_start)
            loop.append(
                self.make_three_point_arc(
                    far_start,
                    tuple(c - abs(distance) * b for c, b in zip(start, behind)),
                    p_start,
                )
            )
        return SmHandle(TopoKind.WIRE, members=loop) if loop else None

    def _end_cap(self, before: SmHandle, after: SmHandle, centre: Point, radius: float) -> SmHandle:
        """The half circle about an end of an open wire from one side's offset to the other's."""
        _, p1, _, _ = self._oriented_ends(before)
        p0, _, _, _ = self._oriented_ends(after)
        _, _, _, tangent = self._oriented_ends(before)
        ahead = _unit(tangent)
        midway = tuple(c + radius * t for c, t in zip(centre, ahead))
        return self.make_three_point_arc(p1, midway, p0)  # type: ignore[arg-type]
    offset_face = _unsupported("offset_face")
    make_bezier_surface = _unsupported("make_bezier_surface")
    make_spline_surface_fit = _unsupported("make_spline_surface_fit")
    filling = _unsupported("filling")
    draft_prism = _unsupported("draft_prism")
    def extrude_generated(
        self, shape: SmHandle, direction: Point
    ) -> tuple[SmHandle, AncestorMap, Any]:
        prism = self.extrude(shape, direction)
        pull = _unit(direction)
        sides = []
        for face in self.sub_shapes(prism, TopoKind.FACE):
            plane = self.planar_surface(face)
            if plane is not None:
                normal = plane.Axis().Direction()
                if abs(_dot((normal.X(), normal.Y(), normal.Z()), pull)) > 1 - 1e-9:
                    continue  # a cap
            sides.append(face)
        entries = []
        for edge in self.sub_shapes(shape, TopoKind.EDGE):
            view = SmCurveView(self, edge)
            middle = view.value((view.first + view.last) / 2)
            generated = [
                face
                for face in sides
                if self._solve_pair(("point", middle), ("face", face.entity))[0] <= _TOLERANCE
            ]
            entries.append((edge, generated))
        return prism, AncestorMap(self, entries), _Record()

    def draft(
        self,
        shape: SmHandle,
        specs: Sequence[tuple[SmHandle, float]],
        pull: Point,
        neutral_plane: Any,
    ) -> tuple[SmHandle, Any]:
        """A draft of a prism: the prism is rebuilt as a tapered extrusion of
        its cap in the neutral plane, which SMLib offers directly, one loop at
        a time (holes are tapered on their own and cut out)."""
        if shape.kind is not TopoKind.SOLID:
            raise DraftFailure("draft", None, shape)
        direction = _unit(pull)
        origin = neutral_plane.Location()
        base = None
        height = 0.0
        for face in self.sub_shapes(shape, TopoKind.FACE):
            plane = self.planar_surface(face)
            if plane is None:
                continue
            normal = plane.Axis().Direction()
            if abs(_dot((normal.X(), normal.Y(), normal.Z()), direction)) < 1 - 1e-9:
                continue
            centre = self.center_of_mass(face, TopoKind.FACE)
            offset = _dot(
                [c - o for c, o in zip(centre, (origin.X(), origin.Y(), origin.Z()))],
                direction,
            )
            if abs(offset) <= _TOLERANCE:
                base = face
            else:
                height = offset
        if base is None or height == 0.0:
            raise DraftFailure("draft", None, shape)
        up = direction if height > 0 else tuple(-c for c in direction)
        height = abs(height)

        def angle_of(wire: SmHandle) -> float:
            """The draft angle asked of the side faces grown from a loop."""
            for edge in wire.members:
                view = SmCurveView(self, edge)
                middle = view.value((view.first + view.last) / 2)
                for side, angle in specs:
                    if self._solve_pair(("point", middle), ("face", side.entity))[0] <= _TOLERANCE:
                        return angle
            raise DraftFailure("draft", None, shape)

        def degrees(angle: float) -> float:
            """SMLib's taper angle in degrees: positive narrows, negative widens."""
            return math.degrees(angle)

        loops = self.sub_shape_uses(base, TopoKind.WIRE)
        outer_angle = angle_of(loops[0])
        outer = self._primitive(
            lambda c: c.CreateTaperExtrude(
                self._loop_curves(loops[0]), height, degrees(outer_angle), 3, self._vec(up)
            )
        )
        if outer.kind is not TopoKind.SOLID:
            raise DraftFailure("draft", None, shape)
        # sides that run into each other (thin arms widening) come back as a
        # solid that lost material: a widening taper must gain volume and a
        # narrowing one lose it, or the draft is no good
        before, after = self.volume(shape), self.volume(outer)
        if outer_angle != 0 and (after - before) * outer_angle > 0:
            raise DraftFailure("draft", None, shape)
        result = outer
        for hole in loops[1:]:
            # a draft narrows the material, so it widens a hole: the plug cut
            # out is tapered the other way
            angle = -degrees(angle_of(hole))
            # the hole's own taper runs a little past both caps so the cut is clean
            curves = self._loop_curves(hole)
            back = self._placement_of_trsf(
                _translation(tuple(-_TOLERANCE * 10 * c for c in up))
            )
            for curve in curves:
                _check(curve.Transform(back, None), "curve transform")
            plug = self._primitive(
                lambda c, curves=curves, angle=angle: c.CreateTaperExtrude(
                    curves, height + _TOLERANCE * 20, angle, 3, self._vec(up)
                )
            )
            if plug.kind is not TopoKind.SOLID:
                raise DraftFailure("draft", None, shape)
            result, _ = self.boolean(BoolOp.CUT, [result], [plug])
        if result.kind is not TopoKind.SOLID:
            raise DraftFailure("draft", None, shape)
        return result, _Record()
    split_by_edges = _unsupported("split_by_edges")
    def solid_from_shell(self, shell: SmHandle) -> SmHandle:
        brep = self._body(shell)
        if not brep.IsManifoldSolid():
            # a closed shell of stitched faces still bounds no region until
            # one is split off the infinite region at one of its faces
            brep, _ = self._copy_body(brep)
            faces = self._entities(brep, TopoKind.FACE)
            if faces:
                faceuse = faces[0].GetFaceuses()[0]
                brep.TestClosureSplitRegion(1, faceuse)
                brep.FindAndSetInfiniteRegion(None, self.sc.SM_VS_INFINITEREGION_SWAP, None)
            if not brep.IsManifoldSolid():
                raise ValueError("the shell does not close a solid")
        return SmHandle(TopoKind.SOLID, brep, partner=shell.partner)
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
