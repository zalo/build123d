"""
build123d kernel facade

name: base.py
by:   Gumyr
date: October 12, 2026

desc:
    The abstract interface build123d's topology layer uses to talk to a
    boundary-representation kernel. Everything the Python layer needs from the
    kernel goes through a ``Kernel``: identity of entities, traversal of
    sub-shapes, geometric queries and the handful of operations that produce
    new shapes. Shape handles are opaque to this layer; only the backend
    knows their type.

    Phase 1 of the facade work: the OCCT backend in ``occt.py`` is the only
    implementation and the public API of build123d is unchanged. The method
    set grows as more of ``topology/`` is routed through it.

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

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from enum import Enum
from typing import Any

Handle = Any
"""An opaque kernel shape handle (``TopoDS_Shape`` for the OCCT backend)."""


class TopoKind(Enum):
    """Topological kind of a shape handle, in dimensional order."""

    VERTEX = 0
    EDGE = 1
    WIRE = 2
    FACE = 3
    SHELL = 4
    SOLID = 5
    COMPSOLID = 6
    COMPOUND = 7

    @property
    def label(self) -> str:
        """The name build123d uses for this kind, e.g. ``"Vertex"``."""
        return _KIND_LABELS[self]

    @classmethod
    def from_label(cls, label: str) -> TopoKind:
        """The kind for a build123d name such as ``"Face"``."""
        return _LABEL_KINDS[label]


_KIND_LABELS = {
    TopoKind.VERTEX: "Vertex",
    TopoKind.EDGE: "Edge",
    TopoKind.WIRE: "Wire",
    TopoKind.FACE: "Face",
    TopoKind.SHELL: "Shell",
    TopoKind.SOLID: "Solid",
    TopoKind.COMPSOLID: "CompSolid",
    TopoKind.COMPOUND: "Compound",
}
_LABEL_KINDS = {label: kind for kind, label in _KIND_LABELS.items()}


class BoolOp(Enum):
    """The boolean operations the topology layer asks for."""

    FUSE = "fuse"
    CUT = "cut"
    COMMON = "common"
    SECTION = "section"
    SPLIT = "split"


class JoinKind(Enum):
    """How offset faces or curves meet at a convex corner."""

    ARC = "arc"
    INTERSECTION = "intersection"
    TANGENT = "tangent"


class SweepTransition(Enum):
    """How a swept profile turns a corner of its path."""

    TRANSFORMED = "transformed"
    ROUND = "round"
    RIGHT = "right"


class WireStatus(Enum):
    """What the kernel found when it chained edges into a wire."""

    OK = "ok"
    NON_MANIFOLD = "non_manifold"
    EMPTY = "empty"
    DISCONNECTED = "disconnected"


Point = tuple[float, float, float]


class DraftFailure(Exception):
    """A draft angle could not be added to a face or the draft could not be built."""

    def __init__(self, stage: str, index: int | None, problematic: Handle | None):
        super().__init__(stage)
        self.stage = stage
        """``"add"`` when a face refused the angle, ``"build"`` otherwise."""
        self.index = index
        """The position in the specs of the face that failed, for ``"add"``."""
        self.problematic = problematic
        """The sub-shape the backend blames, when it names one."""


class SurfaceFilling(ABC):
    """A surface being fitted through edge, edge-on-face and point constraints.

    Constraints are added one at a time and :meth:`build` may be called more
    than once as they accumulate.
    """

    @abstractmethod
    def add_edge(
        self, edge: Handle, continuity: str, face: Handle | None = None
    ) -> None:
        """Constrain the surface to pass through ``edge`` with ``continuity``
        ("C0", "G1" or "G2") to ``face`` when one is given."""

    @abstractmethod
    def add_point(self, point: Point) -> None: ...

    @abstractmethod
    def build(self) -> Handle:
        """The face fitted so far; raises the backend's failure types."""


class AncestorMap:
    """Sub-shapes of one kind inside a parent and, for each, the sub-shapes
    of a higher kind that contain it (an edge's faces, a vertex's edges).

    Keys are looked up by :meth:`Kernel.hash_key`, so a handle that is the
    same entity at the same placement finds its entry whatever its
    orientation, while a moved copy does not.
    """

    def __init__(
        self, kernel: Kernel, entries: Sequence[tuple[Handle, Sequence[Handle]]]
    ):
        self._keys = [key for key, _ in entries]
        self._by_key = {
            kernel.hash_key(key): list(ancestors) for key, ancestors in entries
        }
        self._kernel = kernel

    def keys(self) -> list[Handle]:
        """The sub-shapes, in kernel order."""
        return list(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def _find(self, shape: Handle) -> Hashable | None:
        """The stored key for ``shape``: by hash key, else the key of the
        same entity (a backend may know one vertex under two handles)."""
        key = self._kernel.hash_key(shape)
        if key in self._by_key:
            return key
        for candidate in self._keys:
            if self._kernel.is_same(candidate, shape):
                return self._kernel.hash_key(candidate)
        return None

    def __contains__(self, shape: Handle) -> bool:
        return self._find(shape) is not None

    def get(self, shape: Handle) -> list[Handle]:
        """The ancestors of ``shape``; KeyError when it is not a key."""
        key = self._find(shape)
        if key is None:
            raise KeyError(self._kernel.hash_key(shape))
        return self._by_key[key]

    def items(self) -> list[tuple[Handle, list[Handle]]]:
        return [(key, self.get(key)) for key in self._keys]


class CurveView(ABC):
    """Read-only parametric access to the curve under an edge or wire.

    Parameters are the backend's native ones; ``first``/``last`` bound them.
    ``geom_type`` is the name of the build123d ``GeomType`` member.
    """

    geom_type: str
    first: float
    last: float
    shape: Handle
    """The edge or wire the view was taken from."""

    @abstractmethod
    def is_periodic(self) -> bool: ...

    @abstractmethod
    def period(self) -> float: ...

    @abstractmethod
    def value(self, t: float) -> Point: ...

    @abstractmethod
    def d1(self, t: float) -> tuple[Point, Point]:
        """Point and first derivative at ``t``."""

    @abstractmethod
    def length(self, t0: float | None = None, t1: float | None = None) -> float:
        """Arc length over ``[t0, t1]``, the whole curve by default."""

    @abstractmethod
    def param_at_length(self, length: float, t0: float | None = None) -> float:
        """The parameter ``length`` along the curve from ``t0`` (``first`` by default)."""

    @abstractmethod
    def closest_param(self, point: Point) -> float:
        """The parameter of the point on the curve nearest ``point``."""

    @abstractmethod
    def positions_by_deflection(self, deflection: float) -> list[Point]:
        """Points along the curve whose polyline stays within ``deflection``."""

    @abstractmethod
    def circle(self) -> tuple[Point, Point, float]:
        """Centre, axis direction and radius; only for a CIRCLE curve."""

    @abstractmethod
    def ellipse(self) -> tuple[Point, Point, float, float]:
        """Centre, axis direction, major and minor radii; only for an ELLIPSE."""

    @abstractmethod
    def hyperbola(self) -> tuple[Point, Point, float, float]:
        """Centre, axis direction, major and minor radii; only for a HYPERBOLA."""

    @abstractmethod
    def parabola(self) -> tuple[Point, Point, float]:
        """Centre, axis direction and focal length; only for a PARABOLA."""

    @abstractmethod
    def bezier(self) -> tuple[int, list[Point], list[float] | None]:
        """Degree, poles and weights (None when not rational); only for a BEZIER."""

    @abstractmethod
    def bspline(
        self,
    ) -> tuple[int, bool, list[Point], list[float] | None, list[float], list[int]]:
        """Degree, periodicity, poles, weights (None when not rational), knots
        and multiplicities; only for a BSPLINE."""

    @abstractmethod
    def offset(self) -> tuple[float, Point, Handle]:
        """Offset distance, offset direction and an edge on the basis curve;
        only for an OFFSET curve."""

    @abstractmethod
    def dn(self, t: float, order: int) -> Point:
        """The derivative of ``order`` at ``t``."""

    @abstractmethod
    def is_closed(self) -> bool: ...


class SurfaceView(ABC):
    """Read-only parametric access to the surface under a face.

    Normals follow the face's orientation (they point out of a solid).
    """

    geom_type: str

    @abstractmethod
    def uv_bounds(self) -> tuple[float, float, float, float]:
        """``(umin, umax, vmin, vmax)`` of the face's parameter domain."""

    @abstractmethod
    def is_u_periodic(self) -> bool: ...

    @abstractmethod
    def is_v_periodic(self) -> bool: ...

    @abstractmethod
    def u_period(self) -> float: ...

    @abstractmethod
    def v_period(self) -> float: ...

    @abstractmethod
    def value(self, u: float, v: float) -> Point: ...

    @abstractmethod
    def d1(self, u: float, v: float) -> tuple[Point, Point, Point]:
        """Point and the two first partial derivatives at ``(u, v)``."""

    @abstractmethod
    def dn(self, u: float, v: float, u_order: int, v_order: int) -> Point:
        """The mixed partial derivative of the given orders at ``(u, v)``."""

    @abstractmethod
    def normal(self, u: float, v: float) -> tuple[Point, Point]:
        """Point and (unnormalised) face-oriented normal at ``(u, v)``."""

    @abstractmethod
    def closest_uv(self, point: Point) -> tuple[float, float]:
        """Parameters of the point on the surface nearest ``point``."""

    @abstractmethod
    def plane(self) -> tuple[Point, Point, Point] | None:
        """``(origin, normal, x_dir)`` of a right-handed plane the surface lies in,
        or None when it is not planar within tolerance."""

    @abstractmethod
    def radius(self) -> float:
        """Radius of a CYLINDER or SPHERE surface."""

    @abstractmethod
    def semi_angle(self) -> float:
        """Half angle in radians of a CONE surface."""

    @abstractmethod
    def torus_radii(self) -> tuple[float, float]:
        """Major and minor radii of a TORUS surface."""

    @abstractmethod
    def curvatures(
        self, u: float, v: float, tolerance: float
    ) -> tuple[float, float, Point] | None:
        """Minimum and maximum principal curvature and the surface's own
        normal direction at ``(u, v)``; None where curvature is undefined."""

    @abstractmethod
    def uv_of_point(self, point: Point, tolerance: float) -> tuple[float, float, float]:
        """Parameters of ``point`` on the surface and how far it is from it."""

    @abstractmethod
    def iso_lengths(
        self, u_min: float, u_max: float, v_min: float, v_max: float, tolerance: float
    ) -> tuple[float, float]:
        """Lengths of the middle v-isoparametric curve over ``[u_min, u_max]``
        and of the middle u-isoparametric curve over ``[v_min, v_max]``."""


class KernelGeometry(ABC):
    """Curve/surface views, construction and tessellation: the second half of
    the facade, kept on a mixin so ``Kernel`` stays readable."""

    # ---- views --------------------------------------------------------------

    @abstractmethod
    def curve_of(self, shape: Handle) -> CurveView:
        """The curve under an edge, or the composite curve of a wire."""

    @abstractmethod
    def surface_of(self, face: Handle) -> SurfaceView:
        """The surface under a face, with the face's placement applied."""

    # ---- construction -------------------------------------------------------

    @abstractmethod
    def make_box(
        self, frame: Handle, length: float, width: float, height: float
    ) -> Handle: ...

    @abstractmethod
    def make_cylinder(
        self, frame: Handle, radius: float, height: float, angle: float
    ) -> Handle:
        """``angle`` in radians; a full cylinder is 2*pi."""

    @abstractmethod
    def make_cone(
        self,
        frame: Handle,
        base_radius: float,
        top_radius: float,
        height: float,
        angle: float,
    ) -> Handle: ...

    @abstractmethod
    def make_sphere(
        self, frame: Handle, radius: float, angle1: float, angle2: float, angle3: float
    ) -> Handle: ...

    @abstractmethod
    def make_torus(
        self,
        frame: Handle,
        major_radius: float,
        minor_radius: float,
        angle1: float,
        angle2: float,
        angle3: float,
    ) -> Handle: ...

    @abstractmethod
    def make_wedge(
        self,
        frame: Handle,
        delta_x: float,
        delta_y: float,
        delta_z: float,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
    ) -> Handle: ...

    @abstractmethod
    def extrude(self, shape: Handle, direction: Point) -> Handle:
        """Sweep ``shape`` linearly along ``direction`` (its length is the distance)."""

    @abstractmethod
    def revolve(self, shape: Handle, axis: Handle, angle: float) -> Handle:
        """Revolve ``shape`` about ``axis`` by ``angle`` radians."""

    @abstractmethod
    def loft(self, sections: Sequence[Handle], solid: bool, ruled: bool) -> Handle:
        """Skin ``sections`` (wires, with an optional vertex at either end)."""

    @abstractmethod
    def sweep(
        self,
        profile: Handle,
        path: Handle,
        *,
        is_frenet: bool,
        transition: SweepTransition,
        make_solid: bool,
        binormal: Point | None = None,
        binormal_origin: Point | None = None,
        auxiliary_spine: Handle | None = None,
        curvilinear_equivalence: bool = True,
    ) -> Handle:
        """Sweep a wire profile along a wire path."""

    @abstractmethod
    def fillet(
        self, shape: Handle, edges: Sequence[Handle], radius: float
    ) -> tuple[Handle, Any]:
        """Round ``edges`` of ``shape``; returns the result and the history record."""

    @abstractmethod
    def chamfer(
        self, shape: Handle, specs: Sequence[tuple[Handle, Handle, float, float]]
    ) -> tuple[Handle, Any]:
        """Bevel edges: each spec is ``(edge, reference_face, distance1, distance2)``."""

    @abstractmethod
    def thick_solid(
        self,
        shape: Handle,
        openings: Sequence[Handle],
        thickness: float,
        tolerance: float,
        join: JoinKind,
        remove_internal_edges: bool = False,
    ) -> Handle:
        """Offset the faces of ``shape`` by ``thickness``, leaving ``openings`` open."""

    @abstractmethod
    def make_solid(self, shells: Sequence[Handle]) -> Handle:
        """A solid bounded by ``shells`` (the first outer, the rest voids)."""

    @abstractmethod
    def make_vertex(self, point: Point) -> Handle: ...

    # ---- edges from analytic curves; angles in radians, frames are gp_Ax2 ---

    @abstractmethod
    def make_line(self, start: Point, end: Point) -> Handle: ...

    @abstractmethod
    def make_infinite_line(self, origin: Point, direction: Point) -> Handle:
        """An edge on an unbounded line."""

    @abstractmethod
    def make_circle(self, frame: Handle, radius: float) -> Handle: ...

    @abstractmethod
    def make_circle_arc(
        self, frame: Handle, radius: float, start: float, end: float, sense: bool
    ) -> Handle:
        """The arc from angle ``start`` to ``end``; ``sense`` False runs it the other way."""

    @abstractmethod
    def make_ellipse(
        self, frame: Handle, major: float, minor: float, rotation: float = 0.0
    ) -> Handle:
        """``rotation`` turns the ellipse about the frame's axis."""

    @abstractmethod
    def make_ellipse_arc(
        self,
        frame: Handle,
        major: float,
        minor: float,
        start: float,
        end: float,
        sense: bool,
        rotation: float = 0.0,
    ) -> Handle: ...

    @abstractmethod
    def make_parabola_arc(
        self, frame: Handle, focal_length: float, start: float, end: float, sense: bool
    ) -> Handle: ...

    @abstractmethod
    def make_hyperbola(
        self, frame: Handle, major: float, minor: float, rotation: float = 0.0
    ) -> Handle: ...

    @abstractmethod
    def make_hyperbola_arc(
        self,
        frame: Handle,
        major: float,
        minor: float,
        start: float,
        end: float,
        sense: bool,
        rotation: float = 0.0,
    ) -> Handle: ...

    @abstractmethod
    def make_tangent_arc(self, start: Point, tangent: Point, end: Point) -> Handle: ...

    @abstractmethod
    def make_three_point_arc(self, p1: Point, p2: Point, p3: Point) -> Handle: ...

    @abstractmethod
    def make_bezier(
        self, poles: Sequence[Point], weights: Sequence[float] | None = None
    ) -> Handle: ...

    @abstractmethod
    def make_bspline(
        self,
        poles: Sequence[Point],
        knots: Sequence[float],
        multiplicities: Sequence[int],
        degree: int,
        periodic: bool,
        weights: Sequence[float] | None = None,
    ) -> Handle: ...

    @abstractmethod
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
    ) -> Handle:
        """A B-spline through ``points``; raises ValueError when it cannot be built."""

    @abstractmethod
    def make_spline_fit(
        self,
        points: Sequence[Point],
        tolerance: float,
        min_degree: int,
        max_degree: int,
        smoothing: tuple[float, float, float] | None = None,
    ) -> Handle:
        """A B-spline approximating ``points``; raises ValueError when it cannot be built."""

    @abstractmethod
    def make_helix(
        self,
        center: Point,
        normal: Point,
        radius: float,
        pitch: float,
        height: float,
        cone_angle: float,
        lefthand: bool,
    ) -> Handle:
        """A helix about ``normal`` through ``center``; ``cone_angle`` in radians."""

    # ---- edges from edges -----------------------------------------------------

    @abstractmethod
    def edge_range(self, edge: Handle) -> tuple[float, float]:
        """The curve parameters at the ends of an edge."""

    @abstractmethod
    def vertex_param_on_edge(self, vertex: Handle, edge: Handle) -> float:
        """The curve parameter of a vertex of an edge."""

    @abstractmethod
    def reversed_geometry(self, edge: Handle) -> Handle:
        """A new edge on the reversed curve, running the other way."""

    @abstractmethod
    def trim_edge(self, edge: Handle, start: float, end: float) -> Handle:
        """A new edge on the same curve between the parameters ``start`` and ``end``."""

    @abstractmethod
    def trim_edge_by_length(self, edge: Handle, start: float, length: float) -> Handle:
        """A new edge on the same curve from the parameter ``start`` for ``length``."""

    @abstractmethod
    def approximate_edge(
        self, edge: Handle, tolerance: float, max_segments: int, max_degree: int
    ) -> Handle:
        """A C2 B-spline edge within ``tolerance`` of ``edge``; raises
        RuntimeError when none can be found."""

    @abstractmethod
    def wire_as_bspline_edge(self, wire: Handle, tolerance: float) -> Handle:
        """One B-spline edge through every non-degenerate edge of the wire in
        order; raises RuntimeError when they cannot be joined."""

    @abstractmethod
    def wire_edges(self, wire: Handle) -> list[Handle]:
        """The edges of a wire in connection order, with their orientation in it."""

    @abstractmethod
    def build_curves_3d(
        self, edge: Handle, tolerance: float, max_segments: int | None = None
    ) -> None:
        """Give an edge defined only on a surface its 3D curve."""

    @abstractmethod
    def find_plane(self, shape: Handle) -> tuple[Point, Point, Point] | None:
        """``(origin, normal, x_dir)`` of a plane the shape's edges lie in, or None."""

    @abstractmethod
    def continuity_between(
        self, edge1: Handle, edge2: Handle, u1: float, u2: float, tolerance: float
    ) -> str:
        """The continuity ("C0", "G1", "C1", "G2", "C2", "C3" or "CN") of two
        edges where they meet at the parameters ``u1`` and ``u2``."""

    @abstractmethod
    def planar_intersections(
        self,
        edge: Handle,
        bounds: tuple[float, float],
        other: Handle | None,
        other_bounds: tuple[float, float] | None,
        plane: Handle,
        tolerance: float,
    ) -> list[tuple[float, float]]:
        """The points, in the plane's coordinates, where the projection of
        ``edge`` onto ``plane`` meets that of ``other``, or itself when
        ``other`` is None."""

    @abstractmethod
    def nearest_edge(self, shape: Handle, point: Point) -> tuple[Handle, float] | None:
        """The edge of ``shape`` nearest ``point`` and the parameter on it of
        the nearest point; None when the search fails."""

    @abstractmethod
    def edge_uv_on_face(
        self, edge: Handle, face: Handle, params: Sequence[float]
    ) -> list[tuple[float, float]]:
        """The face's surface parameters along an edge of it, at ``params``."""

    @abstractmethod
    def vertex_uv_on_face(self, vertex: Handle, face: Handle) -> tuple[float, float]:
        """The face's surface parameters at one of its vertices."""

    @abstractmethod
    def frame_at(
        self, shape: Handle, param: float, corrected: bool
    ) -> tuple[Point, Point, Point, Point]:
        """Point, tangent, normal and binormal of the Frenet frame (corrected
        Frenet when ``corrected``) of an edge or wire at ``param``."""

    @abstractmethod
    def make_polygon(self, points: Sequence[Point]) -> Handle:
        """A wire of straight edges through ``points`` in order."""

    @abstractmethod
    def connect_edges_to_wires(
        self, edges: Sequence[Handle], tolerance: float
    ) -> list[Handle]:
        """Chain ``edges`` into as few wires as their shared ends allow."""

    @abstractmethod
    def outer_wire(self, face: Handle) -> Handle: ...

    @abstractmethod
    def fillet_2d(
        self, face: Handle, corners: Sequence[tuple[Handle, float]]
    ) -> tuple[Handle, Any]:
        """A planar face with each ``(vertex, radius)`` corner rounded; plus
        the history record."""

    @abstractmethod
    def chamfer_2d(
        self, face: Handle, corners: Sequence[tuple[Handle, Handle, float, float]]
    ) -> tuple[Handle, Any]:
        """A planar face with each corner between ``(edge1, edge2)`` cut back by
        ``distance1`` along the first and ``distance2`` along the second; plus
        the history record."""

    @abstractmethod
    def fix_small_edges(
        self, wire: Handle, precision: float, max_tolerance: float
    ) -> Handle:
        """A wire with edges shorter than ``precision`` removed and gaps closed."""

    @abstractmethod
    def project(
        self,
        shape: Handle,
        target: Handle,
        *,
        direction: Point | None = None,
        center: Point | None = None,
    ) -> list[Handle]:
        """The wires where ``shape`` (an edge or wire) lands on ``target`` when
        projected along ``direction`` or towards ``center``."""

    @abstractmethod
    def offset_2d(
        self,
        wire: Handle,
        distance: float,
        join: JoinKind,
        plane: Handle | None = None,
    ) -> Handle | None:
        """The planar offset of ``wire`` by ``distance``, within ``plane``
        (``gp_Pln``) when the wire does not fix one; None when it fails."""

    @abstractmethod
    def offset_shape(
        self,
        shape: Handle,
        amount: float,
        tolerance: float,
        *,
        join: JoinKind = JoinKind.ARC,
        intersection: bool = False,
        thicken: bool = False,
        remove_internal_edges: bool = False,
    ) -> Handle:
        """The surface offset of a face or shell by ``amount``, or with
        ``thicken`` the solid between it and its offset; a null handle when
        the backend produces nothing."""

    @abstractmethod
    def make_bezier_surface(
        self,
        points: Sequence[Sequence[Point]],
        weights: Sequence[Sequence[float]] | None = None,
    ) -> Handle:
        """A face on the Bezier surface with the grid of ``points`` as poles."""

    @abstractmethod
    def make_spline_surface_fit(
        self,
        points: Sequence[Sequence[Point]],
        tolerance: float,
        min_degree: int,
        max_degree: int,
        smoothing: tuple[float, float, float] | None = None,
    ) -> Handle:
        """A face on a B-spline surface approximating the grid of ``points``;
        raises ValueError when it cannot be built."""

    @abstractmethod
    def filling(self, **options: Any) -> SurfaceFilling:
        """A new surface filling; ``options`` are the backend's own settings."""

    @abstractmethod
    def sweep_multi(
        self,
        profiles: Sequence[Handle],
        path: Handle,
        *,
        is_frenet: bool,
        make_solid: bool,
        binormal: Point | None = None,
        binormal_origin: Point | None = None,
        auxiliary_spine: Handle | None = None,
    ) -> Handle:
        """Sweep several wire profiles along a wire path, one surface between
        each pair of consecutive profiles."""

    @abstractmethod
    def draft_prism(
        self,
        shape: Handle,
        face: Handle,
        basis: Handle | None,
        taper: float,
        additive: bool,
        *,
        up_to_face: Handle | None = None,
        depth: float | None = None,
    ) -> Handle:
        """Add to or cut from ``shape`` a prism grown from ``face`` with the
        taper angle ``taper`` (radians), up to a face, to a depth, or through all."""

    @abstractmethod
    def extrude_generated(
        self, shape: Handle, direction: Point
    ) -> tuple[Handle, AncestorMap, Any]:
        """:meth:`extrude` with, for each edge of ``shape``, the faces the
        extrusion generated from it; plus the history record."""

    @abstractmethod
    def draft(
        self,
        shape: Handle,
        specs: Sequence[tuple[Handle, float]],
        direction: Point,
        plane: Handle,
    ) -> tuple[Handle, Any]:
        """``shape`` with each ``(face, angle)`` face tilted by ``angle``
        radians about its intersection with ``plane`` (``gp_Pln``), pulled
        along ``direction``; plus the history record. Raises
        :class:`DraftFailure`."""

    @abstractmethod
    def split_by_edges(
        self, shape: Handle, edges: Sequence[Handle]
    ) -> tuple[list[Handle], list[Handle]]:
        """Split a face or shell along ``edges`` lying on it; the pieces to the
        left and to the right of them."""

    @abstractmethod
    def line_intersections(
        self, shape: Handle, origin: Point, direction: Point, tolerance: float
    ) -> list[tuple[Handle, Point]]:
        """Each face of ``shape`` the unbounded line crosses and where."""

    @abstractmethod
    def solid_from_shell(self, shell: Handle) -> Handle:
        """The solid a closed shell bounds, oriented so its volume is positive."""

    @abstractmethod
    def remove_wires(self, face: Handle, wires: Sequence[Handle]) -> Handle:
        """``face`` with ``wires`` (holes) taken out of its boundary."""

    @abstractmethod
    def is_seam(self, edge: Handle, face: Handle) -> bool:
        """Whether an edge is used twice by a face, as the seam of a cylinder is."""

    @abstractmethod
    def to_splines(
        self, shape: Handle, degree: int, tolerance: float, nurbs: bool
    ) -> Handle:
        """``shape`` with its geometry approximated by B-splines of ``degree``."""

    @abstractmethod
    def edge_in_plane(self, edge: Handle, normal: Point, tolerance: float) -> bool:
        """Whether an edge's own 3D curve (its placement left aside) lies in
        a plane with ``normal``."""

    @abstractmethod
    def offset_face(self, face: Handle, amount: float, tolerance: float) -> Handle:
        """The surface offset of ``face`` by ``amount``."""

    @abstractmethod
    def edge_uv_ends_on_face(
        self, edge: Handle, face: Handle
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """For each end of an edge of ``face``: the uv point there and the
        uv tangent leaving the curve at that end (first end: forward, last
        end: backward), unnormalised."""

    @abstractmethod
    def vertex_point(self, vertex: Handle) -> Point:
        """Where a vertex is, its placement applied."""

    @abstractmethod
    def make_wire(self, shapes: Sequence[Handle]) -> tuple[Handle | None, WireStatus]:
        """Chain edges (or the edges of wires) into one wire.

        Consecutive edges are added together so that the kernel may order
        them; a wire is added whole. The wire is None when the status is
        EMPTY or DISCONNECTED; a NON_MANIFOLD wire is still returned.
        """

    @abstractmethod
    def make_wire_placed(self, edges: Sequence[Handle]) -> tuple[Handle, list[Handle]]:
        """Chain ``edges`` one at a time into a wire, returning it and each
        edge as it was placed (the kernel may copy edges so neighbours share
        vertices)."""

    @abstractmethod
    def make_planar_face(self, outer: Handle, inners: Sequence[Handle]) -> Handle:
        """A planar face bounded by the closed wire ``outer`` with ``inners``
        as holes; raises ValueError when the kernel cannot build it."""

    @abstractmethod
    def add_holes(self, face: Handle, wires: Sequence[Handle]) -> Handle:
        """``face`` with ``wires`` cut out of it as holes on its own surface."""

    @abstractmethod
    def face_of_plane(
        self, plane: Handle, bounds: tuple[float, float, float, float] | None = None
    ) -> Handle:
        """The face of a plane (``gp_Pln`` for the OCCT backend), unbounded or
        trimmed to ``(u_min, u_max, v_min, v_max)`` in the plane's own axes."""

    @abstractmethod
    def face_frame(self, face: Handle) -> tuple[Point, Point, Point]:
        """The centre of a face, a direction along its surface's u axis and
        the face's normal (taken where the surface passes nearest the world
        origin): what a Plane is built from."""

    @abstractmethod
    def planar_surface(self, face: Handle) -> Handle | None:
        """The plane (``gp_Pln`` for the OCCT backend, right-handed) a face's
        surface lies in within tolerance, whatever its type, or None."""

    @abstractmethod
    def surface_axis(self, face: Handle) -> tuple[str, Point, Point] | None:
        """The build123d ``GeomType`` name of the face's basis surface (any
        trimming or offsetting stripped) with the origin and direction of
        its axis of rotation; None for a surface that has none."""

    @abstractmethod
    def half_space(self, shape: Handle, point: Point) -> Handle:
        """The half-space solid bounded by a face or shell that contains ``point``."""

    def is_inside_out(self, solid: Handle) -> bool:
        """Whether a solid's shell faces inwards (its volume comes out negative)."""
        return self.volume(solid) < 0

    def same_side(self, piece: Handle, tool: Handle, point: Point, tolerance: float) -> bool:
        """Whether ``piece`` lies on the side of the sheet ``tool`` that
        ``point`` is on: by default, whether it shares anything of length
        with the half-space the sheet bounds there."""
        common, _ = self.boolean(BoolOp.COMMON, [piece], [self.half_space(tool, point)])
        return self.length(common) >= tolerance

    @abstractmethod
    def ruled_surface(self, a: Handle, b: Handle) -> Handle:
        """A face ruled between two edges, or a shell between two wires."""

    @abstractmethod
    def make_shell(self, faces: Sequence[Handle]) -> Handle:
        """A shell holding ``faces`` as they are, with no sewing."""

    @abstractmethod
    def sew(self, faces: Sequence[Handle]) -> tuple[Handle, Any]:
        """Sew ``faces`` together along their shared edges; the result is a
        shell, a single face, or a compound of what could not be joined;
        plus the history record."""

    @abstractmethod
    def make_compound(self, shapes: Sequence[Handle]) -> Handle: ...

    @abstractmethod
    def remove_from_compound(self, compound: Handle, shape: Handle) -> None:
        """Take ``shape`` out of ``compound`` in place."""

    @abstractmethod
    def transformed(
        self, shape: Handle, transform: Handle, copy: bool = True
    ) -> tuple[Handle, Any]:
        """``shape`` under a rigid or similarity transform (``gp_Trsf``),
        geometry types kept; plus the history record."""

    @abstractmethod
    def transformed_general(self, shape: Handle, transform: Handle) -> Handle:
        """``shape`` under a general affine transform (``gp_GTrsf``), which
        converts its geometry to splines where needed."""

    # ---- distance and classification ----------------------------------------

    @abstractmethod
    def distance(self, a: Handle, b: Handle) -> float:
        """The least distance between two shapes."""

    @abstractmethod
    def closest_points(self, a: Handle, b: Handle) -> tuple[float, Point, Point]:
        """The least distance and a pair of points on ``a`` and ``b`` that realise it."""

    @abstractmethod
    def extrema(
        self, a: Handle, b: Handle, deflection: float | None = None
    ) -> tuple[float, list[tuple[Point, Point]]]:
        """The least distance between ``a`` and ``b`` and every pair of
        points that realises it; ``(inf, [])`` when the search fails."""

    @abstractmethod
    def point_in_solid(self, solid: Handle, point: Point, tolerance: float) -> bool:
        """Whether ``point`` is inside ``solid`` or on its boundary."""

    @abstractmethod
    def point_on_face(self, face: Handle, point: Point, tolerance: float) -> bool:
        """Whether ``point`` lies on ``face``, its edges included."""

    @abstractmethod
    def uv_in_face(
        self, face: Handle, uv: tuple[float, float], tolerance: float
    ) -> bool:
        """Whether the surface parameters ``uv`` fall inside the face's trim."""

    # ---- tessellation -------------------------------------------------------

    @abstractmethod
    def mesh(self, shape: Handle, tolerance: float, angular_tolerance: float) -> None:
        """Make sure ``shape`` carries a triangulation at least this fine."""

    @abstractmethod
    def face_triangulation(
        self, face: Handle
    ) -> tuple[list[Point], list[tuple[int, int, int]]] | None:
        """The face's triangulation in world coordinates, triangles wound to
        follow the face's orientation; None when the face has none."""


class Kernel(KernelGeometry):
    """What build123d asks of a B-rep kernel.

    Methods take and return opaque handles. A backend is free to represent a
    handle however it likes, but every method must be total over the handles
    it hands out: build123d never inspects a handle itself.
    """

    name: str = "abstract"

    failure_types: tuple[type[BaseException], ...] = ()
    """The exception types the backend raises when an operation cannot be
    done; what a caller catches around a kernel call that may fail."""

    # ---- handles and identity ------------------------------------------------

    @abstractmethod
    def kind(self, shape: Handle) -> TopoKind:
        """The topological kind of ``shape``; raises ValueError for a null handle."""

    @abstractmethod
    def is_null(self, shape: Handle | None) -> bool:
        """True when ``shape`` is None or the backend's null handle."""

    @abstractmethod
    def is_handle(self, obj: Any) -> bool:
        """True when ``obj`` is one of this backend's shape handles."""

    @abstractmethod
    def downcast(self, shape: Handle) -> Handle:
        """The same shape as the most specific handle type the backend has."""

    @abstractmethod
    def is_same(self, a: Handle, b: Handle) -> bool:
        """Same underlying entity and placement; orientation may differ."""

    @abstractmethod
    def is_equal(self, a: Handle, b: Handle) -> bool:
        """Same entity, placement and orientation."""

    @abstractmethod
    def is_partner(self, a: Handle, b: Handle) -> bool:
        """Same underlying entity; placement and orientation may differ."""

    @abstractmethod
    def hash_key(self, shape: Handle) -> Hashable:
        """A hash consistent with :meth:`is_same`."""

    @abstractmethod
    def copy_deep(self, shape: Handle) -> Handle:
        """An independent copy of ``shape`` and all of its geometry."""

    def is_kind(self, shape: Handle | None, kind: TopoKind) -> bool:
        """True when ``shape`` is a non-null handle of ``kind``."""
        return (
            self.is_handle(shape)
            and not self.is_null(shape)
            and self.kind(shape) == kind
        )

    # ---- traversal ----------------------------------------------------------

    @abstractmethod
    def sub_shapes(self, shape: Handle, kind: TopoKind) -> list[Handle]:
        """Every distinct sub-shape of ``kind`` inside ``shape``, in kernel order."""

    @abstractmethod
    def sub_shape_uses(self, shape: Handle, kind: TopoKind) -> list[Handle]:
        """Every use of a sub-shape of ``kind`` inside ``shape``, repeats included.

        A seam edge of a cylinder appears twice, once per side, each with the
        orientation it has in that use.
        """

    @abstractmethod
    def direct_children(self, shape: Handle) -> list[Handle]:
        """The shapes one level down inside ``shape`` (a compound's members,
        a solid's shells, an edge's vertices), with the parent's placement
        and orientation applied to each."""

    @abstractmethod
    def ancestors(
        self, shape: Handle, sub_kind: TopoKind, ancestor_kind: TopoKind
    ) -> AncestorMap:
        """Every sub-shape of ``sub_kind`` in ``shape`` with the sub-shapes of
        ``ancestor_kind`` that contain it."""

    @abstractmethod
    def edge_vertices(self, edge: Handle) -> tuple[Handle, Handle]:
        """The first and last vertex of an edge in its own parameter direction."""

    @abstractmethod
    def is_closed(self, shape: Handle) -> bool:
        """Whether an edge or wire ends where it starts, or a shell or solid
        is closed."""

    @abstractmethod
    def is_degenerate(self, edge: Handle) -> bool:
        """Whether an edge has no length (a pole of a sphere, an apex of a cone)."""

    # ---- orientation --------------------------------------------------------

    @abstractmethod
    def is_forward(self, shape: Handle) -> bool:
        """True when ``shape`` runs with its underlying geometry."""

    @abstractmethod
    def is_reversed(self, shape: Handle) -> bool:
        """True when ``shape`` runs against its underlying geometry."""

    @abstractmethod
    def reverse(self, shape: Handle) -> None:
        """Flip the orientation of ``shape`` in place."""

    @abstractmethod
    def reversed(self, shape: Handle) -> Handle:
        """The same entity with its orientation flipped."""

    @abstractmethod
    def same_orientation(self, a: Handle, b: Handle) -> bool: ...

    @abstractmethod
    def orient_like(self, shape: Handle, like: Handle) -> None:
        """Give ``shape`` the orientation of ``like``, in place."""

    # ---- validity and repair ------------------------------------------------

    @abstractmethod
    def is_valid(self, shape: Handle) -> bool:
        """The backend's full topological and geometric validity check."""

    @abstractmethod
    def fix(self, shape: Handle) -> Handle:
        """A repaired copy of ``shape`` (tolerances, missing pcurves, orientation)."""

    # ---- measures -----------------------------------------------------------

    @abstractmethod
    def area(self, shape: Handle) -> float:
        """Total surface area of the faces in ``shape``."""

    @abstractmethod
    def volume(self, shape: Handle) -> float:
        """Total volume of the solids in ``shape``."""

    @abstractmethod
    def length(self, shape: Handle) -> float:
        """Total length of the edges in ``shape``."""

    @abstractmethod
    def measure(self, shape: Handle, kind: TopoKind) -> float:
        """The size of ``shape`` measured as a ``kind``: length of edges and
        wires, area of faces and shells, volume of solids and compounds."""

    @abstractmethod
    def center_of_mass(
        self, shape: Handle, kind: TopoKind
    ) -> tuple[float, float, float]:
        """Centre of mass of ``shape`` measured as a ``kind``: edges by length,
        faces by area, solids by volume."""

    @abstractmethod
    def inertia_matrix(self, shape: Handle) -> list[list[float]]:
        """The 3x3 matrix of inertia of the solids in ``shape`` about their centre of mass."""

    @abstractmethod
    def principal_inertia(self, shape: Handle) -> list[tuple[Point, float]]:
        """The three principal axes of inertia of ``shape`` with their moments."""

    @abstractmethod
    def static_moments(self, shape: Handle) -> tuple[float, float, float]:
        """The first moments of the solids in ``shape`` about the coordinate planes."""

    @abstractmethod
    def radius_of_gyration(self, shape: Handle, axis: Handle) -> float:
        """The radius of gyration of the solids in ``shape`` about ``axis`` (``gp_Ax1``)."""

    @abstractmethod
    def has_inconsistent_orientation(self, shape: Handle) -> bool:
        """Whether the shells of ``shape`` have an edge used in the same
        direction by both of its faces."""

    @abstractmethod
    def bounding_box(
        self, shape: Handle, optimal: bool = True
    ) -> tuple[float, float, float, float, float, float] | None:
        """Axis-aligned bounds ``(xmin, ymin, zmin, xmax, ymax, zmax)`` of
        ``shape``, enlarged by the backend's tolerances, or None for an empty
        shape. ``optimal`` asks for the tight box rather than the quick one."""

    # ---- operations ---------------------------------------------------------

    @abstractmethod
    def boolean(
        self,
        op: BoolOp,
        args: Sequence[Handle],
        tools: Sequence[Handle],
        *,
        glue: bool = False,
        fuzzy: float | None = None,
    ) -> tuple[Handle, Any]:
        """Run ``op`` with ``tools`` against ``args``.

        Returns the result and the backend's record of what the operation
        generated and modified (opaque; consumed by ``topology.history``).
        ``glue`` speeds up shapes that only touch; ``fuzzy`` is a merge tolerance.
        """

    # ---- operation records ----------------------------------------------------

    @abstractmethod
    def record_modified(self, record: Any, shape: Handle) -> list[Handle]:
        """What ``record`` (the opaque history an operation handed back) says
        the input sub-shape ``shape`` was rebuilt into; empty when untouched
        or when the record has no answer."""

    @abstractmethod
    def record_generated(self, record: Any, shape: Handle) -> list[Handle]:
        """What ``record`` says was created from the input sub-shape ``shape``."""

    @abstractmethod
    def record_removed(self, record: Any, shape: Handle) -> bool:
        """Whether ``record`` says the input sub-shape ``shape`` was removed."""

    @abstractmethod
    def sewn_shape(self, record: Any) -> Handle:
        """The result of the sewing whose record this is."""

    @abstractmethod
    def section(self, a: Handle, b: Handle) -> tuple[Handle, Any]:
        """The vertices and edges where ``a`` and ``b`` meet, with the pcurves
        on both and approximated section curves; plus the history record."""

    @abstractmethod
    def unify_same_domain(self, shape: Handle) -> tuple[Handle, Any]:
        """``shape`` with faces and edges lying on the same geometry merged;
        plus the history record. Raises when the backend cannot."""

    # ---- placement ----------------------------------------------------------

    @abstractmethod
    def identity_location(self) -> Handle:
        """The placement that leaves a shape where it is."""

    @abstractmethod
    def location_of(self, shape: Handle) -> Handle:
        """The placement carried by ``shape``."""

    @abstractmethod
    def set_location(self, shape: Handle, location: Handle) -> None:
        """Give ``shape`` the absolute placement ``location``, in place."""

    @abstractmethod
    def move(self, shape: Handle, location: Handle) -> None:
        """Displace ``shape`` by ``location`` from where it is, in place."""

    @abstractmethod
    def located(self, shape: Handle, location: Handle) -> Handle:
        """A handle to the same geometry at the absolute placement ``location``."""

    @abstractmethod
    def moved(self, shape: Handle, location: Handle) -> Handle:
        """A handle to the same geometry displaced by ``location`` from where it is."""

    @abstractmethod
    def relocation_between(self, source: Handle, target: Handle) -> Handle:
        """The displacement that takes ``source``'s placement to ``target``'s."""
