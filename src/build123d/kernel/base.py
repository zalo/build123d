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
    """How offset faces meet at a convex edge."""

    ARC = "arc"
    INTERSECTION = "intersection"


class SweepTransition(Enum):
    """How a swept profile turns a corner of its path."""

    TRANSFORMED = "transformed"
    ROUND = "round"
    RIGHT = "right"


Point = tuple[float, float, float]


class CurveView(ABC):
    """Read-only parametric access to the curve under an edge or wire.

    Parameters are the backend's native ones; ``first``/``last`` bound them.
    ``geom_type`` is the name of the build123d ``GeomType`` member.
    """

    geom_type: str
    first: float
    last: float

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

    # ---- handles and identity ------------------------------------------------

    @abstractmethod
    def kind(self, shape: Handle) -> TopoKind:
        """The topological kind of ``shape``; raises ValueError for a null handle."""

    @abstractmethod
    def is_null(self, shape: Handle | None) -> bool:
        """True when ``shape`` is None or the backend's null handle."""

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
        return not self.is_null(shape) and self.kind(shape) == kind

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
    def center_of_mass(
        self, shape: Handle, kind: TopoKind
    ) -> tuple[float, float, float]:
        """Centre of mass of ``shape`` measured as a ``kind``: edges by length,
        faces by area, solids by volume."""

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
    def located(self, shape: Handle, location: Handle) -> Handle:
        """A handle to the same geometry at the absolute placement ``location``."""

    @abstractmethod
    def moved(self, shape: Handle, location: Handle) -> Handle:
        """A handle to the same geometry displaced by ``location`` from where it is."""

    @abstractmethod
    def relocation_between(self, source: Handle, target: Handle) -> Handle:
        """The displacement that takes ``source``'s placement to ``target``'s."""
