"""
build123d kernel facade

name: smlib.py
by:   Gumyr
date: October 12, 2026

desc:
    A second backend behind :class:`~build123d.kernel.base.Kernel`: the SMLib
    NURBS kernel reached through NVIDIA's ``usd_brep`` bindings.

    Phase 2 of the facade work. Handles stay ``TopoDS_Shape`` objects, so
    identity, traversal, placement, wires, compounds and the curve and
    surface views are inherited from the OCCT backend unchanged; what this
    backend takes over are the solid operations that SMLib does well:
    booleans, fillets, whole-body chamfers, shelling and offsetting. Each
    one crosses into SMLib through the OCCT ``.brep`` converters the patched
    ``usd_brep`` build provides and comes back the same way.

    Every operation falls back to OCCT when SMLib raises, rejects the input
    (wires and bare edges cannot cross the bridge) or hands back a shape
    OCCT finds invalid. The counts of operations done by SMLib and of
    fallbacks are kept in :attr:`SmlibKernel.stats`; set
    ``BUILD123D_KERNEL_STRICT=1`` to raise instead of falling back and
    ``BUILD123D_KERNEL_LOG=1`` to print each fallback to stderr.

    SMLib keeps no generated/modified record, so the history record handed
    back is reconstructed by matching the sub-shapes of the result against
    those of the inputs by geometry; see :meth:`SmlibKernel._reconstruct`.

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

import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools, BRepTools_History
from OCP.TopoDS import TopoDS_Shape

from build123d.kernel.base import BoolOp, JoinKind, TopoKind
from build123d.kernel.occt import OcctKernel

_FINGERPRINT_DIGITS = 4
"""Sub-shapes of a result are matched to those of the inputs by centre and
size rounded to this many decimals; SMLib round trips are exact on planar and
analytic geometry and within 1e-5 on the rest."""


class SmlibUnavailable(ImportError):
    """``usd_brep`` could not be imported."""


class SmlibKernel(OcctKernel):
    """SMLib through ``usd_brep`` for solid operations, OCCT for the rest."""

    name = "smlib"

    def __init__(self) -> None:
        try:
            import usd_brep  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel
        except ImportError as err:
            raise SmlibUnavailable(
                "the smlib kernel needs the patched usd_brep build on the path: "
                "run with ~/usdbrep-venv/bridge/run312.sh (Python 3.12 only)"
            ) from err
        self.sm = usd_brep
        self.stats: Counter[str] = Counter()
        """``"<op>"`` for each operation SMLib completed, ``"<op>.fallback"``
        for each one OCCT had to do instead."""
        self.fallbacks: list[tuple[str, str]] = []
        """``(operation, reason)`` for every fallback, in order."""
        self.strict = os.environ.get("BUILD123D_KERNEL_STRICT", "") not in ("", "0")
        self.log = os.environ.get("BUILD123D_KERNEL_LOG", "") not in ("", "0")

    # ---- crossing the bridge --------------------------------------------------

    def _to_smlib(self, shape: TopoDS_Shape) -> list[Any]:
        """The SMLib bodies of ``shape``: one per solid, shell or face root."""
        with tempfile.NamedTemporaryFile(suffix=".brep", delete=False) as handle:
            path = handle.name
        try:
            BRepTools.Write_s(shape, path)
            breps = self.sm.usd.import_occt_brep(path, False)
        finally:
            os.unlink(path)
        if not breps:
            raise ValueError("the shape has no solid, shell or face to send to SMLib")
        return list(breps)

    def _one_body(self, shape: TopoDS_Shape) -> Any:
        breps = self._to_smlib(shape)
        if len(breps) != 1:
            raise ValueError(f"expected one body, the shape gave {len(breps)}")
        return breps[0]

    def _from_smlib(self, brep: Any) -> TopoDS_Shape:
        text = self.sm.usd.export_occt_brep_text(brep)
        with tempfile.NamedTemporaryFile("w", suffix=".brep", delete=False) as handle:
            handle.write(text)
            path = handle.name
        try:
            shape = TopoDS_Shape()
            BRepTools.Read_s(shape, path, BRep_Builder())
        finally:
            os.unlink(path)
        if shape.IsNull():
            raise ValueError("SMLib returned an empty body")
        shape = self.downcast(shape)
        if not self.is_valid(shape):
            raise ValueError("SMLib returned a body OCCT finds invalid")
        return shape

    def _attempt(
        self, op: str, run: Callable[[], Any], fallback: Callable[[], Any]
    ) -> Any:
        """Run ``run`` through SMLib, or ``fallback`` through OCCT when it fails."""
        try:
            result = run()
        except Exception as err:  # pylint: disable=broad-exception-caught
            reason = str(err).splitlines()[0][:160] if str(err) else type(err).__name__
            if self.strict:
                raise
            self.stats[f"{op}.fallback"] += 1
            self.fallbacks.append((op, reason))
            if self.log:
                print(f"[smlib] {op}: fell back to occt: {reason}", file=sys.stderr)
            return fallback()
        self.stats[op] += 1
        return result

    # ---- matching sub-shapes across the bridge ---------------------------------

    def _fingerprint(self, shape: TopoDS_Shape) -> tuple:
        kind = self.kind(shape)
        if kind is TopoKind.VERTEX:
            return (kind,) + tuple(
                round(c, _FINGERPRINT_DIGITS) for c in self.vertex_point(shape)
            )
        center = self.center_of_mass(shape, kind)
        size = self.measure(shape, kind)
        return (
            (kind,)
            + tuple(round(c, _FINGERPRINT_DIGITS) for c in center)
            + (round(size, _FINGERPRINT_DIGITS),)
        )

    @staticmethod
    def _close(a: Sequence[float], b: Sequence[float], tolerance: float) -> bool:
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))

    def _body_edges(self, brep: Any, edges: Sequence[TopoDS_Shape]) -> list[Any]:
        """The SMLib edges of ``brep`` that are the OCCT ``edges``."""
        wanted = []
        for edge in edges:
            curve = self.curve_of(edge)
            wanted.append((curve.value((curve.first + curve.last) / 2), curve.length()))
        candidates = list(brep.edges())
        found = []
        for middle, length in wanted:
            tolerance = 1e-4 * max(1.0, length)
            matches = [
                candidate
                for candidate in candidates
                if abs(candidate.length() - length) <= tolerance
                and self._close(candidate.closest_point(middle)[0], middle, tolerance)
            ]
            if len(matches) != 1:
                raise ValueError(f"{len(matches)} SMLib edges match an edge to fillet")
            found.append(matches[0])
        return found

    def _body_faces(self, brep: Any, faces: Sequence[TopoDS_Shape]) -> list[Any]:
        """The SMLib faces of ``brep`` that are the OCCT ``faces``."""
        candidates = [(face, face.area(), face.centroid()) for face in brep.faces()]
        found = []
        for face in faces:
            area = self.area(face)
            center = self.center_of_mass(face, TopoKind.FACE)
            tolerance = 1e-4 * max(1.0, area)
            matches = [
                candidate
                for candidate, candidate_area, centroid in candidates
                if abs(candidate_area - area) <= tolerance
                and self._close(centroid, center, 1e-4 * max(1.0, area**0.5))
            ]
            if len(matches) != 1:
                raise ValueError(f"{len(matches)} SMLib faces match a face to open")
            found.append(matches[0])
        return found

    def _reconstruct(
        self, inputs: Sequence[TopoDS_Shape], result: TopoDS_Shape
    ) -> BRepTools_History:
        """A history record matching result sub-shapes to input sub-shapes by
        geometry: a sub-shape that comes back unchanged in centre and size is
        recorded as modified into its counterpart (the bridge rebuilds every
        entity, so none is ever the same handle)."""
        record = BRepTools_History()
        by_print: dict[tuple, list[TopoDS_Shape]] = {}
        for kind in (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE):
            for sub in self.sub_shapes(result, kind):
                by_print.setdefault(self._fingerprint(sub), []).append(sub)
        for shape in inputs:
            for kind in (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE):
                for sub in self.sub_shapes(shape, kind):
                    for match in by_print.get(self._fingerprint(sub), []):
                        record.AddModified(sub, match)
        return record

    # ---- operations -------------------------------------------------------------

    def boolean(
        self,
        op: BoolOp,
        args: Sequence[TopoDS_Shape],
        tools: Sequence[TopoDS_Shape],
        *,
        glue: bool = False,
        fuzzy: float | None = None,
    ) -> tuple[TopoDS_Shape, Any]:
        if op not in (BoolOp.FUSE, BoolOp.CUT, BoolOp.COMMON) or glue or fuzzy:
            return super().boolean(op, args, tools, glue=glue, fuzzy=fuzzy)

        def run() -> tuple[TopoDS_Shape, Any]:
            arg_bodies = [body for shape in args for body in self._to_smlib(shape)]
            tool_bodies = [body for shape in tools for body in self._to_smlib(shape)]
            booleans = self.sm.booleans
            if op is BoolOp.FUSE:
                bodies = arg_bodies + tool_bodies
                result = bodies[0]
                for body in bodies[1:]:
                    result = booleans.boolean_union(result, body)
            elif op is BoolOp.CUT:
                result = arg_bodies[0]
                for body in arg_bodies[1:]:
                    result = booleans.boolean_union(result, body)
                for body in tool_bodies:
                    result = booleans.boolean_difference(result, body)
            else:
                result = arg_bodies[0]
                for body in arg_bodies[1:]:
                    result = booleans.boolean_union(result, body)
                for body in tool_bodies:
                    result = booleans.boolean_intersection(result, body)
            shape = self._from_smlib(result)
            return shape, self._reconstruct([*args, *tools], shape)

        return self._attempt(
            f"boolean.{op.value}",
            run,
            lambda: super(SmlibKernel, self).boolean(
                op, args, tools, glue=glue, fuzzy=fuzzy
            ),
        )

    def fillet(
        self, shape: TopoDS_Shape, edges: Sequence[TopoDS_Shape], radius: float
    ) -> tuple[TopoDS_Shape, Any]:
        def run() -> tuple[TopoDS_Shape, Any]:
            body = self._one_body(shape)
            filleted = self.sm.fillets.fillet_edges(
                body, self._body_edges(body, edges), radius=radius
            )
            result = self._from_smlib(filleted)
            return result, self._reconstruct([shape], result)

        return self._attempt(
            "fillet", run, lambda: super(SmlibKernel, self).fillet(shape, edges, radius)
        )

    def chamfer(
        self,
        shape: TopoDS_Shape,
        specs: Sequence[tuple[TopoDS_Shape, TopoDS_Shape, float, float]],
    ) -> tuple[TopoDS_Shape, Any]:
        distances = {(d1, d2) for _, _, d1, d2 in specs}
        all_edges = len(specs) == len(self.sub_shapes(shape, TopoKind.EDGE))
        symmetric = (
            len(distances) == 1 and len({d for pair in distances for d in pair}) == 1
        )
        if not (all_edges and symmetric):
            # SMLib chamfers whole bodies with one distance
            return super().chamfer(shape, specs)

        def run() -> tuple[TopoDS_Shape, Any]:
            body = self._one_body(shape)
            chamfered = self.sm.fillets.chamfer_fillet(body, specs[0][2])
            result = self._from_smlib(chamfered)
            return result, self._reconstruct([shape], result)

        return self._attempt(
            "chamfer", run, lambda: super(SmlibKernel, self).chamfer(shape, specs)
        )

    def thick_solid(
        self,
        shape: TopoDS_Shape,
        openings: Sequence[TopoDS_Shape],
        thickness: float,
        tolerance: float,
        join: JoinKind,
        remove_internal_edges: bool = False,
    ) -> TopoDS_Shape:
        def run() -> TopoDS_Shape:
            body = self._one_body(shape)
            shelled = self.sm.offset.shell_brep(
                body, thickness, faces_to_shell=self._body_faces(body, openings)
            )
            return self._from_smlib(shelled)

        return self._attempt(
            "thick_solid",
            run,
            lambda: super(SmlibKernel, self).thick_solid(
                shape, openings, thickness, tolerance, join, remove_internal_edges
            ),
        )

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
        occt = lambda: super(SmlibKernel, self).offset_shape(  # noqa: E731
            shape,
            amount,
            tolerance,
            join=join,
            intersection=intersection,
            thicken=thicken,
            remove_internal_edges=remove_internal_edges,
        )
        if thicken or not self.is_kind(shape, TopoKind.SOLID):
            return occt()

        def run() -> TopoDS_Shape:
            return self._from_smlib(
                self.sm.offset.offset_brep(self._one_body(shape), amount)
            )

        return self._attempt("offset_shape", run, occt)
