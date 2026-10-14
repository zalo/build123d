"""
build123d topology

name: history.py
by:   Gumyr
date: September 10, 2026

desc:
    What an operation did to the sub-shapes of its inputs, as the kernel
    reports it: which sub-shapes it left alone, which it rebuilt (modified),
    which it created from another (generated) and which it removed. The
    record from one operation can be merged with the next and the chain read
    from the original inputs to the final result.

    The builders use this to answer Select.LAST and Select.NEW without
    comparing shapes before and after: a face the operation only trimmed is
    a modified face, not a new one.

    A record is a stack of layers. A layer is either the opaque record a
    kernel operation handed back, read through the kernel facade on demand,
    or a set of answers kept in Python. Layers compose: a sub-shape modified
    by one operation into another, which the next operation modified again,
    maps to the final shape.

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

from collections.abc import Hashable, Iterable
from typing import Any

from typing_extensions import Self

from build123d.kernel import Handle, TopoKind, kernel

TRACKED_KINDS = (TopoKind.VERTEX, TopoKind.EDGE, TopoKind.FACE, TopoKind.SOLID)
"""The sub-shape kinds a record follows; wires, shells and compounds are
containers and are not tracked."""


def tracked_subshapes(shapes: Iterable[Handle | None]) -> list[Handle]:
    """Every distinct vertex, edge, face and solid of the given shapes."""
    found: dict[Hashable, Handle] = {}
    for shape in shapes:
        if kernel.is_null(shape):
            continue
        for kind in TRACKED_KINDS:
            for sub in kernel.sub_shapes(shape, kind):
                found.setdefault(kernel.hash_key(sub), sub)
    return list(found.values())


class _Layer:
    """One operation's answers: what each tracked sub-shape became."""

    def __init__(self, record: Any = None):
        self.record = record
        self.modified: dict[Hashable, list[Handle]] = {}
        self.generated: dict[Hashable, list[Handle]] = {}
        self.removed: set[Hashable] = set()

    def modified_of(self, shape: Handle) -> list[Handle]:
        found = self.modified.get(kernel.hash_key(shape))
        if found is not None:
            return list(found)
        if self.record is not None:
            return kernel.record_modified(self.record, shape)
        return []

    def generated_of(self, shape: Handle) -> list[Handle]:
        found = self.generated.get(kernel.hash_key(shape))
        if found is not None:
            return list(found)
        if self.record is not None:
            return kernel.record_generated(self.record, shape)
        return []

    def is_removed(self, shape: Handle) -> bool:
        if kernel.hash_key(shape) in self.removed:
            return True
        if self.record is not None:
            return kernel.record_removed(self.record, shape)
        return False


class ShapeHistory:
    """The record of what an operation did to its inputs' sub-shapes.

    Each tracked sub-shape of an input ends up in one of four states in the
    result: untouched (it is still there, unchanged), modified (rebuilt into
    one or more sub-shapes of the same kind), generating (a sub-shape of
    another kind was made from it, as a face is made from an edge by an
    extrusion) or removed. Booleans and ``clean`` hand over a complete
    record; other algorithms answer per sub-shape and :meth:`from_algorithm`
    collects those answers.

    Records chain with :meth:`merge`, so that after a fuse and a clean a face
    of the original input still maps to the face it became.

    A record also knows which inputs were already there (``before``) and which
    the operation brought in (``brought``): for ``a + b`` the argument and the
    tool. That is what lets a sub-shape of the result be classified as last or
    new, see :meth:`is_last` and :meth:`is_new`.
    """

    def __init__(
        self,
        record: Any = None,
        before: Iterable[Handle] = (),
        brought: Iterable[Handle] = (),
    ):
        self.layers: list[_Layer] = [_Layer(record)]
        self.before: list[Handle] = list(before)
        self.brought: list[Handle] = list(brought)
        self._traces: tuple[Trace, Trace] | None = None
        self._produced: set[Hashable] | None = None

    # ---- Constructors ----

    @classmethod
    def from_boolean(
        cls,
        record: Any,
        before: Iterable[Handle] = (),
        brought: Iterable[Handle] = (),
    ) -> ShapeHistory:
        """The record a boolean operation (or splitter) kept while building."""
        return cls(record, before, brought)

    @classmethod
    def from_unify(cls, record: Any) -> ShapeHistory:
        """The record of a ``clean``."""
        return cls(record)

    @classmethod
    def from_algorithm(
        cls,
        algorithm: Any,
        inputs: Iterable[Handle],
        result: Handle,
    ) -> ShapeHistory:
        """Collect the per-sub-shape answers of a kernel algorithm.

        A sub-shape is removed when it is neither still in ``result`` nor
        modified into something that is; the algorithm's own opinion on
        deletion is not consulted, as several guess for the kinds they do not
        track. An algorithm that will not answer for a kind is taken to have
        no answer for it.
        """
        inputs = list(inputs)
        history = cls(before=inputs)
        layer = history.layers[0]
        present = {kernel.hash_key(sub) for sub in tracked_subshapes([result])}
        for sub in tracked_subshapes(inputs):
            key = kernel.hash_key(sub)
            modified = kernel.record_modified(algorithm, sub)
            if modified:
                layer.modified[key] = modified
            generated = kernel.record_generated(algorithm, sub)
            if generated:
                layer.generated[key] = generated
            if not modified and key not in present:
                layer.removed.add(key)
        return history

    @classmethod
    def from_sewing(cls, sewing: Any, inputs: Iterable[Handle]) -> ShapeHistory:
        """The record of sewing faces into a shell.

        Sewing reports the faces it rebuilt and the edges and vertices it
        merged; a face it dropped as a duplicate is neither, and is recorded
        as removed.
        """
        inputs = list(inputs)
        history = cls(before=inputs)
        layer = history.layers[0]
        present = {
            kernel.hash_key(sub)
            for sub in tracked_subshapes([kernel.sewn_shape(sewing)])
        }
        for sub in tracked_subshapes(inputs):
            key = kernel.hash_key(sub)
            modified = kernel.record_modified(sewing, sub)
            if modified:
                layer.modified[key] = modified
            elif key not in present:
                layer.removed.add(key)
        return history

    @classmethod
    def from_relocation(cls, before: Handle, after: Handle) -> ShapeHistory:
        """The record of moving a shape: each sub-shape modified into its moved self.

        A move keeps every underlying entity and changes only placements, so
        the pairing is by entity. Sub-shapes whose placement did not change
        are left untouched.
        """
        history = cls(before=[before])
        layer = history.layers[0]
        identity = kernel.identity_location()
        placed: dict[Hashable, Handle] = {}
        for sub in tracked_subshapes([after]):
            placed[kernel.hash_key(kernel.located(sub, identity))] = sub
        for sub in tracked_subshapes([before]):
            moved = placed.get(kernel.hash_key(kernel.located(sub, identity)))
            if moved is not None and not kernel.is_same(moved, sub):
                layer.modified[kernel.hash_key(sub)] = [moved]
        return history

    @classmethod
    def of(cls, *shapes: object) -> ShapeHistory | None:
        """The records the given shapes carry from the operations that made them,
        merged into one, or None if none of them carries one.

        A ``Shape`` made by an operation that records holds the record as
        ``_history``; anything else, including a list of shapes, is looked
        through for shapes that do.
        """
        found: list[ShapeHistory] = []
        for shape in shapes:
            if isinstance(shape, (list, tuple)):
                nested = cls.of(*shape)
                if nested is not None:
                    found.append(nested)
            else:
                history = getattr(shape, "_history", None)
                if isinstance(history, ShapeHistory):
                    found.append(history)
        if not found:
            return None
        if len(found) == 1:
            return found[0]
        merged = cls(
            before=[s for h in found for s in h.before],
            brought=[s for h in found for s in h.brought],
        )
        merged.layers = [layer for history in found for layer in history.layers]
        return merged

    # ---- Instance Methods ----

    def add_modified(self, before: Handle, after: Handle) -> Self:
        """Record that ``before`` was rebuilt into ``after``.

        For what an algorithm did but does not say: a 2D fillet rebuilds the
        face it works on without reporting it.
        """
        if kernel.kind(before) in TRACKED_KINDS:
            self.layers[-1].modified.setdefault(kernel.hash_key(before), []).append(
                after
            )
            self._traces = None
            self._produced = None
        return self

    def add_generated(self, source: Handle, made: Handle) -> Self:
        """Record that ``made`` was created from ``source``."""
        if kernel.kind(source) in TRACKED_KINDS:
            self.layers[-1].generated.setdefault(kernel.hash_key(source), []).append(
                made
            )
            self._traces = None
            self._produced = None
        return self

    def remove(self, shape: Handle) -> Self:
        """Record that ``shape`` did not survive the operation."""
        if kernel.kind(shape) in TRACKED_KINDS:
            self.layers[-1].removed.add(kernel.hash_key(shape))
            self._traces = None
            self._produced = None
        return self

    def has_modified(self) -> bool:
        """Does the record say any sub-shape was rebuilt?"""
        return any(layer.modified for layer in self.layers) or any(
            layer.record is not None for layer in self.layers
        )

    def merge(self, later: ShapeHistory | None) -> Self:
        """Extend this record with the operation that came after it."""
        if later is not None:
            self.layers.extend(later.layers)
            self._traces = None
            self._produced = None
        return self

    def with_inputs(self, before: Iterable[Handle], brought: Iterable[Handle]) -> Self:
        """Say which inputs were already there and which were brought in."""
        self.before = list(before)
        self.brought = list(brought)
        self._traces = None
        self._produced = None
        return self

    def _from(self) -> tuple[Trace, Trace]:
        """The record read from the result's side, for each set of inputs."""
        if self._traces is None:
            self._traces = (Trace(self, self.before), Trace(self, self.brought))
        return self._traces

    def is_last(self, shape: Handle) -> bool:
        """Did the operation bring this sub-shape of the result in, or create it?

        A sub-shape that descends from something brought in, untouched or
        rebuilt, is last, including one that also descends from what was there
        before, such as the vertex a new edge shares with an old one. One that
        descends only from what was there before is not. One the record cannot
        attribute to any input is new, and so last.
        """
        from_before, from_brought = self._from()
        if from_brought.is_untouched(shape) or from_brought.is_modified(shape):
            return True
        return not (from_before.is_untouched(shape) or from_before.is_modified(shape))

    def is_new(self, shape: Handle) -> bool:
        """Did this sub-shape of the result exist in no input, in any form?"""
        from_before, from_brought = self._from()
        for trace in (from_brought, from_before):
            if trace.is_untouched(shape) or trace.is_modified(shape):
                return False
        return True

    def _intermediates(self) -> set[Hashable]:
        """The sub-shapes some layer made from the inputs: results of one
        operation that the next one consumed. Only original inputs are
        answered for; an intermediate queried directly is unknown, as it is
        to a kernel record merged across operations."""
        if self._produced is None:
            produced: set[Hashable] = set()
            current = tracked_subshapes(self.before + self.brought)
            for layer in self.layers:
                following: list[Handle] = []
                for item in current:
                    if layer.is_removed(item):
                        continue
                    became = layer.modified_of(item) + layer.generated_of(item)
                    for made in became:
                        produced.add(kernel.hash_key(made))
                    following.extend(layer.modified_of(item) or [item])
                current = following
            self._produced = produced
        return self._produced

    def _follow(self, shape: Handle) -> tuple[list[Handle], bool, bool]:
        """Where ``shape`` ends up through every layer: the shapes it became,
        whether any layer changed it, and whether it was removed."""
        if kernel.hash_key(shape) in self._intermediates():
            return [shape], False, False
        current = [shape]
        changed = False
        for layer in self.layers:
            following: list[Handle] = []
            for item in current:
                if layer.is_removed(item):
                    changed = True
                    continue
                became = layer.modified_of(item)
                if became:
                    changed = True
                    following.extend(became)
                else:
                    following.append(item)
            current = following
        return current, changed, changed and not current

    def modified(self, shape: Handle) -> list[Handle]:
        """What an input sub-shape was rebuilt into, if anything."""
        if kernel.kind(shape) not in TRACKED_KINDS:
            return []
        final, changed, _ = self._follow(shape)
        return final if changed else []

    def generated(self, shape: Handle) -> list[Handle]:
        """What was made from an input sub-shape, if anything."""
        if kernel.kind(shape) not in TRACKED_KINDS:
            return []
        if kernel.hash_key(shape) in self._intermediates():
            return []
        made: list[Handle] = []
        current = [shape]
        for index, layer in enumerate(self.layers):
            for item in current:
                for product in layer.generated_of(item):
                    # what later operations made of the product
                    survivors = [product]
                    for after in self.layers[index + 1 :]:
                        survivors = [
                            s
                            for item2 in survivors
                            if not after.is_removed(item2)
                            for s in (after.modified_of(item2) or [item2])
                        ]
                    made.extend(survivors)
            following: list[Handle] = []
            for item in current:
                if layer.is_removed(item):
                    continue
                following.extend(layer.modified_of(item) or [item])
            current = following
        seen: set[Hashable] = set()
        unique = []
        for item in made:
            key = kernel.hash_key(item)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def is_removed(self, shape: Handle) -> bool:
        """Was an input sub-shape removed by the operation?"""
        if kernel.kind(shape) not in TRACKED_KINDS:
            return False
        return self._follow(shape)[2]

    def trace(self, inputs: Iterable[Handle]) -> Trace:
        """Index the record from the result's side for the given inputs."""
        return Trace(self, inputs)


class Trace:
    """The record of a :class:`ShapeHistory` read from the result's side.

    Built for a set of inputs, it answers for any sub-shape of the result
    whether it was one of those inputs' sub-shapes left untouched, was
    modified from one, or was generated from one.
    """

    def __init__(self, history: ShapeHistory, inputs: Iterable[Handle]):
        self.untouched: set[Hashable] = set()
        self.modified_from: dict[Hashable, Handle] = {}
        self.generated_from: dict[Hashable, Handle] = {}
        for sub in tracked_subshapes(inputs):
            self.untouched.add(kernel.hash_key(sub))
            for shape in history.modified(sub):
                self.modified_from[kernel.hash_key(shape)] = sub
            for shape in history.generated(sub):
                self.generated_from[kernel.hash_key(shape)] = sub

    def is_untouched(self, shape: Handle) -> bool:
        """Was this result sub-shape an input sub-shape, unchanged?"""
        return kernel.hash_key(shape) in self.untouched

    def is_modified(self, shape: Handle) -> bool:
        """Was this result sub-shape rebuilt from an input sub-shape?"""
        return kernel.hash_key(shape) in self.modified_from

    def is_generated(self, shape: Handle) -> bool:
        """Was this result sub-shape made from an input sub-shape of another kind?"""
        return kernel.hash_key(shape) in self.generated_from

    def is_descendant(self, shape: Handle) -> bool:
        """Did this result sub-shape come from the inputs in any way?"""
        return (
            self.is_untouched(shape)
            or self.is_modified(shape)
            or self.is_generated(shape)
        )
