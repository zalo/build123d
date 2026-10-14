"""Time common build123d operations on the selected kernel.

Usage: BUILD123D_KERNEL=occt|smcore python bench_kernels.py [repeats]
Prints one line per operation: name, median ms, min ms, and a checksum.
"""

import json
import math
import os
import statistics
import sys
import time

from build123d import *  # noqa: F401,F403
from build123d.kernel import kernel
from build123d.topology import SkipClean, SkipHistory

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def bench(name, fn, check=None):
    times = []
    value = None
    for _ in range(REPEATS):
        t = time.perf_counter()
        value = fn()
        times.append((time.perf_counter() - t) * 1000)
    result = check(value) if check else None
    print(
        json.dumps(
            {
                "op": name,
                "median_ms": round(statistics.median(times), 3),
                "min_ms": round(min(times), 3),
                "check": result,
            }
        )
    )


def vol(s):
    return round(s.volume, 4)


def no_history(fn, clean=True):
    """``fn()`` with the record of the operation switched off (and the
    clean, when ``clean`` is False)."""
    with SkipHistory():
        if clean:
            return fn()
        with SkipClean():
            return fn()


def area(s):
    return round(s.area, 4)


def main():
    print(json.dumps({"kernel": kernel.name, "repeats": REPEATS}))
    box = Box(10, 20, 30)
    sphere = Sphere(8)
    cyl = Cylinder(5, 30)

    bench("box", lambda: Box(10, 20, 30), vol)
    bench("sphere", lambda: Sphere(8), vol)
    bench("cylinder", lambda: Cylinder(5, 30), vol)
    bench("box volume", lambda: box.volume, lambda v: round(v, 4))
    bench("sphere volume", lambda: sphere.volume, lambda v: round(v, 4))
    bench("sphere area", lambda: sphere.area, lambda v: round(v, 4))
    bench("box bounding box", lambda: box.bounding_box(), lambda b: round(b.size.Z, 4))
    bench("box faces/edges/vertices", lambda: (box.faces(), box.edges(), box.vertices()), lambda t: len(t[1]))

    bench("fuse box+cylinder", lambda: box + cyl, vol)
    bench("cut box-cylinder", lambda: box - cyl, vol)
    bench("common box&sphere", lambda: box & sphere, vol)
    bench("cut box-sphere", lambda: box - sphere, vol)
    bench("cut box-sphere no history", lambda: no_history(lambda: box - sphere), vol)
    bench("cut box-sphere no history no clean", lambda: no_history(lambda: box - sphere, clean=False), vol)
    bench("fuse box+cylinder no history", lambda: no_history(lambda: box + cyl), vol)
    bench("fuse 8 boxes", lambda: Compound(GridLocations(8, 8, 2, 4) * Box(10, 10, 10)) + Box(60, 60, 2), vol)

    bench("fillet box edges", lambda: fillet(box.edges(), 2), vol)
    bench("chamfer box edges", lambda: chamfer(box.edges(), 2), vol)
    bench("fillet cylinder edges", lambda: fillet(cyl.edges().filter_by(GeomType.CIRCLE), 1), vol)

    rect = Rectangle(10, 20)
    circ = Circle(6)
    bench("sketch fuse rect+circle", lambda: rect + circ, area)
    bench("sketch cut rect-circle", lambda: rect - circ, area)
    bench("rect fillet 2d", lambda: fillet(rect.vertices(), 2), area)
    bench("rect offset 2d", lambda: offset(rect, 2), area)
    bench("polygon face", lambda: Face(Wire.make_polygon([(0, 0), (10, 0), (10, 10), (5, 15), (0, 10), (0, 0)])), area)

    profile = Rectangle(10, 20)
    bench("extrude", lambda: extrude(profile, 30), vol)
    bench("revolve", lambda: revolve(Pos(20, 0) * Rectangle(5, 10), Axis.Y, 270), vol)
    bench("loft", lambda: loft([Circle(10).face(), Pos(0, 0, 20) * Rectangle(10, 10).face()]), vol)
    path = Spline((0, 0, 0), (10, 5, 10), (20, 0, 20))
    bench("sweep", lambda: sweep(Plane((0, 0, 0), z_dir=path % 0) * Circle(2), path), vol)
    bench("helix", lambda: Helix(5, 30, 4), lambda e: round(e.length, 4))
    bench("spline", lambda: Spline((0, 0), (5, 5), (10, 0), (15, 5)), lambda e: round(e.length, 4))
    bench("offset solid", lambda: offset(box, 1), vol)
    bench("split box", lambda: split(box, Plane.XY.offset(3)), vol)
    bench("shell box", lambda: offset(box, -1, openings=box.faces().sort_by(Axis.Z)[-1]), vol)

    bench("tessellate sphere", lambda: Sphere(8).tessellate(0.01), lambda t: len(t[1]) > 0)
    bench("distance box-sphere", lambda: (Pos(30, 0, 0) * box).distance_to(sphere), lambda d: round(d, 3))
    bench("point in solid", lambda: box.is_inside((1, 1, 1)), bool)
    bench("edge positions", lambda: [Edge.make_circle(5).position_at(i / 50) for i in range(51)], len)
    bench("face normal", lambda: box.faces()[0].normal_at(), lambda v: round(v.length, 3))
    bench("deepcopy", lambda: box.__deepcopy__({}), vol)
    bench("move/rotate", lambda: box.moved(Location((1, 2, 3), (10, 20, 30))), vol)


if __name__ == "__main__":
    main()
