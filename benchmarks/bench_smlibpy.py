"""Time the same operations as bench_kernels.py through smlibpy.

Usage: python bench_smlibpy.py [repeats]   (in a venv with smlibpy installed)
Prints one JSON line per operation: name, median ms, min ms, checksum, or
an "n/a" line when smlibpy has no equivalent.
"""

import copy
import json
import statistics
import sys
import time

from smlibpy import *  # noqa: F401,F403

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 7


def bench(name, fn, check=None):
    try:
        times = []
        value = None
        for _ in range(REPEATS):
            t = time.perf_counter()
            value = fn()
            times.append((time.perf_counter() - t) * 1000)
        result = check(value) if check else None
        print(json.dumps({"op": name, "median_ms": round(statistics.median(times), 3), "min_ms": round(min(times), 3), "check": result}))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"op": name, "na": f"{type(exc).__name__}: {str(exc)[:80]}"}))


def vol(s):
    return round(s.volume, 4)


def area(s):
    return round(s.area, 4)


def main():
    print(json.dumps({"kernel": "smlibpy", "repeats": REPEATS}))
    box = Box(10, 20, 30)
    sphere = Sphere(8)
    cyl = Cylinder(5, 30)

    bench("box", lambda: Box(10, 20, 30), vol)
    bench("sphere", lambda: Sphere(8), vol)
    bench("cylinder", lambda: Cylinder(5, 30), vol)
    bench("box volume", lambda: box.volume, lambda v: round(v, 4))
    bench("sphere volume", lambda: sphere.volume, lambda v: round(v, 4))
    bench("sphere area", lambda: sphere.area, lambda v: round(v, 4))
    bench("box bounding box", lambda: box.bbox(), lambda b: round(b.size[2] if not hasattr(b.size, "z") else b.size.z, 4))
    bench("box faces/edges/vertices", lambda: (box.faces(), box.edges(), box.vertices()), lambda t: len(t[1]))

    bench("fuse box+cylinder", lambda: box + cyl, vol)
    bench("cut box-cylinder", lambda: box - cyl, vol)
    bench("common box&sphere", lambda: box & sphere, vol)
    bench("cut box-sphere", lambda: box - sphere, vol)
    bench("fuse 8 boxes", lambda: Box(60, 60, 2).fuse(GridLocations(8, 8, 2, 4) * Box(10, 10, 10)), vol)

    bench("fillet box edges", lambda: box.fillet(box.edges(), 2), vol)
    bench("chamfer box edges", lambda: box.chamfer(box.edges(), 2), vol)
    bench("fillet cylinder edges", lambda: cyl.fillet(cyl.edges().circles(), 1), vol)

    rect = Rectangle(10, 20)
    circ = Circle(6)
    bench("sketch fuse rect+circle", lambda: rect + circ, area)
    bench("sketch cut rect-circle", lambda: rect - circ, area)
    bench("rect fillet 2d", lambda: rect.fillet(2), area)
    bench("rect offset 2d", lambda: rect.offset(2), area)
    bench("polygon face", lambda: Polygon((0, 0), (10, 0), (10, 10), (5, 15), (0, 10)), area)

    profile = Rectangle(10, 20)
    bench("extrude", lambda: extrude(profile, 30), vol)
    bench("revolve", lambda: revolve(Pos(20, 0) * Rectangle(5, 10), Axis.Y, 270), vol)
    bench("loft", lambda: loft([Circle(10), Pos(0, 0, 20) * Rectangle(10, 10)]), vol)
    path = Spline((0, 0, 0), (10, 5, 10), (20, 0, 20))
    bench("sweep", lambda: Circle(2).on(Plane(path.point_at(0), z_dir=path.tangent_at(0))).sweep(path), vol)
    bench("helix", lambda: Helix(5, 30, 4), lambda e: round(e.length, 4))
    bench("spline", lambda: Spline((0, 0), (5, 5), (10, 0), (15, 5)), lambda e: round(e.length, 4))
    bench("offset solid", lambda: box.offset(1), vol)
    bench("split box", lambda: box.split(Plane.XY.moved(Location((0, 0, 3)))), vol)
    bench("shell box", lambda: box.shell(1, box.faces().top()), vol)

    bench("tessellate sphere", lambda: Sphere(8).mesh(0.01), lambda m: True)
    bench("distance box-sphere", lambda: (Pos(30, 0, 0) * box).distance_to(sphere), lambda d: round(d, 3))
    bench("point in solid", lambda: box.bbox().contains((1, 1, 1)), bool)
    bench("edge positions", lambda: [Circle(5).edges()[0].point_at(i / 50) for i in range(51)], len)
    bench("face normal", lambda: box.faces()[0].normal_at(), lambda v: 1.0)
    bench("deepcopy", lambda: copy.deepcopy(box), vol)
    bench("move/rotate", lambda: box.moved(Location((1, 2, 3), (10, 20, 30))), vol)


if __name__ == "__main__":
    main()
