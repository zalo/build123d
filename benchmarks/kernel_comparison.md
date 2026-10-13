# Kernel speed comparison: OCCT vs SMLib (smcore backend)

Measured with `benchmarks/bench_kernels.py` on 2026-10-13, branch `kernel-smlib`,
same machine, 7 repeats per operation, median wall time in milliseconds. Each
operation runs through the build123d API with the kernel selected by
`BUILD123D_KERNEL` (`occt`, or `smcore` for the SMLib backend over the
generated `_smlib_core` bindings).

```
BUILD123D_KERNEL=occt   .venv/bin/python benchmarks/bench_kernels.py 7 > occt.json
BUILD123D_KERNEL=smcore ~/usdbrep-venv/bridge/run312.sh benchmarks/bench_kernels.py 7 > smcore.json
```

"Same result" compares a checksum of each result (volume, area, length or
count rounded to 4 places).

| Operation | OCCT ms | SMLib ms | SMLib / OCCT | Same result |
|---|---:|---:|---:|---|
| box | 0.09 | 0.40 | 4.34 | yes |
| sphere | 0.07 | 1.11 | 16.16 | yes |
| cylinder | 0.07 | 2.21 | 30.71 | yes |
| box volume | 0.08 | 0.31 | 3.95 | yes |
| sphere volume | 0.07 | 16.05 | 229.33 | yes |
| sphere area | 0.07 | 16.22 | 245.73 | yes |
| box bounding box | 0.01 | 0.02 | 1.14 | yes |
| box faces/edges/vertices | 0.16 | 0.09 | 0.54 | yes |
| fuse box+cylinder | 3.43 | 4.03 | 1.17 | yes |
| cut box-cylinder | 2.65 | 6.17 | 2.32 | yes |
| common box&sphere | 3.01 | 6.99 | 2.32 | no (1748.8199 vs 1748.82) |
| cut box-sphere | 3.16 | 7.38 | 2.33 | no (4251.1801 vs 4251.18) |
| fuse 8 boxes | 35.49 | 16.82 | 0.47 | yes |
| fillet box edges | 9.98 | 14.49 | 1.45 | yes |
| chamfer box edges | 10.83 | 10.43 | 0.96 | yes |
| fillet cylinder edges | 2.40 | 6.05 | 2.52 | yes |
| sketch fuse rect+circle | 1.25 | 1.22 | 0.97 | yes |
| sketch cut rect-circle | 0.86 | 1.33 | 1.54 | yes |
| rect fillet 2d | 8.90 | 6.92 | 0.78 | yes |
| rect offset 2d | 1.57 | 2.50 | 1.60 | yes |
| polygon face | 0.18 | 0.09 | 0.49 | yes |
| extrude | 0.25 | 0.45 | 1.81 | yes |
| revolve | 0.32 | 1.92 | 5.99 | yes |
| loft | 2.36 | 3.54 | 1.50 | no (3961.4835 vs 3605.9308) |
| sweep | 1.40 | 36.66 | 26.24 | no (383.1496 vs 383.1497) |
| helix | 0.08 | 0.08 | 0.94 | yes |
| spline | 0.03 | 0.02 | 0.73 | yes |
| offset solid | 5.32 | 27.25 | 5.12 | yes |
| split box | 1.35 | 5.36 | 3.97 | yes |
| shell box | 2.79 | 4.24 | 1.52 | yes |
| tessellate sphere | 36.58 | 96.37 | 2.63 | yes |
| distance box-sphere | 0.47 | 0.67 | 1.42 | yes |
| point in solid | 0.04 | 0.01 | 0.27 | yes |
| edge positions | 0.56 | 10.44 | 18.75 | yes |
| face normal | 0.05 | 0.02 | 0.52 | yes |
| deepcopy | 0.06 | 0.03 | 0.53 | yes |
| move/rotate | 0.07 | 0.10 | 1.46 | yes |

## Reading the numbers

- Primitives are 4-30x slower on SMLib: every primitive is built through the
  generic `SmPrimitiveCreation` sweep machinery rather than an analytic solid.
- Mass properties: the SMLib backend integrates them itself (planar faces by
  Green's theorem around the loops, faces covering a whole rectangle of their
  parameter space by Gauss quadrature on the knot spans, graded toward poles,
  other trimmed faces by Green's theorem in parameter space through their uv
  trim curves) because the kernel's own integrator costs 150 ms on a sphere
  at the accuracy the tests need; a body with trimmed pole-bearing patches
  still falls back to that integrator (~300 ms for a box with all edges
  filleted).
- Booleans are 1.3-2.5x slower one at a time but fusing many disjoint bodies
  is 2x faster (bodies that do not touch are kept apart and merged only with
  what they overlap).
- Fillets and chamfers are on par; 2D fillets, offsets, polygon faces,
  traversal, copying and point classification are faster on SMLib.
- Sweeps, revolves, solid offsets and splits are 2-28x slower; the sweep
  result also carries a dense knot vector that makes its volume cost 150 ms.
- Position lookups along curves cost 0.1 ms each on SMLib (the kernel's
  arc-length parameter search); lines are special-cased.
- The circle-to-rectangle loft differs because SMLib skins a smooth surface
  where OCCT's default loft is ruled between the sections in a different way;
  both are valid lofts.

## History

| Date | Change | Effect |
|---|---|---|
| 2026-10-13 (first run) | baseline after the transform/boolean-history work | fillet 12 box edges 3.0 s, sphere volume 148 ms, offset solid 117 ms |
| 2026-10-13 (second run) | cheap face hashing, face-wise mass properties, inside-out check, length cache | fillet 15 ms, sphere volume 4 ms, offset 27 ms |
| 2026-10-13 (third run) | pole-graded quadrature, uv-loop integration, cornered sweeps | sphere volume 16 ms (exact to 1e-11), chamfer on par, split 5 ms |
