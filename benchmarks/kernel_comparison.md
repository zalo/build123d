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
| box | 0.09 | 0.41 | 4.35 | yes |
| sphere | 0.07 | 1.11 | 15.93 | yes |
| cylinder | 0.07 | 2.19 | 32.28 | yes |
| box volume | 0.08 | 1.29 | 16.93 | yes |
| sphere volume | 0.07 | 4.31 | 61.56 | yes |
| sphere area | 0.07 | 4.31 | 64.31 | yes |
| box bounding box | 0.01 | 0.01 | 1.07 | yes |
| box faces/edges/vertices | 0.17 | 0.09 | 0.56 | yes |
| fuse box+cylinder | 3.30 | 4.24 | 1.28 | yes |
| cut box-cylinder | 2.59 | 6.30 | 2.43 | yes |
| common box&sphere | 3.07 | 7.03 | 2.29 | no (1748.8199 vs 1748.82) |
| cut box-sphere | 3.10 | 7.45 | 2.41 | no (4251.1801 vs 4251.18) |
| fuse 8 boxes | 33.38 | 16.36 | 0.49 | yes |
| fillet box edges | 10.10 | 14.52 | 1.44 | yes |
| chamfer box edges | 11.43 | 9.49 | 0.83 | yes |
| fillet cylinder edges | 2.49 | 5.19 | 2.09 | yes |
| sketch fuse rect+circle | 1.14 | 1.19 | 1.04 | yes |
| sketch cut rect-circle | 0.73 | 1.32 | 1.80 | yes |
| rect fillet 2d | 8.97 | 6.93 | 0.77 | yes |
| rect offset 2d | 1.56 | 2.51 | 1.61 | yes |
| polygon face | 0.18 | 0.09 | 0.51 | yes |
| extrude | 0.25 | 0.62 | 2.48 | yes |
| revolve | 0.32 | 1.92 | 6.04 | yes |
| loft | 2.36 | 3.53 | 1.49 | no (3961.4835 vs 3605.9308) |
| sweep | 1.40 | 39.01 | 27.78 | no (383.1496 vs 383.1497) |
| helix | 0.08 | 0.07 | 0.93 | yes |
| spline | 0.03 | 0.02 | 0.76 | yes |
| offset solid | 5.33 | 27.28 | 5.12 | yes |
| split box | 1.37 | 9.90 | 7.24 | yes |
| shell box | 2.81 | 5.66 | 2.01 | yes |
| tessellate sphere | 37.44 | 96.53 | 2.58 | yes |
| distance box-sphere | 0.54 | 0.67 | 1.24 | yes |
| point in solid | 0.04 | 0.01 | 0.23 | yes |
| edge positions | 0.54 | 10.41 | 19.13 | yes |
| face normal | 0.04 | 0.03 | 0.59 | yes |
| deepcopy | 0.06 | 0.03 | 0.50 | yes |
| move/rotate | 0.07 | 0.10 | 1.39 | yes |

## Reading the numbers

- Primitives are 4-30x slower on SMLib: every primitive is built through the
  generic `SmPrimitiveCreation` sweep machinery rather than an analytic solid.
- Mass properties: the SMLib backend integrates them itself (planar faces by
  Green's theorem around the loops, faces covering a whole rectangle of their
  parameter space by Gauss quadrature on the knot spans) because the kernel's
  own integrator costs 150 ms on a sphere at the accuracy the tests need; a
  body with trimmed curved faces still falls back to that integrator
  (~300 ms for a box with all edges filleted).
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
