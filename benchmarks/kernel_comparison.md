# Kernel speed comparison: OCCT vs SMLib (smcore backend)

Measured with `benchmarks/bench_kernels.py` on 2026-10-13 (fifth run, after commit 396e2dba), branch `kernel-smlib`,
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
| box | 0.10 | 1.35 | 14.18 | yes |
| sphere | 0.07 | 1.11 | 14.96 | yes |
| cylinder | 0.07 | 2.38 | 34.00 | yes |
| box volume | 0.08 | 0.32 | 4.25 | yes |
| sphere volume | 0.07 | 15.52 | 218.63 | yes |
| sphere area | 0.07 | 15.50 | 234.79 | yes |
| box bounding box | 0.01 | 0.01 | 1.07 | yes |
| box faces/edges/vertices | 0.17 | 0.09 | 0.52 | yes |
| fuse box+cylinder | 3.14 | 2.58 | 0.82 | yes |
| cut box-cylinder | 2.44 | 2.94 | 1.21 | yes |
| common box&sphere | 3.02 | 5.29 | 1.75 | no (1748.8199 vs 1748.82) |
| cut box-sphere | 3.18 | 5.60 | 1.76 | no (4251.1801 vs 4251.18) |
| cut box-sphere no history | 3.12 | 5.55 | 1.78 | no (4251.1801 vs 4251.18) |
| cut box-sphere no history no clean | 2.95 | 4.70 | 1.59 | no (4251.1801 vs 4251.18) |
| fuse box+cylinder no history | 2.97 | 2.40 | 0.81 | yes |
| fuse 8 boxes | 35.19 | 11.94 | 0.34 | yes |
| fillet box edges | 10.08 | 14.58 | 1.45 | yes |
| chamfer box edges | 10.89 | 10.13 | 0.93 | yes |
| fillet cylinder edges | 2.50 | 5.27 | 2.11 | yes |
| sketch fuse rect+circle | 1.52 | 2.08 | 1.37 | yes |
| sketch cut rect-circle | 0.77 | 1.08 | 1.41 | yes |
| rect fillet 2d | 8.98 | 9.13 | 1.02 | yes |
| rect offset 2d | 1.57 | 3.75 | 2.40 | yes |
| polygon face | 0.19 | 0.38 | 2.01 | yes |
| extrude | 0.25 | 2.06 | 8.22 | yes |
| revolve | 0.33 | 2.79 | 8.60 | yes |
| loft | 2.40 | 24.85 | 10.34 | no (3961.4835 vs 3966.3515) |
| sweep | 1.39 | 38.84 | 27.90 | no (383.1496 vs 383.1497) |
| helix | 0.08 | 0.08 | 0.93 | yes |
| spline | 0.03 | 0.05 | 1.82 | yes |
| offset solid | 5.31 | 27.27 | 5.13 | yes |
| split box | 1.44 | 5.80 | 4.02 | yes |
| shell box | 2.81 | 4.32 | 1.54 | yes |
| tessellate sphere | 37.97 | 100.75 | 2.65 | yes |
| distance box-sphere | 0.47 | 0.67 | 1.44 | yes |
| point in solid | 0.04 | 0.01 | 0.27 | yes |
| edge positions | 0.55 | 10.37 | 18.92 | yes |
| face normal | 0.04 | 0.03 | 0.57 | yes |
| deepcopy | 0.06 | 0.03 | 0.47 | yes |
| move/rotate | 0.07 | 0.10 | 1.39 | yes |

## Against smlibpy (another Python layer over the same SMLib kernel)

[smlibpy](https://github.com/awv-informatik/smlibpy) 0.2.3 (`pip install
smlibpy`, kernel compiled in) exposes a build123d-like API over the same
SMLib kernel. `benchmarks/bench_smlibpy.py` runs the same operations through
it; medians in milliseconds, 7 repeats, same machine and day. The ratio
column is smlibpy over the build123d SMLib backend (below 1 = smlibpy
faster).

| Operation | OCCT ms | build123d+SMLib ms | smlibpy ms | smlibpy / build123d+SMLib | Same result |
|---|---:|---:|---:|---:|---|
| box | 0.09 | 0.40 | 0.19 | 0.47 | yes |
| sphere | 0.07 | 1.11 | 0.57 | 0.51 | yes |
| cylinder | 0.07 | 2.21 | 0.88 | 0.40 | yes |
| box volume | 0.08 | 0.31 | 0.00 | 0.00 | yes |
| sphere volume | 0.07 | 16.05 | 0.00 | 0.00 | yes |
| sphere area | 0.07 | 16.22 | 0.00 | 0.00 | yes |
| box bounding box | 0.01 | 0.02 | 0.01 | 0.31 | yes |
| box faces/edges/vertices | 0.16 | 0.09 | 0.01 | 0.10 | yes |
| fuse box+cylinder | 3.43 | 4.03 | 1.39 | 0.35 | yes |
| cut box-cylinder | 2.65 | 6.17 | n/a | - | - |
| common box&sphere | 3.01 | 6.99 | 4.85 | 0.69 | yes |
| cut box-sphere | 3.16 | 7.38 | 4.79 | 0.65 | yes |
| fuse 8 boxes | 35.49 | 16.82 | 13.13 | 0.78 | yes |
| fillet box edges | 9.98 | 14.49 | 28.40 | 1.96 | yes |
| chamfer box edges | 10.83 | 10.43 | 11.28 | 1.08 | yes |
| fillet cylinder edges | 2.40 | 6.05 | 16.98 | 2.81 | yes |
| sketch fuse rect+circle | 1.25 | 1.22 | 0.65 | 0.53 | yes |
| sketch cut rect-circle | 0.86 | 1.33 | 0.48 | 0.36 | yes |
| rect fillet 2d | 8.90 | 6.92 | 0.38 | 0.05 | yes |
| rect offset 2d | 1.57 | 2.50 | 0.70 | 0.28 | yes |
| polygon face | 0.18 | 0.09 | 0.05 | 0.56 | yes |
| extrude | 0.25 | 0.45 | 0.43 | 0.96 | yes |
| revolve | 0.32 | 1.92 | 2.47 | 1.29 | yes |
| loft | 2.36 | 3.54 | 3.19 | 0.90 | yes |
| sweep | 1.40 | 36.66 | 20.82 | 0.57 | yes |
| helix | 0.08 | 0.08 | 0.10 | 1.34 | no (153.7516 vs 153.7513) |
| spline | 0.03 | 0.02 | 0.09 | 4.84 | no (23.8395 vs 22.3593) |
| offset solid | 5.32 | 27.25 | 3.80 | 0.14 | no (8392.6843 vs 8448.0) |
| split box | 1.35 | 5.36 | 0.85 | 0.16 | yes |
| shell box | 2.79 | 4.24 | 5.15 | 1.21 | yes |
| tessellate sphere | 36.58 | 96.37 | 60.73 | 0.63 | yes |
| distance box-sphere | 0.47 | 0.67 | 19.43 | 28.95 | yes |
| point in solid | 0.04 | 0.01 | 0.01 | 0.55 | yes |
| edge positions | 0.56 | 10.44 | 4.72 | 0.45 | yes |
| face normal | 0.05 | 0.02 | 0.01 | 0.21 | yes |
| deepcopy | 0.06 | 0.03 | 0.05 | 1.47 | yes |
| move/rotate | 0.07 | 0.10 | 0.05 | 0.51 | yes |

How to read it:

- smlibpy's mass properties cost nothing at query time because it computes
  them when the body is made (its box takes 0.19 ms including volume, area
  and centre); the build123d backend integrates them on demand. Both get
  the same values.
- smlibpy is 2-3x faster on primitives and simple booleans, 3-20x faster on
  2D sketch operations (the build123d backend does 2D fillets and offsets in
  Python), 6x faster on planar splits and 7x faster on solid offsets (it
  returns the sharp-cornered offset: 8448 against the rounded 8393 that
  build123d's `offset` asks for by default).
- The build123d backend is 2x faster on fillets, 2.7x on cylinder fillets,
  on par on chamfers, lofts and extrusions, and 30x faster on body-to-body
  distances (0.67 ms against 21 ms).
- smlibpy refuses the cut of a box by a cylinder whose caps lie in the box's
  faces ("not a closed, manifold solid"); the build123d backend returns the
  expected 3644.
- Its spline is a different curve type (length 22.36 against 23.84 for the
  interpolating spline), so that row compares construction cost only.

## Where the time of a boolean goes, and the history switch

A stage-by-stage timing of `box - sphere` on the SMLib backend (fourth run
code, then the fifth):

| Stage | Fourth run ms | Fifth run ms |
|---|---:|---:|
| kernel merge (`SmMerge.ManifoldBoolean`) | 5.0 | 4.5 |
| bounding boxes of the operands and the result | 1.4 | 0.02 |
| record of what each input face, edge and vertex became | 1.0 | 0.2 |
| copies of the operands (the merge consumes its inputs) | 0.15 | 0.15 |
| clean (`unify_same_domain`) | 0.7 | 0.7 |
| whole operation through build123d | 7.9 | 5.7 |

- The bounding boxes were the tight ones, which SMLib computes afresh for
  every body (0.4 ms for a sphere, 1 ms for a cylinder, 2.3 ms for a
  filleted box); they only decide which bodies may touch, so the loose
  control-polygon box (microseconds) does now.
- The record was built by geometry: every input face, edge and vertex was
  dropped onto every result entity. The kernel can carry it instead: an
  `SmLongAttribute` on each entity of the working copy survives the merge,
  is copied onto both pieces of a split entity and kept on a trimmed one,
  and a new entity carries none. Reading the tags off the result is one
  pass. Inputs left without a tagged survivor (a face merged into a
  coplanar one keeps one tag only) still go through the geometric match.
- `SkipHistory` (next to `SkipClean` in `build123d.topology`) switches the
  record off for the booleans inside it: results then carry no history,
  `Select.LAST`/`Select.NEW` raise on them, and a builder answers by
  identity (everything not there before is new). With the tags it saves
  about 0.2 ms per boolean on SMLib and nothing on OCCT, whose algorithms
  keep their history whether or not it is read. Together with `SkipClean`
  the SMLib cut of a box by a cylinder drops from 3.3 ms to 1.8 ms, under
  OCCT's 2.6 ms; the remaining time is the kernel merge.
- Invalidating the operands instead of copying them would save the 0.15 ms
  of the copies and leave the Python objects holding consumed bodies (the
  kernel frees them), which is how the earlier worker crashes came about;
  it is not worth a switch.

## Reading the numbers

- Fourth run: primitives and extrusions got slower (box 0.4 to 1.4 ms, extrude
  0.45 to 2.2 ms) because every planar face is now put on an explicit plane
  with OCCT's parameterization after the kernel builds it (`ReplaceSurface`
  per face). Sketch booleans and polygon faces doubled for the same reason:
  faces are rebuilt with their loops' uv extents. Lofts of unlike sections
  now split the sections alike and skin them between shared rails (26 ms).

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
- Booleans with a sphere are 1.8x slower, box with cylinder on par (fuse) to
  1.2x (cut); fusing many bodies is 3x faster (bodies that do not touch are
  kept apart and merged only with what they overlap). The fifth run took the
  bounding boxes and the geometric history matching out of every boolean,
  see the section above; the rest is the kernel merge and the clean.
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
| 2026-10-13 (fifth run) | loose bounding boxes for the touch test, boolean history from kernel attributes instead of geometry matching, `SkipHistory` switch | cut box-sphere 8.2 to 5.6 ms, cut box-cylinder 7.3 to 2.9 ms, fuse 8 boxes 21.6 to 11.9 ms (3x faster than OCCT) |
| 2026-10-13 (fourth run) | OCCT-style planar frames (faces reframed on explicit planes), NURBS line segments, compatible-wire lofts, mitred sweeps, clean merging faces and edges | box 1.4 ms and extrude 2.2 ms (reframing cost), sketch booleans 2x slower, loft 26 ms through the compatible path, fuse box+cylinder on par |
