# Where SMLib beats OCCT, and where it does not

Observations from building the `smcore` backend of build123d on NVIDIA's SMLib
(branch `kernel-smlib`, 2026-10-13). Timings refer to
[kernel_comparison.md](kernel_comparison.md); the rest comes from the test
suite and from direct probes of the two kernels.

## Where SMLib genuinely came out ahead

- **Curve length accuracy.** On the rational Bezier in `test_bezier`,
  OCCT's `GCPnts_AbscissaPoint` length is off by about 2.5e-3; SMLib's
  `Length` agrees with a 20,000-point quadrature to 1e-9. The test's expected
  value was recorded from OCCT, so SMLib fails that test by being right.
- **Point classification and distance solves.** `Point3DClassify` runs in
  0.01 ms against OCCT's 0.04 ms, and the topology solvers (point, curve and
  body against body) were reliable across every pairing tried. The one solver
  mode that refuses curves lying on each other has a sampled fallback.
- **Fusing many bodies.** Fusing 8 boxes onto a plate runs 2x faster than
  OCCT (16 ms vs 35 ms), and ring compounds of 20 boxes fuse correctly once
  bodies that do not touch are kept apart. The merge is quick on simple
  operands.
- **Imprint and non-manifold merge.** `SM_BO_IMPRINT` gives a crash-free
  section tool and `SM_BO_MERGE` a general splitter: splitting a solid by any
  closed sheet falls out of the region structure. OCCT needs
  `BRepAlgoAPI_Splitter` plus a half-space construction for the same thing.
- **Analytic chamfers and tapers as primitives.** `CreateTaperExtrude` builds
  a drafted extrusion directly, keeping cones as cones, and the fillet
  executive takes per-edge parameters in one call. Chamfers benchmark on par
  with OCCT and box fillets within 1.5x.
- **Uniform curve representation.** Every curve, circles and ellipses
  included, is a NURBS with one evaluation, knot and control-polygon API.
  Sweeps, skins, projections and the Green's-theorem integrations were
  straightforward to write; there is no Geom/BRep/Adaptor split to navigate.

## Where SMLib looks faster but the backend is doing the work

- 2D fillets (0.8x), 2D offsets, polygon faces (0.5x), traversal, deep copies
  and face normals are faster because the backend does them in Python on
  cheap handles or straight on the curve objects. These are wins of the
  facade, not of the kernel.

## Where SMLib is clearly behind

- Primitives are 4-30x slower, sweeps 26x, solid offsets 5x, splits 4x,
  tessellation 2.6x.
- Mass properties: the kernel's integrator is precise but costs 150 ms on a
  sphere at the accuracy the tests need, and its per-face variant returns
  zero flux on trimmed walls and refuses sphere faces. The backend integrates
  mass properties itself.
- Fillets fail on cases OCCT handles (the maker-coin detent edges at radius 2,
  under every tolerance setting) and booleans fail on coincident-face cases
  (a key cap minus its own scaled copy).
- Chamfer rails are approximate NURBS (3e-5 off a circle, whatever the
  approximation tolerance), not circles.
- No text, no hidden-line removal, no STEP, and several binding-level traps:
  non-const arrays of pointers returned after a consuming call dangle, the
  n-sided patch creator segfaults through the bindings, and the intersection
  edge list of a merge is not safe to read.

## Summary

SMLib is more accurate on curve lengths, has very good solvers and point
classification, and its imprint and merge machinery is a nicer foundation
for sections and splits. For everything built on top of that it is slower
and less robust than OCCT, and most of the places where the comparison
shows it winning on speed are the thin Python layer rather than the kernel.
