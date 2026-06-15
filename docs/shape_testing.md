# Shape Testing

This note documents the helper-based shape-testing stack introduced alongside the existing `ShapeTester`.

The new entry point is `ShapeContractTest`. It keeps the current VecGeom shape contracts replayable and debuggable, but splits them into reusable families that can run:

- per solid,
- per shape family,
- per helper family,
- per tier (`fast`, `medium`, `slow`),
- or as curated hand-picked reproductions through `manual_edge_cases`.

The existing `ShapeTester` and `shape_test*` executables still exist. This document describes the new helper-level system and the contract vocabulary shared with the reusable checks in `test/VecGeomTest/ShapeContractChecks.h`.

## Terminology

| Term | Meaning |
| --- | --- |
| `case_name` | One concrete configured solid, such as `box`, `tube_fullphi`, or `polycone_cms_like`. |
| `family` | A shape family used to group cases, such as `tube`, `cone`, or `sphere`. |
| `test_family` | One helper family inside `ShapeContractTest`, such as `contracts`, `normals`, or `distance_to_out`. |
| `tier` | One sampling profile: `fast`, `medium`, or `slow`. |
| sample group | The topology bucket of a sampled point: `inside`, `surface`, `outside`. `edge` exists in the model but is not generated automatically today. |
| replay | Rerunning one recorded sampled ray with `-replay_index` so the exact failing point and direction can be inspected. |
| manual edge case | A curated hand-written reproduction stored in `TestCaseManualEdgeCases.h`. |

## Test Families

| Family | Purpose | Inputs |
| --- | --- | --- |
| `contracts` | The current VecGeom surface/inside/outside navigation conventions. | sampled inside, surface, and outside rays |
| `normals` | Direct and propagated `Normal()` validity, unit length, and orientation checks. | sampled surface rays plus propagated inside-exit and outside-entry hits |
| `surface` | Surface-point and grazing-ray behavior. | sampled surface rays and derived grazing directions |
| `distance_to_out` | Inside-to-exit distance propagation checks. | sampled inside rays |
| `distance_to_in` | Outside-to-entry distance propagation checks. | sampled outside points plus paired inside targets |
| `safeties` | Safety magnitude and safety-sphere checks. | sampled inside and outside points |
| `hit_consistency` | Cross-checks between distances, propagated hit points, `Inside`, and `Normal`. | sampled inside and outside points |
| `manual_edge_cases` | Hand-picked reproductions for chosen solids, topologies, and helper families. | curated explicit rays or outside/inside point pairs |
| `all` | Run every sampled helper family above except `manual_edge_cases` in one process, reusing one sample cache. | same as selected solid/tier sample cache |

## Shape-Level Contracts

The tables below describe the public shape-level rules checked by the helper suite. They are written as API-level conventions, not as the internal implementation steps used by the tests. Failure reports still print an internal `convention_bit` from `ShapeConventionBit`, but that bitset is only the lower-level breakdown used to report which supporting check failed.

`solid_tolerance` is resolved per configured sampled solid. The default is
`vecgeom::kTolerance`. Planar-only solid families use `vecgeom::kTolerance`,
while second-order families use `vecgeom::kConeTolerance`. Larger values should
only be configured explicitly for a family that demonstrably needs them.

### 1. Classification and Wrong-Side Rules

| API / topology | Convention |
| --- | --- |
| `Inside(point)` on a sampled inside point | returns `kInside` |
| `Inside(point)` on a sampled surface point | returns `kSurface` |
| `Inside(point)` on a sampled outside point | returns `kOutside` |
| `DistanceToIn(point, dir)` from inside | wrong-side value |
| `DistanceToOut(point, dir)` from outside | wrong-side value |
| `SafetyToIn(point)` from inside | wrong-side value |
| `SafetyToOut(point)` from outside | wrong-side value |

### 2. Surface Distance Rules

| API / situation | Convention |
| --- | --- |
| `DistanceToIn(point, dir)` on a surface point with entering direction | zero within tolerance |
| `DistanceToIn(point, dir)` on a surface point with exiting direction | positive |
| `DistanceToOut(point, dir)` on a surface point with exiting direction | zero within tolerance |
| `DistanceToOut(point, dir)` on a surface point with entering direction | positive |
| `DistanceToOut(point, dir)` on any tested surface ray | finite |
| derived exact-grazing smooth-surface ray with tangential material continuation | `DistanceToIn` is zero within tolerance and `DistanceToOut` is finite and advancing |
| derived shallow-inward smooth-surface ray with material continuation `> kTolerance` | `DistanceToIn` has zero projected displacement within tolerance |
| derived shallow-outward smooth-surface ray | `DistanceToOut` has zero projected displacement within tolerance |
| any tested direction on a surface point | `DistanceToIn` and `DistanceToOut` are not both `<= kTolerance` |

The last rule is general. It is intentionally evaluated against `kTolerance`,
not the per-solid `solid_tolerance`, because it checks whether the tested ray
actually advances. Grazing directions are one important subset used by the
tests to probe it.

The finite `DistanceToOut`, derived shallow-ray, and exact tangential
material-continuation rules are enabled for all registered solid cases. The
`surface` helper derives exact-grazing rays for every smooth sampled surface
point. A deterministic sparse subset also checks shallow inward/outward rays and
exact tangential rays whose scale-aware forward probe is inside the solid,
covering concave second-order material continuation without doubling the
surface-family runtime.

Tangential material continuation is not limited to first-order inward motion.
For concave or hollow second-order boundaries, a ray with unresolved normal
motion can still enter material through curvature. In that case
`DistanceToIn` must accept the zero entry and `DistanceToOut` must ignore the
current zero root when looking for the next exit. Plane-like surface plateaus
and section seams are excluded from this exact-tangent rule because a farther
inside probe there represents a finite face/section transition, not immediate
curvature-driven material continuation.

### 3. Safety Rules

| API / topology | Convention |
| --- | --- |
| `SafetyToIn(point)` on a surface point | `<= solid_tolerance` |
| `SafetyToOut(point)` on a surface point | `<= solid_tolerance` |
| `SafetyToOut(point)` from inside | positive |
| `SafetyToIn(point)` from outside | positive |
| positive safety from the correct side | does not exceed the corresponding distance within tolerance |
| safety sphere grown from the correct side | does not cross the boundary |

For surface points, the helper checks keep the historical upper-bound rule:
surface safety must not exceed `solid_tolerance`. The checks do not require an
exact zero just to prove a surface classification.

### 4. Normal Rules

| API / situation | Convention |
| --- | --- |
| `Normal(point, n)` on a surface point | returns `true` |
| `Normal(point, n)` on a surface point | returns a unit vector within tolerance |
| `Normal(point, n)` on a surface point | points topologically outward |
| propagated boundary hit reached from the correct side | if used for a normal-oriented rule, it is a valid surface point and the normal keeps the same outward convention |

### 5. Distance-Propagation Rules

| API / situation | Convention |
| --- | --- |
| `DistanceToOut(point, dir)` from inside | positive and finite |
| `DistanceToIn(point, dir)` from outside toward the solid | positive and finite |
| positive finite `DistanceToOut` | propagates to a point classified as `kSurface` |
| positive finite `DistanceToIn` | propagates to a point classified as `kSurface` |
| outside-to-inside ray used for entry checks | enters before the paired inside target point along the ray |

### 6. Boundary-Consistency Rules

| Situation | Convention |
| --- | --- |
| point reached by a positive finite entry or exit distance | is on the boundary according to `Inside` |
| safety values at a propagated boundary point | are within boundary tolerance |
| `Normal`, `DistanceToIn`, and `DistanceToOut` evaluated at a propagated boundary point | agree on boundary orientation and sign expectations |
| outside-to-entry-to-exit path | remains self-consistent across entry point, interior propagation, and exit point |

### 7. Manual Edge Cases (`manual_edge_cases`)

This family has no contract catalog of its own. It reuses one of the sampled helper families on a curated point/direction (or outside point plus paired inside target) declared in `TestCaseManualEdgeCases.h`.

Use this family when:

- a random sample already exposed a bug and you want a permanent hand-picked reproduction,
- you need to exercise a real edge or corner case that is hard to sample automatically,
- you want one canonical debugging ray for a solid/method/topology combination.

## Sampling Tiers

Current default sampling statistics are:

| Tier | Default points | Intended use |
| --- | --- | --- |
| `fast` | `10000` | merge-request / smoke coverage |
| `medium` | `100000` | heavier regular validation |
| `slow` | `1000000` | nightly / stress validation |

By default the three tiers use the same deterministic sample stream for a given solid and differ only by the number of points. A solid can opt into an independent `medium` or `slow` profile through `TestCaseSolid`, but the default policy is:

- same seed,
- same `stream_id`,
- same outside-point sampling tuning,
- only higher statistics in larger tiers.

This makes `medium` a superset of `fast`, and `slow` a superset of both, unless a solid explicitly overrides that behavior.

## CTest Registration

CTest registers only one tier at configure time through:

```cmake
-DVECGEOM_SHAPE_CONTRACT_CTEST_TIER=<fast|medium|slow>
```

The default is `medium`.

The local helper executable is always available for all tiers:

```bash
.../ShapeContractTest -tier fast   ...
.../ShapeContractTest -tier medium ...
.../ShapeContractTest -tier slow   ...
```

CTest registers every configured solid case with all sampled helper families.
If a new case exposes a real bug, fix the bug before adding that case to the
CTest registry.

## Building and Running

### Configure the CTest Tier

Choose the CTest tier at configure time with:

```bash
cmake -S . -B <build-dir> -DVECGEOM_SHAPE_CONTRACT_CTEST_TIER=<fast|medium|slow>
cmake --build <build-dir>
```

### Run Through CTest

```bash
cd <build-dir>
ctest --output-on-failure -R 'ShapeContractTest:medium:'
ctest --output-on-failure -R 'ShapeContractTest:slow:'
```

### Run the Executable Directly

```bash
<build-dir>/test/ShapeContractTest -help
<build-dir>/test/ShapeContractTest -list_cases
<build-dir>/test/ShapeContractTest -list_families
<build-dir>/test/ShapeContractTest -list_test_families
<build-dir>/test/ShapeContractTest -list_manual_cases
```

Examples:

```bash
<build-dir>/test/ShapeContractTest -tier fast -test_family contracts -case_name box
<build-dir>/test/ShapeContractTest -tier fast -test_family all -family tube
<build-dir>/test/ShapeContractTest -tier slow -test_family normals -case_name sphere_section
<build-dir>/test/ShapeContractTest -tier fast -test_family manual_edge_cases -case_name box
```

With no arguments, the executable runs the `fast` `contracts` family for every configured solid.

## Debugging Failures

### Sampled Families

When a sampled family fails, the error report prints a **How to debug** block containing:

- the first recorded occurrence,
- a copy-pasteable replay command,
- the replay function,
- the evaluator function,
- the geometry API involved,
- the implementation-oriented function name for `gdb`,
- any supporting API used in the check.

Typical workflow:

1. Run the failing test directly or through CTest.
2. Copy the `replay_command`.
3. Rerun it directly:

```bash
<build-dir>/test/ShapeContractTest \
  -tier slow \
  -test_family normals \
  -case_name polycone_nearly_repeated_z \
  -seed 50 \
  -stream_id 28 \
  -npoints 1000000 \
  -replay_index 19901
```

4. If needed, debug with `gdb` and the printed `implementation_function`.

`-replay_index` is only valid for a single solid and a single sampled helper family.

### Manual Edge Cases

Manual cases are already explicit reproductions. Use:

```bash
<build-dir>/test/ShapeContractTest \
  -tier fast \
  -test_family manual_edge_cases \
  -case_name box \
  -manual_case_name box_inside_exit_positive_x
```

For manual cases, use `-manual_case_name`, `-manual_method`, and `-manual_topology` instead of `-replay_index`.

## CLI Summary

| Option | Meaning |
| --- | --- |
| `-tier <fast|medium|slow>` | sampling tier |
| `-test_family <...>` | helper family |
| `-case_name <name|all>` | one configured solid case |
| `-family <name|all>` | all solids in one shape family |
| `-npoints <count>` | override the configured sample count |
| `-seed <seed>` | override the deterministic base seed |
| `-stream_id <id>` | override the logical deterministic sub-stream |
| `-replay_index <index>` | replay one sampled ray |
| `-grazing_tolerance <value>` | tilt grazing rays in the `surface` family |
| `-manual_case_name <name|all>` | select one curated manual case |
| `-manual_method <...|all>` | filter manual cases by helper family |
| `-manual_topology <inside|surface|edge|outside|all>` | filter manual cases by topology |
| `-list_cases` | print configured solid cases |
| `-list_families` | print configured shape families |
| `-list_test_families` | print configured helper families |
| `-list_manual_cases` | print curated manual edge cases |
| `-help` | print usage |

## Adding a New Sampled Solid

1. Add a `Make...TestSolid()` factory in the matching family header under `test/VecGeomTest/`.
2. Add the case to `GetTestCaseSolids()` in `TestCaseSolids.h`, grouped with the same shape family.
3. Give it:
   - a stable `case_name`,
   - a factory function,
   - an `implementation_debug_type`,
   - default `fast`, `medium`, and, if needed, `slow` sampling profiles.
4. Add the same `case_name` to `SHAPE_CONTRACT_SOLID_CASES` in `test/CMakeLists.txt`.
5. Rebuild and run:

```bash
cmake -S . -B <build-dir> -DVECGEOM_SHAPE_CONTRACT_CTEST_TIER=medium
cmake --build <build-dir>
<build-dir>/test/ShapeContractTest -tier fast   -test_family contracts -case_name <new_case>
<build-dir>/test/ShapeContractTest -tier medium -test_family contracts -case_name <new_case>
<build-dir>/test/ShapeContractTest -tier slow   -test_family contracts -case_name <new_case>
cd <build-dir>
ctest --output-on-failure -R 'ShapeContractTest:'
```

6. If the case exposes a real bug, fix the bug before adding the case to
   `SHAPE_CONTRACT_SOLID_CASES`.

## Adding a New Manual Edge Case

1. Choose an existing `solid_case_name` from `TestCaseSolids.h`.
2. Add a `ManualEdgeCase` entry in `TestCaseManualEdgeCases.h`, grouped with the same shape family.
3. Set:
   - `target_family_name` to the helper family you want to reproduce,
   - `topology` to `inside`, `surface`, `edge`, or `outside`,
   - `point`,
   - either `direction`, or `target_point` with `uses_target_point = true` for outside-to-inside style reproductions.
4. For `surface` manual cases, set `grazing_tolerance` if you want the case to use a near-grazing direction instead of exact grazing.
5. Run:

```bash
<build-dir>/test/ShapeContractTest -list_manual_cases
<build-dir>/test/ShapeContractTest \
  -test_family manual_edge_cases \
  -case_name <solid_case_name> \
  -manual_case_name <new_manual_case>
```

6. Manual cases registered for CTest must pass. Keep known-failing
   reproductions out of the registered manual CTest set until the corresponding
   bug is fixed.

## Code Map

| Path | Role |
| --- | --- |
| `test/core/ShapeContractTest.cpp` | CLI runner, family dispatch, replay, in-process sample cache |
| `test/VecGeomTest/ShapeContractChecks.h` | reusable helper-family implementations and replay helpers |
| `test/VecGeomTest/TestCaseSolids.h` | sampled solid registry |
| `test/VecGeomTest/TestCaseManualEdgeCases.h` | curated manual reproductions |
| `test/CMakeLists.txt` | CTest registration for sampled and manual shape-contract cases |
| `test/VecGeomTest/ShapeTester.*` | legacy wrapper and whole-solid compatibility path |
