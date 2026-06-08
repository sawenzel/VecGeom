/**
 * @file TestCaseManualEdgeCases.h
 * @brief Curated hand-picked rays for explicit helper-family reproduction.
 *
 * Unlike the sampled registries, these cases are meant to reproduce one chosen
 * solid, topology, and helper family with a hand-authored point/direction (or
 * outside-point/inside-target pair). ShapeContractTest exposes them through the
 * `manual_edge_cases` family and the `-manual_*` selectors.
 *
 * See docs/shape_testing.md for the public manual-edge-case workflow and the
 * extension rules followed by the registry below.
 */
//
// How to add a manual edge case:
// 1. Pick an existing solid from TestCaseSolids.h and keep the new case grouped
//    with the same shape family below.
// 2. Choose the helper method to reproduce via target_family_name.
// 3. Choose the topology bucket via topology:
//      inside / surface / edge / outside
// 4. For outside distance_to_in / hit_consistency style cases, set
//    uses_target_point=true and provide a paired inside target_point instead of
//    an explicit direction.
// 5. Keep the case enabled here. If it exposes a real bug that is not fixed in
//    the same merge request, disable only the manual_edge_cases CTest family
//    for that solid in test/CMakeLists.txt.

#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASEMANUALEDGECASES_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASEMANUALEDGECASES_HH

#include <algorithm>
#include <string>
#include <vector>

#include "VecGeomTest/ShapeCheckResult.h"

namespace vecgeom {
namespace test {

/**
 * @brief One curated hand-picked replay case for the manual helper family.
 */
struct ManualEdgeCase {
  const char *name               = "";
  const char *solid_case_name    = "";
  const char *target_family_name = "";
  ShapeSampleCategory topology   = ShapeSampleCategory::kUnknown;
  Vec_t point;
  Vec_t direction;
  bool uses_target_point = false;
  Vec_t target_point;
  Precision grazing_tolerance = 0.;
  const char *description     = "";
};

/**
 * @brief Return the effective ray direction for one manual edge case.
 *
 * Cases targeting an outside-to-inside entry store a `target_point` instead of
 * an explicit direction so the intended topology is obvious in the registry.
 */
inline Vec_t EffectiveManualEdgeCaseDirection(const ManualEdgeCase &manual_case)
{
  if (!manual_case.uses_target_point) return manual_case.direction;

  const Vec_t delta         = manual_case.target_point - manual_case.point;
  const Precision magnitude = delta.Mag();
  return magnitude > 0. ? delta.Unit() : Vec_t(0., 0., 0.);
}

/**
 * @brief Return the full curated manual edge-case registry.
 */
inline const std::vector<ManualEdgeCase> &GetManualEdgeCases()
{
  static const std::vector<ManualEdgeCase> cases = {
      // Box
      {"box_surface_grazing_positive_x", "box", "surface", ShapeSampleCategory::kSurface, Vec_t(10., 0., 0.),
       Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0., "Smooth +X face grazing ray on the box surface."},
      {"box_surface_entering_positive_x", "box", "contracts", ShapeSampleCategory::kSurface, Vec_t(10., 0., 0.),
       Vec_t(-1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Surface entering ray on the +X box face for convention checks."},
      {"box_surface_normal_positive_x", "box", "normals", ShapeSampleCategory::kSurface, Vec_t(10., 0., 0.),
       Vec_t(-1., 0., 0.), false, Vec_t(0., 0., 0.), 0., "Surface normal check on the +X box face."},
      {"box_inside_exit_positive_x", "box", "distance_to_out", ShapeSampleCategory::kInside, Vec_t(0., 0., 0.),
       Vec_t(1., 0., 0.), false, Vec_t(0., 0., 0.), 0., "Inside radial exit ray towards the +X box face."},
      {"box_outside_entry_positive_x", "box", "distance_to_in", ShapeSampleCategory::kOutside, Vec_t(12., 0., 0.),
       Vec_t(0., 0., 0.), true, Vec_t(0., 0., 0.), 0., "Outside entry ray targeting the box center from +X."},
      {"box_inside_safety_positive_x", "box", "safeties", ShapeSampleCategory::kInside, Vec_t(0., 0., 0.),
       Vec_t(1., 0., 0.), false, Vec_t(0., 0., 0.), 0., "Inside safety check paired with a +X exit ray in the box."},
      {"box_outside_hit_consistency_positive_x", "box", "hit_consistency", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 0.), Vec_t(0., 0., 0.), true, Vec_t(0., 0., 0.), 0.,
       "Outside -> inside propagated hit-consistency ray for the box."},
      // Tube
      {"tube_fullphi_surface_grazing_outer_r", "tube_fullphi", "surface", ShapeSampleCategory::kSurface,
       Vec_t(10., 0., 0.), Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Outer-radius grazing ray on the full-phi tube."},
      {"tube_fullphi_surface_entering_outer_r", "tube_fullphi", "contracts", ShapeSampleCategory::kSurface,
       Vec_t(10., 0., 0.), Vec_t(-1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Surface entering ray on the outer radius of the full-phi tube."},
      {"tube_fullphi_surface_normal_outer_r", "tube_fullphi", "normals", ShapeSampleCategory::kSurface,
       Vec_t(10., 0., 0.), Vec_t(-1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Surface normal check on the outer radius of the full-phi tube."},
      {"tube_fullphi_inside_radial_exit", "tube_fullphi", "distance_to_out", ShapeSampleCategory::kInside,
       Vec_t(7.5, 0., 0.), Vec_t(1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Inside radial exit ray from the tube shell mid-radius."},
      {"tube_fullphi_outside_radial_entry", "tube_fullphi", "distance_to_in", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 0.), Vec_t(0., 0., 0.), true, Vec_t(7.5, 0., 0.), 0.,
       "Outside radial entry ray targeting a known inside point in the tube shell."},
      {"tube_fullphi_inside_safety_radial", "tube_fullphi", "safeties", ShapeSampleCategory::kInside,
       Vec_t(7.5, 0., 0.), Vec_t(1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Inside safety check paired with a radial exit ray in the tube shell."},
      {"tube_fullphi_outside_hit_consistency_radial", "tube_fullphi", "hit_consistency", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 0.), Vec_t(0., 0., 0.), true, Vec_t(7.5, 0., 0.), 0.,
       "Outside -> inside propagated hit-consistency ray for the full-phi tube."},
      // Orb
      {"orb_surface_grazing_positive_x", "orb", "surface_exit", ShapeSampleCategory::kSurface, Vec_t(8., 0., 0.),
       Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0., "Smooth +X surface grazing exit ray on the orb."},
      {"orb_surface_shallow_inward_positive_x", "orb", "contracts", ShapeSampleCategory::kSurface, Vec_t(8., 0., 0.),
       Vec_t(-1.e-6, 0.9999999999995, 0.), false, Vec_t(0., 0., 0.), 0.,
       "Smooth +X surface ray with a finite shallow inward radial component."},
      // Hype
      {"hype_top_cap_tolerance_outward_exit", "hype", "surface_exit", ShapeSampleCategory::kSurface,
       Vec_t(60., 0., 50. + vecgeom::kTolerance), Vec_t(0., 0., 1.), false, Vec_t(0., 0., 0.), 0.,
       "Tolerated top-cap surface start just outside z must have a zero outward DistanceToOut."},
      {"hype_bottom_cap_tolerance_outward_exit", "hype", "surface_exit", ShapeSampleCategory::kSurface,
       Vec_t(60., 0., -50. - vecgeom::kTolerance), Vec_t(0., 0., -1.), false, Vec_t(0., 0., 0.), 0.,
       "Tolerated bottom-cap surface start just outside z must have a zero outward DistanceToOut."},
      // Tube
      {"tube_fullphi_cap_grazing_entry_continuation", "tube_fullphi", "grazing_entry", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 20.), Vec_t(-1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray lies on the top cap plane and crosses the outer ring; if DistanceToIn accepts that grazing entry, "
       "DistanceToOut must carry the ray across the cap annulus to the next ring."},
      {"tube_fullphi_cap_near_grazing_entry_continuation", "tube_fullphi", "grazing_entry",
       ShapeSampleCategory::kOutside, Vec_t(12., 0., 20.), Vec_t(-1., 0., -1.e-10), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance inward z component on the top cap plane; near-grazing entries must keep the "
       "same non-zero DistanceToOut continuation convention as exact grazing."},
      {"tube_fullphi_cap_near_grazing_outward_miss", "tube_fullphi", "grazing_miss", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 20.), Vec_t(-1., 0., 1.e-10), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance outward z component on the top cap plane; it must not report a cap-plane "
       "DistanceToIn hit because there is no owned continuation interval."},

      // Paraboloid
      {"paraboloid_side_surface_normal", "paraboloid", "normals", ShapeSampleCategory::kSurface,
       Vec_t(-5.4537221697500797, -5.8275575265773867, -1.3426524280024168),
       Vec_t(-0.38200388098470184, -0.89699774286683653, 0.2224142176310373), false, Vec_t(0., 0., 0.), 0.,
       "Parabolic-side surface replay where the normal must point topologically outward."},
      {"paraboloid_side_surface_grazing", "paraboloid", "surface", ShapeSampleCategory::kSurface,
       Vec_t(-5.4537221697500797, -5.8275575265773867, -1.3426524280024168),
       Vec_t(-0.38200388098470184, -0.89699774286683653, 0.2224142176310373), false, Vec_t(0., 0., 0.), 0.,
       "Parabolic-side grazing replay where DistanceToIn and DistanceToOut must not both collapse to zero."},
      {"paraboloid_side_surface_shallow_entry", "paraboloid", "contracts", ShapeSampleCategory::kSurface,
       Vec_t(8.2462112512353212, 0., 0.), Vec_t(-1.0186502561360584e-11, 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Parabolic-side surface ray with only sub-tolerance material continuation must not be accepted as an entry."},
      {"paraboloid_top_edge_tangent", "paraboloid", "surface", ShapeSampleCategory::kEdge, Vec_t(10., 0., 10.),
       Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Top cap/parabolic-side edge tangent must not report both DistanceToIn and DistanceToOut as zero."},
      {"paraboloid_bottom_edge_tangent", "paraboloid", "surface", ShapeSampleCategory::kEdge, Vec_t(6., 0., -10.),
       Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Bottom cap/parabolic-side edge tangent must not report both DistanceToIn and DistanceToOut as zero."},

      // Sphere
      {"sphere_narrow_phi_phi_plane_grazing", "sphere_narrow_phi", "surface", ShapeSampleCategory::kSurface,
       Vec_t(7.0183278628341847, 3.0503842933005898, -3.1897304680763678),
       Vec_t(0.58396691489260943, 0.31306753242111229, 0.74898021499540546), false, Vec_t(0., 0., 0.), 0.,
       "Narrow-phi sphere replay where the generated phi-plane grazing ray must not report both DistanceToIn and "
       "DistanceToOut as zero."},
      {"sphere_section_phi_plane_grazing", "sphere_section", "surface", ShapeSampleCategory::kSurface,
       Vec_t(-6.4694147812101477, 11.205355096293074, -10.260967364622779),
       Vec_t(-0.45590986957403301, 0.36092696892014486, 0.81355879562027045), false, Vec_t(0., 0., 0.), 0.,
       "Section sphere replay where the generated phi-plane grazing ray must not report both DistanceToIn and "
       "DistanceToOut as zero."},

      // Boolean
      {"boolean_intersection_rotated_boxes_rotated_face_normal", "boolean_intersection_rotated_boxes", "normals",
       ShapeSampleCategory::kSurface, Vec_t(0.20710678118654757, -0.79289321881345243, 0.),
       Vec_t(0.70710678118654757, 0.70710678118654757, 0.), false, Vec_t(0., 0., 0.), 0.,
       "Surface point on the rotated right constituent of an intersection; normal selection must compute "
       "SafetyToOut in constituent-local coordinates."},
      {"boolean_nested_transformed_subtraction_inner_cut_grazing", "boolean_nested_transformed_subtraction", "surface",
       ShapeSampleCategory::kSurface, Vec_t(2.8, 0., 0.), Vec_t(0., 0., 1.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the transformed cutter inner face that remains as the nested subtraction boundary."},
      // These polyhedron-left subtraction probes complement random sampling by
      // hitting edge starts, shared planes, and grazing rays that are otherwise
      // zero-measure topologies.
      {"boolean_subtraction_polyhedron_z_slab_shared_edge_grazing", "boolean_subtraction_polyhedron_exact_z_slab",
       "surface", ShapeSampleCategory::kEdge, Vec_t(5., 0., 2.), Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the shared z-slab cut and outer polyhedron side, then grazes along the edge-like boundary."},
      {"boolean_subtraction_polyhedron_z_slab_shared_plane_entering", "boolean_subtraction_polyhedron_exact_z_slab",
       "contracts", ShapeSampleCategory::kSurface, Vec_t(0., 0., 2.), Vec_t(0., 0., 1.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the shared z-slab cut plane and enters the remaining upper polyhedron component."},
      {"boolean_subtraction_polyhedron_radial_shell_top_edge_grazing",
       "boolean_subtraction_polyhedron_exact_radial_shell", "surface", ShapeSampleCategory::kEdge, Vec_t(3., 0., 5.),
       Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the top edge of the inner prism left after subtracting the exactly matching radial shell."},
      {"boolean_subtraction_polyhedron_side_cut_tube_edge_grazing", "boolean_subtraction_polyhedron_side_cut_tube",
       "surface", ShapeSampleCategory::kEdge, Vec_t(5., 0.714142842854285, 0.), Vec_t(0., 0., 1.), false,
       Vec_t(0., 0., 0.), 0.,
       "Ray starts on the intersection edge between the left polyhedron side and the subtracted tube opening."},
      {"boolean_subtraction_polyhedron_phi_seam_rotated_top_edge_grazing",
       "boolean_subtraction_polyhedron_phi_seam_rotated", "surface", ShapeSampleCategory::kEdge,
       Vec_t(2.2275163104709197, 1.1349762493488669, 5.), Vec_t(0., 0., 1.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the rotated cutter phi seam where it meets the shared top z plane."},
      {"boolean_subtraction_polyhedron_near_coincident_cut_plane_grazing",
       "boolean_subtraction_polyhedron_near_coincident", "surface", ShapeSampleCategory::kSurface,
       Vec_t(2., 0., -20. * vecgeom::kConeTolerance), Vec_t(1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the near-coincident cutter top plane and grazes along that Boolean boundary."},

      // Generic polycone
      {"generic_polycone_irregular_surface_entering_outer_shell", "generic_polycone_irregular", "contracts",
       ShapeSampleCategory::kSurface, Vec_t(2.2348126399610928, 8.7181198360926615, 0.5),
       Vec_t(0.2898461953301249, -0.7252236297254806, 0.62453172052382588), false, Vec_t(0., 0., 0.), 0.,
       "Surface-entering replay ray on the irregular generic polycone outer shell where DistanceToIn and SafetyToIn "
       "previously stayed positive on a surface point."},
      {"generic_polycone_irregular_outside_entry_outer_shell_upper", "generic_polycone_irregular", "distance_to_in",
       ShapeSampleCategory::kOutside, Vec_t(-19.158840713111452, -33.920580785138945, 13.220637837805752),
       Vec_t(0., 0., 0.), true, Vec_t(5.8765881963591688, -1.3579933407038025, 2.3590887344045499), 0.,
       "Outside-to-inside replay ray whose ApproachSolid point landed near the upper outer shell and previously "
       "collapsed to DistanceToIn == 0."},
      {"generic_polycone_irregular_outside_entry_outer_shell_mid", "generic_polycone_irregular", "distance_to_in",
       ShapeSampleCategory::kOutside, Vec_t(-43.620431324114101, 33.367883641868751, -47.743502220288306),
       Vec_t(0., 0., 0.), true, Vec_t(-6.9768590480932753, -1.9493795535573799, 4.0939785634756207), 0.,
       "Outside-to-inside replay ray whose ApproachSolid point landed near the mid outer shell and previously "
       "collapsed to DistanceToIn == 0."},
      {"generic_polycone_irregular_shallow_surface_entry_outer_shell", "generic_polycone_irregular", "contracts",
       ShapeSampleCategory::kSurface, Vec_t(-5.1352140257766186, 7.391182482398392, 0.12297458592897575),
       Vec_t(-0.18865236894158444, -0.13111298697105264, -0.97325210934282436), false, Vec_t(0., 0., 0.), 0.,
       "Shallow inward ray from the exact outer shell where the section cone kernel reported the later crossing "
       "instead of the zero DistanceToIn entry."},
      {"generic_polycone_irregular_inside_near_outer_shell_safety_to_in", "generic_polycone_irregular", "contracts",
       ShapeSampleCategory::kInside, Vec_t(7.9604176988052195, -2.9783436513347485, 0.30039621002943101),
       Vec_t(-0.17271031333314549, 0.27234560960769244, 0.94657224584063371), false, Vec_t(0., 0., 0.), 0.,
       "Inside point close to the irregular outer shell where SafetyToIn must keep the wrong-side negative "
       "convention instead of being clamped to zero."},
      {"generic_polycone_irregular_lower_corner_entering_to_out", "generic_polycone_irregular", "surface",
       ShapeSampleCategory::kSurface, Vec_t(-0.037247392199863581, 0.99930607546347894, 0.),
       Vec_t(0.5814130076837466, -0.20624047999702957, 0.78703480158550454), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the exposed lower RZ corner and points into the first section; DistanceToOut must skip the "
       "current boundary and return the next exit."},
      {"generic_polycone_zigzag_top_corner_entering_to_out", "generic_polycone_zigzag_profile", "surface",
       ShapeSampleCategory::kSurface, Vec_t(-0.91076575325808495, 9.9584389218216582, 140.),
       Vec_t(-0.75238141223081711, -0.57721172253262865, -0.3174095743680031), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts on the exposed top corner of the zig-zag contour and points into the last section; DistanceToOut "
       "must not report the starting surface."},
      {"cuttube_section_inner_cut_plane_grazing", "cuttube_section_inner", "surface", ShapeSampleCategory::kSurface,
       Vec_t(4.1438716887445688, 0., -0.20947942872035785),
       Vec_t(0.46997756897067333, 0.5375170920854937, 0.70014031478009531), false, Vec_t(0., 0., 0.), 0.,
       "Smooth cut-plane surface replay whose grazing direction previously drove the embedded infinite-z tube path "
       "through invalid z-plane arithmetic."},
      {"cone_thin_shell_surface_entering_outer_lower_ring", "cone_thin_shell", "contracts",
       ShapeSampleCategory::kSurface, Vec_t(-43.093354838211269, -25.356710388709519, -200.0),
       Vec_t(0.90448163727274844, 0.15468343036337337, -0.39747453277842609), false, Vec_t(0., 0., 0.), 0.,
       "Surface-entering replay ray on the lower outer ring of the thin-shell cone from the 10M-point stress run."},
      {"cone_fullphi_cap_grazing_entry_continuation", "cone_fullphi", "grazing_entry", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 12.), Vec_t(-1., 0., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray lies on the upper cap plane and crosses the outer conical ring; a finite grazing entry must be "
       "matched by a non-zero cap-annulus DistanceToOut continuation."},
      {"cone_fullphi_cap_near_grazing_entry_continuation", "cone_fullphi", "grazing_entry",
       ShapeSampleCategory::kOutside, Vec_t(12., 0., 12.), Vec_t(-1., 0., -1.e-10), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance inward z component on the upper cap plane; the finite near-grazing entry must "
       "still hand off to a non-zero DistanceToOut continuation."},
      {"cone_fullphi_cap_near_grazing_outward_miss", "cone_fullphi", "grazing_miss", ShapeSampleCategory::kOutside,
       Vec_t(12., 0., 12.), Vec_t(-1., 0., 1.e-10), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance outward z component on the upper cap plane; it must not report a cap-plane "
       "DistanceToIn hit because there is no owned continuation interval."},
      {"cone_narrow_phi_outside_phi_entry", "cone_narrow_phi", "distance_to_in", ShapeSampleCategory::kOutside,
       Vec_t(2.8232263578195393, 1.0145900317433738, 0.0), Vec_t(0., 0., 0.), true,
       Vec_t(2.812939466458048, 1.0427710957073528, 0.0), 0.,
       "Outside-to-inside entry ray crossing the start-phi plane of the narrow-phi hollow cone."},
      {"cone_narrow_phi_surface_edge_zero_distances", "cone_narrow_phi", "surface", ShapeSampleCategory::kSurface,
       Vec_t(0.61463510936581522, 0.22435935479431501, -8.7655697525015839),
       Vec_t(-0.59316595886414847, 0.71320369634116421, -0.37348980276585314), false, Vec_t(0., 0., 0.), 0.,
       "Surface-family replay ray on the narrow-phi cone where DistanceToIn and DistanceToOut both collapse to zero "
       "in the 10M-point stress run."},
      {"polycone_nearly_repeated_z_inside_exit_shared_plane", "polycone_nearly_repeated_z", "normals",
       ShapeSampleCategory::kInside, Vec_t(5.4792351593165733, -34.959087641413987, 34.407379510600208),
       Vec_t(0.54113638620723326, 0.66772399699134932, -0.51119084045439689), false, Vec_t(0., 0., 0.), 0.,
       "Inside-exit replay ray from the 10M stress run where DistanceToOut lands on the nearly repeated shared-z "
       "plane but the propagated boundary point is still classified inside."},
      {"polycone_two_section_sharp_jump_surface_exit_exposed_annulus", "polycone_two_section_sharp_jump", "contracts",
       ShapeSampleCategory::kSurface, Vec_t(20., 0., -20.), Vec_t(0., 0., -1.), false, Vec_t(0., 0., 0.), 0.,
       "Surface-exit ray on the exposed annulus of a repeated-z transition where the boundary point belongs only to "
       "the upper section and DistanceToOut must still return zero instead of wrong-side -1."},
      {"polycone_two_section_sharp_jump_transition_circle_grazing", "polycone_two_section_sharp_jump", "surface",
       ShapeSampleCategory::kEdge, Vec_t(70., 0., -20.), Vec_t(0., 1., 0.), false, Vec_t(0., 0., 0.), 0.,
       "Ray starts exactly on the outer circle of the exposed transition annulus; the grazing surface convention must "
       "not report both DistanceToIn and DistanceToOut as zero."},
      {"polycone_two_section_sharp_jump_shared_plane_grazing_entry_continuation", "polycone_two_section_sharp_jump",
       "grazing_entry", ShapeSampleCategory::kOutside, Vec_t(80., 0., -20.), Vec_t(-1., 0., 0.), false,
       Vec_t(0., 0., 0.), 0.,
       "Outside ray lies on a repeated-z transition plane. If the polycone accepts the grazing entry, the chosen hit "
       "must be the one with a non-zero DistanceToOut continuation in the owned radial interval."},
      {"polycone_two_section_sharp_jump_shared_plane_near_grazing_up_entry_continuation",
       "polycone_two_section_sharp_jump", "grazing_entry", ShapeSampleCategory::kOutside, Vec_t(80., 0., -20.),
       Vec_t(-1., 0., 1.e-10), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance positive z component on the repeated-z transition plane; the upper-section "
       "entry must still expose a non-zero DistanceToOut continuation."},
      {"polycone_two_section_sharp_jump_shared_plane_near_grazing_down_entry_continuation",
       "polycone_two_section_sharp_jump", "grazing_entry", ShapeSampleCategory::kOutside, Vec_t(80., 0., -20.),
       Vec_t(-1., 0., -1.e-10), false, Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance negative z component on the repeated-z transition plane; the lower-section "
       "entry must still expose a non-zero DistanceToOut continuation."},
      {"polycone_two_section_sharp_jump_top_cap_near_grazing_outward_miss", "polycone_two_section_sharp_jump",
       "grazing_miss", ShapeSampleCategory::kOutside, Vec_t(80., 0., 120.), Vec_t(-1., 0., 1.e-10), false,
       Vec_t(0., 0., 0.), 0.,
       "Outside ray has a sub-tolerance outward z component on the terminal top cap plane; unlike an internal "
       "handoff plane, it must not report a DistanceToIn hit."},
  };
  return cases;
}

/**
 * @brief Find one curated manual edge case by its stable name.
 */
inline const ManualEdgeCase *FindManualEdgeCase(const std::string &name)
{
  for (auto const &manual_case : GetManualEdgeCases()) {
    if (manual_case.name == name) return &manual_case;
  }
  return nullptr;
}

/**
 * @brief Return every curated manual edge case for one sampled solid case.
 */
inline std::vector<const ManualEdgeCase *> FindManualEdgeCasesForSolid(const std::string &solid_case_name)
{
  std::vector<const ManualEdgeCase *> matches;
  for (auto const &manual_case : GetManualEdgeCases()) {
    if (manual_case.solid_case_name == solid_case_name) matches.push_back(&manual_case);
  }
  return matches;
}

/**
 * @brief Return the stable list of curated manual edge-case names.
 */
inline const std::vector<std::string> &GetManualEdgeCaseNames()
{
  static const std::vector<std::string> names = [] {
    std::vector<std::string> values;
    values.reserve(GetManualEdgeCases().size());
    for (auto const &manual_case : GetManualEdgeCases()) {
      values.emplace_back(manual_case.name);
    }
    return values;
  }();
  return names;
}

/**
 * @brief Return the sampled-solid names that currently have manual edge cases.
 */
inline const std::vector<std::string> &GetManualEdgeCaseSolidNames()
{
  static const std::vector<std::string> solid_names = [] {
    std::vector<std::string> values;
    values.reserve(GetManualEdgeCases().size());
    for (auto const &manual_case : GetManualEdgeCases()) {
      if (std::find(values.begin(), values.end(), manual_case.solid_case_name) == values.end()) {
        values.emplace_back(manual_case.solid_case_name);
      }
    }
    return values;
  }();
  return solid_names;
}

} // namespace test
} // namespace vecgeom

#endif
