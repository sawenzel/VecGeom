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
      {"cuttube_section_inner_cut_plane_grazing", "cuttube_section_inner", "surface", ShapeSampleCategory::kSurface,
       Vec_t(4.1438716887445688, 0., -0.20947942872035785),
       Vec_t(0.46997756897067333, 0.5375170920854937, 0.70014031478009531), false, Vec_t(0., 0., 0.), 0.,
       "Smooth cut-plane surface replay whose grazing direction previously drove the embedded infinite-z tube path "
       "through invalid z-plane arithmetic."},
      {"cone_thin_shell_surface_entering_outer_lower_ring", "cone_thin_shell", "contracts",
       ShapeSampleCategory::kSurface, Vec_t(-43.093354838211269, -25.356710388709519, -200.0),
       Vec_t(0.90448163727274844, 0.15468343036337337, -0.39747453277842609), false, Vec_t(0., 0., 0.), 0.,
       "Surface-entering replay ray on the lower outer ring of the thin-shell cone from the 10M-point stress run."},
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
