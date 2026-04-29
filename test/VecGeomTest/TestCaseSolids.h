/**
 * @file TestCaseSolids.h
 * @brief Central registry of sampled solid cases used by ShapeContractTest.
 *
 * The helper executable builds its `-case_name` and `-family` selections from
 * this registry. Every configured case stays locally runnable through the
 * executable, while CTest can selectively disable individual helper families
 * in test/CMakeLists.txt when a known bug is still under investigation.
 *
 * See docs/shape_testing.md for the public contract-family descriptions,
 * tier behavior, and the end-to-end procedure for adding a new sampled solid.
 */
//
// How to add a new test case:
// 1. Add a Make...TestSolid() factory in the matching family header included
//    below, keeping the factory grouped with the same shape type.
// 2. Add a TestCaseSolid entry in GetTestCaseSolids(), again grouped with the
//    same shape family. Keep case names prefixed by the family name so the
//    helper executable can select by `-family` as well as `-case_name`.
// 3. Add the case name to SHAPE_CONTRACT_SOLID_CASES in test/CMakeLists.txt.
// 4. Reconfigure/build with your preferred CMake workflow. Example:
//      cmake -S . -B <build-dir> -DVECGEOM_SHAPE_CONTRACT_CTEST_TIER=medium
//      cmake --build <build-dir>
//    Then run:
//      <build-dir>/test/ShapeContractTest -tier fast -test_family contracts -case_name <new_case>
//      <build-dir>/test/ShapeContractTest -tier medium -test_family contracts -case_name <new_case>
//      <build-dir>/test/ShapeContractTest -tier slow -test_family contracts -case_name <new_case>
//      ctest --output-on-failure -R 'ShapeContractTest:'
// 5. If the case fails and the bug is not fixed in the same merge request,
//    keep it enabled here so it stays locally runnable, and add the failing
//    family or tier/family entry to the SHAPE_CONTRACT_DISABLED_CTEST_* lists
//    in test/CMakeLists.txt.

#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASESOLIDS_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASESOLIDS_HH

#include <algorithm>
#include <string>
#include <vector>

#include "VecGeomTest/TestCaseBoolean.h"
#include "VecGeomTest/TestCaseBox.h"
#include "VecGeomTest/TestCaseCommon.h"
#include "VecGeomTest/TestCaseCone.h"
#include "VecGeomTest/TestCaseCutTube.h"
#include "VecGeomTest/TestCaseEllipsoid.h"
#include "VecGeomTest/TestCaseEllipticalCone.h"
#include "VecGeomTest/TestCaseEllipticalTube.h"
#include "VecGeomTest/TestCaseExtruded.h"
#include "VecGeomTest/TestCaseGenTrap.h"
#include "VecGeomTest/TestCaseHype.h"
#include "VecGeomTest/TestCaseMultiUnion.h"
#include "VecGeomTest/TestCaseOrb.h"
#include "VecGeomTest/TestCaseParaboloid.h"
#include "VecGeomTest/TestCaseParallelepiped.h"
#include "VecGeomTest/TestCasePolycone.h"
#include "VecGeomTest/TestCasePolyhedron.h"
#include "VecGeomTest/TestCaseSExtru.h"
#include "VecGeomTest/TestCaseSphere.h"
#include "VecGeomTest/TestCaseTet.h"
#include "VecGeomTest/TestCaseTessellated.h"
#include "VecGeomTest/TestCaseTorus2.h"
#include "VecGeomTest/TestCaseTrapezoid.h"
#include "VecGeomTest/TestCaseTrd.h"
#include "VecGeomTest/TestCaseTube.h"

namespace vecgeom {
namespace test {

/**
 * @brief Return the full configured sampled-solid registry.
 */
inline const std::vector<TestCaseSolid> &GetTestCaseSolids()
{
  constexpr Precision kPlanarTolerance      = vecgeom::kTolerance;
  constexpr Precision kSecondOrderTolerance = vecgeom::kConeTolerance;
  constexpr Precision kEllipticTolerance    = 1.e-6;

  static const std::vector<TestCaseSolid> solids = {
      // Box
      {"box", &MakeBoxTestSolid, "BoxImplementation", MakeFastContractSampling(10000, 19, 0),
       MakeMediumContractSampling(100000, 19, 0), {}, false, false, kPlanarTolerance},

      // Tube
      {"tube_fullphi", &MakeTubeFullPhiTestSolid, "TubeImplementation<vecgeom::cxx::TubeTypes::UniversalTube>",
       MakeFastContractSampling(10000, 23, 1, 4., 25.), MakeMediumContractSampling(100000, 23, 1, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"tube_section", &MakeTubeSectionTestSolid, "TubeImplementation<vecgeom::cxx::TubeTypes::UniversalTube>",
       MakeFastContractSampling(10000, 24, 2, 4., 25.), MakeMediumContractSampling(100000, 24, 2, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"tube_thin_wall_long", &MakeTubeThinWallLongTestSolid,
       "TubeImplementation<vecgeom::cxx::TubeTypes::UniversalTube>", MakeFastContractSampling(10000, 39, 17, 4., 25.),
       MakeMediumContractSampling(100000, 39, 17, 4., 25.), {}, false, false, kSecondOrderTolerance},
      {"tube_narrow_phi", &MakeTubeNarrowPhiTestSolid, "TubeImplementation<vecgeom::cxx::TubeTypes::UniversalTube>",
       MakeFastContractSampling(10000, 40, 18, 4., 25.), MakeMediumContractSampling(100000, 40, 18, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"tube_almost_full_phi", &MakeTubeAlmostFullPhiTestSolid,
       "TubeImplementation<vecgeom::cxx::TubeTypes::UniversalTube>", MakeFastContractSampling(10000, 41, 19, 4., 25.),
       MakeMediumContractSampling(100000, 41, 19, 4., 25.), {}, false, false, kSecondOrderTolerance},
      {"tube_short_disk", &MakeTubeShortDiskTestSolid, "TubeImplementation<vecgeom::cxx::TubeTypes::UniversalTube>",
       MakeFastContractSampling(10000, 56, 34, 4., 25.), MakeMediumContractSampling(100000, 56, 34, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // Cone
      {"cone_fullphi", &MakeConeFullPhiTestSolid, "ConeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 25, 3, 4., 25.), MakeMediumContractSampling(100000, 25, 3, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"cone_section", &MakeConeSectionTestSolid, "ConeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 26, 4, 4., 25.), MakeMediumContractSampling(100000, 26, 4, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"cone_thin_shell", &MakeConeThinShellTestSolid, "ConeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 42, 20, 4., 25.), MakeMediumContractSampling(100000, 42, 20, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"cone_almost_cylinder", &MakeConeAlmostCylinderTestSolid,
       "ConeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>", MakeFastContractSampling(10000, 43, 21, 4., 25.),
       MakeMediumContractSampling(100000, 43, 21, 4., 25.), {}, false, false, kSecondOrderTolerance},
      {"cone_narrow_phi", &MakeConeNarrowPhiTestSolid, "ConeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 57, 35, 4., 25.), MakeMediumContractSampling(100000, 57, 35, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"cone_almost_full_phi", &MakeConeAlmostFullPhiTestSolid,
       "ConeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>", MakeFastContractSampling(10000, 58, 36, 4., 25.),
       MakeMediumContractSampling(100000, 58, 36, 4., 25.), {}, false, false, kSecondOrderTolerance},

      // Ellipsoid
      {"ellipsoid", &MakeEllipsoidTestSolid, "EllipsoidImplementation",
       MakeFastContractSampling(10000, 59, 37, 4., 25.), MakeMediumContractSampling(100000, 59, 37, 4., 25.), {},
       false, false, kEllipticTolerance},

      // EllipticalCone
      {"elliptical_cone", &MakeEllipticalConeTestSolid, "EllipticalConeImplementation",
       MakeFastContractSampling(10000, 60, 38, 4., 25.), MakeMediumContractSampling(100000, 60, 38, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // EllipticalTube
      {"elliptical_tube", &MakeEllipticalTubeTestSolid, "EllipticalTubeImplementation",
       MakeFastContractSampling(10000, 30, 8, 4., 25.), MakeMediumContractSampling(100000, 30, 8, 4., 25.), {},
       false, false, kEllipticTolerance},
      {"elliptical_tube_thin", &MakeEllipticalTubeThinTestSolid, "EllipticalTubeImplementation",
       MakeFastContractSampling(10000, 52, 30, 4., 25.), MakeMediumContractSampling(100000, 52, 30, 4., 25.), {},
       false, false, kEllipticTolerance},
      {"elliptical_tube_long", &MakeEllipticalTubeLongTestSolid, "EllipticalTubeImplementation",
       MakeFastContractSampling(10000, 53, 31, 4., 25.), MakeMediumContractSampling(100000, 53, 31, 4., 25.), {},
       false, false, kEllipticTolerance},

      // Parallelepiped
      {"parallelepiped_general", &MakeParallelepipedGeneralTestSolid, "ParallelepipedImplementation",
       MakeFastContractSampling(10000, 32, 10, 4., 25.), MakeMediumContractSampling(100000, 32, 10, 4., 25.), {},
       false, false, kPlanarTolerance},
      {"parallelepiped_high_shear", &MakeParallelepipedHighShearTestSolid, "ParallelepipedImplementation",
       MakeFastContractSampling(10000, 61, 39, 4., 25.), MakeMediumContractSampling(100000, 61, 39, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Tet
      {"tet", &MakeTetTestSolid, "TetImplementation", MakeFastContractSampling(10000, 33, 11, 4., 25.),
       MakeMediumContractSampling(100000, 33, 11, 4., 25.), {}, false, false, kPlanarTolerance},
      {"tet_sliver_like", &MakeTetSliverLikeTestSolid, "TetImplementation",
       MakeFastContractSampling(10000, 54, 32, 4., 25.), MakeMediumContractSampling(100000, 54, 32, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Trd
      {"trd_boxlike", &MakeTrdBoxlikeTestSolid, "TrdImplementation<vecgeom::cxx::TrdTypes::UniversalTrd>",
       MakeFastContractSampling(10000, 35, 13, 4., 25.), MakeMediumContractSampling(100000, 35, 13, 4., 25.), {},
       false, false, kPlanarTolerance},
      {"trd_extreme_aspect", &MakeTrdExtremeAspectTestSolid, "TrdImplementation<vecgeom::cxx::TrdTypes::UniversalTrd>",
       MakeFastContractSampling(10000, 44, 22, 4., 25.), MakeMediumContractSampling(100000, 44, 22, 4., 25.), {},
       false, false, kPlanarTolerance},
      {"trd_increasing_xy", &MakeTrdIncreasingXYTestSolid, "TrdImplementation<vecgeom::cxx::TrdTypes::UniversalTrd>",
       MakeFastContractSampling(10000, 29, 7, 4., 25.), MakeMediumContractSampling(100000, 29, 7, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Trapezoid
      {"trapezoid_corners", &MakeTrapezoidFromCornersTestSolid, "TrapezoidImplementation",
       MakeFastContractSampling(10000, 36, 14, 4., 25.), MakeMediumContractSampling(100000, 36, 14, 4., 25.), {},
       false, false, kPlanarTolerance},
      {"trapezoid_extreme_skew", &MakeTrapezoidExtremeSkewTestSolid, "TrapezoidImplementation",
       MakeFastContractSampling(10000, 45, 23, 4., 25.), MakeMediumContractSampling(100000, 45, 23, 4., 25.), {},
       false, false, kPlanarTolerance},
      {"trapezoid_near_para", &MakeTrapezoidNearParallelepipedTestSolid, "TrapezoidImplementation",
       MakeFastContractSampling(10000, 55, 33, 4., 25.), MakeMediumContractSampling(100000, 55, 33, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Polycone / GenericPolycone
      {"polycone_cms_like", &MakePolyconeCmsLikeTestSolid,
       "PolyconeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 37, 15, 4., 25.), MakeMediumContractSampling(100000, 37, 15, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"generic_polycone_irregular", &MakeGenericPolyconeIrregularTestSolid, "GenericPolyconeImplementation",
       MakeFastContractSampling(10000, 38, 16, 4., 25.), MakeMediumContractSampling(100000, 38, 16, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"polycone_two_section_sharp_jump", &MakePolyconeTwoSectionSharpJumpTestSolid,
       "PolyconeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 48, 26, 4., 25.), MakeMediumContractSampling(100000, 48, 26, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"polycone_many_section_alternating", &MakePolyconeManySectionAlternatingTestSolid,
       "PolyconeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 49, 27, 4., 25.), MakeMediumContractSampling(100000, 49, 27, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"polycone_nearly_repeated_z", &MakePolyconeNearlyRepeatedZTestSolid,
       "PolyconeImplementation<vecgeom::cxx::ConeTypes::UniversalCone>",
       MakeFastContractSampling(10000, 50, 28, 4., 25.), MakeMediumContractSampling(100000, 50, 28, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"generic_polycone_zigzag_profile", &MakeGenericPolyconeZigZagProfileTestSolid, "GenericPolyconeImplementation",
       MakeFastContractSampling(10000, 51, 29, 4., 25.), MakeMediumContractSampling(100000, 51, 29, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // Paraboloid
      {"paraboloid", &MakeParaboloidTestSolid, "ParaboloidImplementation",
       MakeFastContractSampling(10000, 31, 9, 4., 25.), MakeMediumContractSampling(100000, 31, 9, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // CutTube
      {"cuttube_section_inner", &MakeCutTubeSectionInnerTestSolid, "CutTubeImplementation",
       MakeFastContractSampling(10000, 34, 12, 4., 25.), MakeMediumContractSampling(100000, 34, 12, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // Hype
      {"hype", &MakeHypeTestSolid, "HypeImplementation<vecgeom::cxx::HypeTypes::UniversalHype>",
       MakeFastContractSampling(10000, 62, 40, 4., 25.), MakeMediumContractSampling(100000, 62, 40, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // Orb
      {"orb", &MakeOrbTestSolid, "OrbImplementation", MakeFastContractSampling(10000, 27, 5, 4., 25.),
       MakeMediumContractSampling(100000, 27, 5, 4., 25.), {}, false, false, kSecondOrderTolerance},

      // Sphere
      {"sphere_thin_shell", &MakeSphereThinShellTestSolid, "SphereImplementation",
       MakeFastContractSampling(10000, 46, 24, 4., 25.), MakeMediumContractSampling(100000, 46, 24, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"sphere_narrow_phi", &MakeSphereNarrowPhiTestSolid, "SphereImplementation",
       MakeFastContractSampling(10000, 47, 25, 4., 25.), MakeMediumContractSampling(100000, 47, 25, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"sphere_narrow_theta", &MakeSphereNarrowThetaTestSolid, "SphereImplementation",
       MakeFastContractSampling(10000, 63, 41, 4., 25.), MakeMediumContractSampling(100000, 63, 41, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"sphere_almost_full_phi", &MakeSphereAlmostFullPhiTestSolid, "SphereImplementation",
       MakeFastContractSampling(10000, 64, 42, 4., 25.), MakeMediumContractSampling(100000, 64, 42, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"sphere_section", &MakeSphereSectionTestSolid, "SphereImplementation",
       MakeFastContractSampling(10000, 28, 6, 4., 25.), MakeMediumContractSampling(100000, 28, 6, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // Torus2
      {"torus2_general", &MakeTorus2GeneralTestSolid, "TorusImplementation2",
       MakeFastContractSampling(10000, 65, 43, 4., 25.), MakeMediumContractSampling(100000, 65, 43, 4., 25.), {},
       false, false, kSecondOrderTolerance},

      // Polyhedron
      {"polyhedron_phi_section", &MakePolyhedronPhiSectionTestSolid, "PolyhedronImplementation",
       MakeFastContractSampling(10000, 66, 44, 4., 25.), MakeMediumContractSampling(100000, 66, 44, 4., 25.), {},
       false, false, kPlanarTolerance},

      // GenTrap
      {"gentrap_twisted", &MakeGenTrapTwistedTestSolid, "GenTrapImplementation",
       MakeFastContractSampling(10000, 67, 45, 4., 25.), MakeMediumContractSampling(100000, 67, 45, 4., 25.), {},
       false, false, kPlanarTolerance},

      // SExtru
      {"sextru_circular", &MakeSExtruCircularTestSolid, "SExtruImplementation",
       MakeFastContractSampling(10000, 68, 46, 4., 25.), MakeMediumContractSampling(100000, 68, 46, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Extruded
      {"extruded_multilayer", &MakeExtrudedMultiLayerTestSolid, "ExtrudedImplementation",
       MakeFastContractSampling(10000, 69, 47, 4., 25.), MakeMediumContractSampling(100000, 69, 47, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Tessellated
      {"tessellated_orb", &MakeTessellatedOrbTestSolid, "TessellatedImplementation",
       MakeFastContractSampling(10000, 70, 48, 4., 25.), MakeMediumContractSampling(100000, 70, 48, 4., 25.), {},
       false, false, kPlanarTolerance},

      // MultiUnion
      {"multiunion_boxes", &MakeMultiUnionBoxesTestSolid, "MultiUnionImplementation",
       MakeFastContractSampling(10000, 71, 49, 4., 25.), MakeMediumContractSampling(100000, 71, 49, 4., 25.), {},
       false, false, kPlanarTolerance},

      // Boolean
      {"boolean_union_offset_boxes", &MakeBooleanUnionOffsetBoxesTestSolid, "BooleanUnionImplementation",
       MakeFastContractSampling(10000, 72, 50, 4., 25.), MakeMediumContractSampling(100000, 72, 50, 4., 25.), {},
       false, false, kPlanarTolerance},
      {"boolean_intersection_box_tube", &MakeBooleanIntersectionBoxTubeTestSolid, "BooleanIntersectionImplementation",
       MakeFastContractSampling(10000, 73, 51, 4., 25.), MakeMediumContractSampling(100000, 73, 51, 4., 25.), {},
       false, false, kSecondOrderTolerance},
      {"boolean_subtraction_box_tube", &MakeBooleanSubtractionBoxTubeTestSolid,
       "BooleanImplementation<vecgeom::kSubtraction>", MakeFastContractSampling(10000, 74, 52, 4., 25.),
       MakeMediumContractSampling(100000, 74, 52, 4., 25.), {}, false, false, kSecondOrderTolerance},
  };
  return solids;
}

/**
 * @brief Find one configured solid case by its stable `-case_name`.
 */
inline const TestCaseSolid *FindTestCaseSolid(const std::string &name)
{
  for (auto const &solid : GetTestCaseSolids()) {
    if (solid.name == name) return &solid;
  }
  return nullptr;
}

/**
 * @brief Map a configured solid case name to its shape family label.
 *
 * Family names are used by the CLI `-family` selector and by the CTest
 * registration helpers.
 */
inline std::string GetTestCaseFamilyName(const std::string &case_name)
{
  auto starts_with = [&case_name](const char *prefix) { return case_name.rfind(prefix, 0) == 0; };

  if (case_name == "box") return "box";
  if (starts_with("tube_")) return "tube";
  if (starts_with("cone_")) return "cone";
  if (case_name == "ellipsoid") return "ellipsoid";
  if (starts_with("elliptical_cone")) return "elliptical_cone";
  if (starts_with("elliptical_tube")) return "elliptical_tube";
  if (starts_with("parallelepiped")) return "parallelepiped";
  if (starts_with("tet")) return "tet";
  if (starts_with("trd")) return "trd";
  if (starts_with("trapezoid")) return "trapezoid";
  if (starts_with("polycone")) return "polycone";
  if (starts_with("generic_polycone")) return "generic_polycone";
  if (case_name == "paraboloid") return "paraboloid";
  if (starts_with("cuttube")) return "cuttube";
  if (case_name == "hype") return "hype";
  if (case_name == "orb") return "orb";
  if (starts_with("sphere")) return "sphere";
  if (starts_with("torus2")) return "torus2";
  if (starts_with("polyhedron")) return "polyhedron";
  if (starts_with("gentrap")) return "gentrap";
  if (starts_with("sextru")) return "sextru";
  if (starts_with("extruded")) return "extruded";
  if (starts_with("tessellated")) return "tessellated";
  if (starts_with("multiunion")) return "multiunion";
  if (starts_with("boolean")) return "boolean";
  return "";
}

/**
 * @brief Return every configured sampled solid in one shape family.
 */
inline std::vector<const TestCaseSolid *> FindTestCaseSolidsByFamily(const std::string &family)
{
  std::vector<const TestCaseSolid *> matches;
  for (auto const &solid : GetTestCaseSolids()) {
    if (GetTestCaseFamilyName(solid.name) == family) matches.push_back(&solid);
  }
  return matches;
}

/**
 * @brief Return the stable list of configured sampled solid case names.
 */
inline const std::vector<std::string> &GetTestCaseSolidNames()
{
  static const std::vector<std::string> names = [] {
    std::vector<std::string> values;
    values.reserve(GetTestCaseSolids().size());
    for (auto const &solid : GetTestCaseSolids()) {
      values.emplace_back(solid.name);
    }
    return values;
  }();
  return names;
}

/**
 * @brief Return the stable list of configured shape-family labels.
 */
inline const std::vector<std::string> &GetTestCaseFamilyNames()
{
  static const std::vector<std::string> families = [] {
    std::vector<std::string> values;
    values.reserve(GetTestCaseSolids().size());
    for (auto const &solid : GetTestCaseSolids()) {
      auto family = GetTestCaseFamilyName(solid.name);
      if (family.empty()) continue;
      if (std::find(values.begin(), values.end(), family) == values.end()) values.push_back(family);
    }
    return values;
  }();
  return families;
}

} // namespace test
} // namespace vecgeom

#endif
