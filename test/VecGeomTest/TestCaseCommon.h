/**
 * @file TestCaseCommon.h
 * @brief Shared declarations and sampling helpers for helper-level solid registries.
 *
 * See docs/shape_testing.md for the public meaning of tiers, sample-stream
 * reuse between fast/medium/slow, and how these sampling helpers are consumed
 * by ShapeContractTest.
 */

#ifndef VECGEOM_TEST_VECGEOMTEST_TESTCASECOMMON_HH
#define VECGEOM_TEST_VECGEOMTEST_TESTCASECOMMON_HH

#include <memory>
#include <string>
#include <vector>

#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeomTest/ShapeSampleSet.h"

namespace vecgeom {
namespace test {

/**
 * @brief One configured solid entry exposed through the helper-level test registry.
 *
 * Each entry binds a stable case name, a factory that constructs a standalone
 * placed solid, a debugger-oriented implementation type hint, and the default
 * fast/medium/slow sampling profiles used by ShapeContractTest, together with
 * the solid-specific contract tolerance inherited from the legacy ShapeTester
 * coverage for the same family.
 */
struct TestCaseSolid {
  const char *name = "";
  std::unique_ptr<vecgeom::VPlacedVolume> (*make_shape)();
  // Keep a debugger-oriented kernel type name per configured solid so failure
  // reports can print a practical gdb breakpoint target. The outer
  // implementation type can often omit the inline vecgeom::cxx namespace, but
  // template arguments such as ConeTypes::UniversalCone still need the fully
  // qualified spelling that gdb resolves reliably.
  const char *implementation_debug_type = "";
  ShapeSamplingConfig fast_contract_sampling;
  // By default this mirrors the fast profile except for max_points. Set
  // use_independent_medium_sampling when a solid truly needs a distinct
  // medium-only seed or outside-point tuning.
  ShapeSamplingConfig medium_contract_sampling;
  ShapeSamplingConfig slow_contract_sampling;
  // Keep fast/medium on the same deterministic sample stream by default. Set
  // this flag only when a solid truly needs a medium-only sampling profile.
  bool use_independent_medium_sampling = false;
  // Slow uses the same fast sample stream by default and only increases the
  // sample count. Set this flag when a solid needs a dedicated slow-only
  // profile with different outside tuning or a different seed/stream pair.
  bool use_independent_slow_sampling = false;
  // Contract tolerance associated with this solid family. Keep this at or
  // above vecgeom::kTolerance; curved second-order families typically use
  // vecgeom::kConeTolerance.
  Precision solid_tolerance = vecgeom::kTolerance;
};

/**
 * @brief Build a standalone placed solid and intentionally leak its support graph.
 *
 * Several helper solids depend on logical volumes or other support objects that
 * must outlive the returned placed volume. These tests are short-lived
 * executables, so keeping that graph alive for the process lifetime is the
 * simplest way to hand back a usable placed leaf.
 */
inline vecgeom::VPlacedVolume *MakeLeakedStandalonePlacedVolume(
    const char *name, vecgeom::VUnplacedVolume *unplaced, vecgeom::Transformation3D const *transformation = nullptr)
{
  // Some helper solids, especially Booleans and generic test-only unplaced
  // helpers, depend on supporting logical/unplaced objects that outlive the
  // returned placed volume. Keep that support graph alive for the process
  // lifetime by allocating it on the heap and returning only the placed leaf.
  auto *logical = new vecgeom::LogicalVolume(name, unplaced);
  if (transformation != nullptr) return logical->Place(new vecgeom::Transformation3D(*transformation));
  return logical->Place();
}

/**
 * @brief Wrap a leaked standalone placed solid in a unique_ptr for test code.
 */
inline std::unique_ptr<vecgeom::VPlacedVolume> MakeStandalonePlacedTestSolid(const char *name,
                                                                             vecgeom::VUnplacedVolume *unplaced)
{
  return std::unique_ptr<vecgeom::VPlacedVolume>(MakeLeakedStandalonePlacedVolume(name, unplaced));
}

/**
 * @brief Build the default fast helper-tier sampling profile for one solid.
 */
inline ShapeSamplingConfig MakeFastContractSampling(int max_points, unsigned long seed, unsigned long stream_id)
{
  ShapeSamplingConfig config;
  config.max_points      = max_points;
  config.inside_percent  = 25.;
  config.outside_percent = 25.;
  config.seed            = seed;
  config.stream_id       = stream_id;
  config.reseed          = true;
  return config;
}

/**
 * @brief Build the default slow helper-tier sampling profile for one solid.
 */
inline ShapeSamplingConfig MakeSlowContractSampling(int max_points, unsigned long seed, unsigned long stream_id,
                                                    Precision outside_max_radius_multiple    = 10.,
                                                    Precision outside_random_direction_ratio = 50.)
{
  ShapeSamplingConfig config;
  config.max_points                     = max_points;
  config.inside_percent                 = 25.;
  config.outside_percent                = 25.;
  config.outside_max_radius_multiple    = outside_max_radius_multiple;
  config.outside_random_direction_ratio = outside_random_direction_ratio;
  config.seed                           = seed;
  config.stream_id                      = stream_id;
  config.reseed                         = true;
  return config;
}

/**
 * @brief Build the fast helper-tier profile with explicit outside-point tuning.
 */
inline ShapeSamplingConfig MakeFastContractSampling(int max_points, unsigned long seed, unsigned long stream_id,
                                                    Precision outside_max_radius_multiple,
                                                    Precision outside_random_direction_ratio)
{
  auto config                           = MakeFastContractSampling(max_points, seed, stream_id);
  config.outside_max_radius_multiple    = outside_max_radius_multiple;
  config.outside_random_direction_ratio = outside_random_direction_ratio;
  return config;
}

/**
 * @brief Build the default medium helper-tier sampling profile for one solid.
 */
inline ShapeSamplingConfig MakeMediumContractSampling(int max_points, unsigned long seed, unsigned long stream_id,
                                                      Precision outside_max_radius_multiple    = 10.,
                                                      Precision outside_random_direction_ratio = 50.)
{
  ShapeSamplingConfig config;
  config.max_points                     = max_points;
  config.inside_percent                 = 25.;
  config.outside_percent                = 25.;
  config.outside_max_radius_multiple    = outside_max_radius_multiple;
  config.outside_random_direction_ratio = outside_random_direction_ratio;
  config.seed                           = seed;
  config.stream_id                      = stream_id;
  config.reseed                         = true;
  return config;
}

/**
 * @brief Return true when two sampling profiles differ only by statistics.
 *
 * Fast and medium are expected to share the same deterministic stream and
 * outside-point tuning unless a solid explicitly opts into an independent
 * medium profile. This helper keeps that policy testable in the runner.
 */
inline bool SamplingConfigsMatchExceptMaxPoints(const ShapeSamplingConfig &lhs, const ShapeSamplingConfig &rhs)
{
  return lhs.inside_percent == rhs.inside_percent && lhs.outside_percent == rhs.outside_percent &&
         lhs.edge_percent == rhs.edge_percent && lhs.outside_max_radius_multiple == rhs.outside_max_radius_multiple &&
         lhs.outside_random_direction_ratio == rhs.outside_random_direction_ratio && lhs.seed == rhs.seed &&
         lhs.stream_id == rhs.stream_id && lhs.reseed == rhs.reseed;
}

} // namespace test
} // namespace vecgeom

#endif
