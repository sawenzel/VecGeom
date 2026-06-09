// Reusable convention/contract checks extracted from ShapeTester.
//
// The goal of this helper layer is to keep the legacy ShapeTester semantics
// while making the convention logic reusable from focused unit tests. The
// current shape_test*/convention_test* executables still drive the same checks
// through ShapeTester, but the core predicates now live here.
//
// See docs/shape_testing.md for the user-facing contract tables, helper-family
// taxonomy, and replay/debug workflow that map onto the bits and helpers below.

#ifndef VECGEOM_TEST_VECGEOMTEST_SHAPECONTRACTCHECKS_HH
#define VECGEOM_TEST_VECGEOMTEST_SHAPECONTRACTCHECKS_HH

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "VecGeom/volumes/UnplacedVolume.h"
#include "VecGeomTest/ApproxEqual.h"
#include "VecGeomTest/ShapeCheckResult.h"
#include "VecGeomTest/ShapeSampleSet.h"

namespace vecgeom {
namespace test {

enum ShapeConventionBit {
  kSurfaceDistanceToInEntering  = 0,
  kSurfaceDistanceToInExiting   = 1,
  kSurfaceDistanceToOutExiting  = 2,
  kSurfaceDistanceToOutEntering = 3,
  kSurfaceSafetyToIn            = 4,
  kSurfaceSafetyToOut           = 5,
  kInsideDistanceToIn           = 6,
  kInsideDistanceToOut          = 7,
  kInsideSafetyToIn             = 8,
  kInsideSafetyToOut            = 9,
  kOutsideDistanceToIn          = 10,
  kOutsideDistanceToOut         = 11,
  kOutsideSafetyToIn            = 12,
  kOutsideSafetyToOut           = 13,
  kNormalSurfaceValid           = 14,
  kNormalSurfaceUnitLength      = 15,
  kNormalSurfaceOutward         = 16,
  kNormalInsideExitOnSurface    = 17,
  kNormalInsideExitValid        = 18,
  kNormalInsideExitUnitLength   = 19,
  kNormalInsideExitOutward      = 20,
  kNormalOutsideEntryOnSurface  = 21,
  kNormalOutsideEntryValid      = 22,
  kNormalOutsideEntryUnitLength = 23,
  kNormalOutsideEntryInward     = 24,
  kSurfaceRayNotBothZero        = 25,
  kSurfaceGrazingNotBothZero    = 26,
  kSurfaceDistanceToOutFinite   = 27,
  kSurfaceShallowInward         = 28,
  kSurfaceShallowOutward        = 29,
  kDistanceToOutPositive        = 30,
  kDistanceToOutWithinExtent    = 31,
  kDistanceToOutAboveSafety     = 32,
  kDistanceToOutOnSurface       = 33,
  kDistanceToInApproachFinite   = 34,
  kDistanceToInPositive         = 35,
  kDistanceToInFinite           = 36,
  kDistanceToInAboveSafety      = 37,
  kDistanceToInWithinTarget     = 38,
  kDistanceToInOnSurface        = 39,
  kSafetyToOutPositive          = 40,
  kSafetyToOutDistanceBound     = 41,
  kSafetyToOutSafeSphere        = 42,
  kSafetyToInPositive           = 43,
  kSafetyToInDistanceBound      = 44,
  kSafetyToInSafeSphere         = 45,
  kHitInsideExitFinite          = 46,
  kHitInsideExitOnSurface       = 47,
  kHitInsideExitSafetyToIn      = 48,
  kHitInsideExitSafetyToOut     = 49,
  kHitInsideExitNormalAgreement = 50,
  kHitOutsideEntryFinite        = 51,
  kHitOutsideEntryOnSurface     = 52,
  kHitOutsideEntrySafetyToIn    = 53,
  kHitOutsideEntrySafetyToOut   = 54,
  kHitOutsideEntryDistanceToIn  = 55,
  kHitOutsideEntryDistanceToOut = 56,
  kHitOutsideEntryDistanceSign  = 57,
  kHitOutsideExitOnSurface      = 58,
  kHitOutsideExitSafetyToIn     = 59,
  kHitOutsideExitSafetyToOut    = 60,
  kShapeConventionBitCount      = 61
};

// Keep the legacy message ordering stable because the convention bitset and the
// end-of-run report both rely on this positional mapping.
inline const std::vector<std::string> &ShapeConventionMessages()
{
  static const std::vector<std::string> messages = {
      "DistanceToIn()  : For Point On Surface and Entering the Shape",
      "DistanceToIn()  : For Point On Surface and Exiting the Shape",
      "DistanceToOut() : For Point On Surface and Exiting the Shape",
      "DistanceToOut() : For Point On Surface and Entering the Shape",
      "SafetyToIn()    : For Point On Surface ",
      "SafetyToOut()   : For Point On Surface ",
      "DistanceToIn()  : For Inside Point",
      "DistanceToOut() : For Inside Point",
      "SafetyToIn()    : For Inside Point",
      "SafetyToOut()   : For Inside Point",
      "DistanceToIn()  : For Outside Point",
      "DistanceToOut() : For Outside Point",
      "SafetyToIn()    : For Outside Point",
      "SafetyToOut()   : For Outside Point",
      "Normal()        : For Point On Surface",
      "Normal()        : Unit vector for Point On Surface",
      "Normal()        : Topologically outward for Point On Surface",
      "Normal()        : Exit point from Inside ray on Surface",
      "Normal()        : Valid at exit point from Inside ray",
      "Normal()        : Unit vector at exit point from Inside ray",
      "Normal()        : Outward at exit point from Inside ray",
      "Normal()        : Entry point from Outside ray on Surface",
      "Normal()        : Valid at entry point from Outside ray",
      "Normal()        : Unit vector at entry point from Outside ray",
      "Normal()        : Inward at entry point from Outside ray",
      "Surface()       : DistanceToIn/Out not both zero for Surface ray",
      "Surface()       : DistanceToIn/Out not both zero for grazing ray",
      "Surface()       : DistanceToOut finite for Surface ray",
      "Surface()       : Shallow inward ray has zero DistanceToIn",
      "Surface()       : Shallow outward ray has zero DistanceToOut",
      "DistanceToOut() : Positive for Inside Point",
      "DistanceToOut() : Finite and within extent for Inside Point",
      "DistanceToOut() : Above SafetyToOut for Inside Point",
      "DistanceToOut() : Exit point on Surface for Inside Point",
      "DistanceToIn()  : Bounding-box approach finite for Outside -> "
      "Inside ray",
      "DistanceToIn()  : Positive for Outside -> Inside ray",
      "DistanceToIn()  : Finite for Outside -> Inside ray",
      "DistanceToIn()  : Above SafetyToIn for Outside -> Inside ray",
      "DistanceToIn()  : Entry point before paired Inside target",
      "DistanceToIn()  : Entry point on Surface for Outside -> Inside ray",
      "SafetyToOut()   : Positive for Inside Point",
      "SafetyToOut()   : Not above DistanceToOut for Inside Point",
      "SafetyToOut()   : Safety sphere from Inside stays Inside or Surface",
      "SafetyToIn()    : Positive for Outside Point",
      "SafetyToIn()    : Not above DistanceToIn for Outside Point",
      "SafetyToIn()    : Safety sphere from Outside stays Outside or Surface",
      "Hit()           : DistanceToOut finite for Inside consistency ray",
      "Hit()           : Exit point on Surface for Inside consistency ray",
      "Hit()           : SafetyToIn within tolerance at Inside exit point",
      "Hit()           : SafetyToOut within tolerance at Inside exit point",
      "Hit()           : Normal and DistanceToIn agree at Inside exit point",
      "Hit()           : DistanceToIn finite for Outside consistency ray",
      "Hit()           : Entry point on Surface for Outside consistency ray",
      "Hit()           : SafetyToIn within tolerance at Outside entry point",
      "Hit()           : SafetyToOut within tolerance at Outside entry point",
      "Hit()           : DistanceToIn finite at Outside entry point",
      "Hit()           : DistanceToOut finite at Outside entry point",
      "Hit()           : DistanceToOut non-negative at Outside entry point",
      "Hit()           : Exit point on Surface after Outside entry point",
      "Hit()           : SafetyToIn within tolerance at Outside exit point",
      "Hit()           : SafetyToOut within tolerance at Outside exit point"};
  return messages;
}

struct ShapeContractSampleView {
  const std::vector<Vec_t> *points     = nullptr;
  const std::vector<Vec_t> *directions = nullptr;
  int offset_inside                    = 0;
  int offset_surface                   = 0;
  int offset_edge                      = 0;
  int offset_outside                   = 0;
  int max_points_inside                = 0;
  int max_points_surface               = 0;
  int max_points_edge                  = 0;
  int max_points_outside               = 0;

  const Vec_t &Point(int index) const { return (*points)[index]; }
  const Vec_t &Direction(int index) const { return (*directions)[index]; }
  int TotalPoints() const { return points ? static_cast<int>(points->size()) : 0; }
};

// View adapter over ShapeSampleSet so helper code can work either with the
// extracted sampler or with ShapeTester-owned point/direction arrays.
inline ShapeContractSampleView MakeShapeContractSampleView(const ShapeSampleSet &samples)
{
  ShapeContractSampleView view;
  view.points             = &samples.points;
  view.directions         = &samples.directions;
  view.offset_inside      = samples.offset_inside;
  view.offset_surface     = samples.offset_surface;
  view.offset_edge        = samples.offset_edge;
  view.offset_outside     = samples.offset_outside;
  view.max_points_inside  = samples.max_points_inside;
  view.max_points_surface = samples.max_points_surface;
  view.max_points_edge    = samples.max_points_edge;
  view.max_points_outside = samples.max_points_outside;
  return view;
}

using ShapeContractReporter = std::function<void(const ShapeRecordDecision &, const ShapeCheckContext &,
                                                 const std::string &, const Vec_t &, const Vec_t &, Precision)>;

inline ShapeSampleCategory ShapeSampleCategoryForIndex(const ShapeContractSampleView &samples, int sample_index)
{
  if (sample_index < 0 || sample_index >= samples.TotalPoints()) return ShapeSampleCategory::kUnknown;
  if (sample_index >= samples.offset_inside && sample_index < samples.offset_surface)
    return ShapeSampleCategory::kInside;
  if (sample_index >= samples.offset_surface && sample_index < samples.offset_edge)
    return ShapeSampleCategory::kSurface;
  if (sample_index >= samples.offset_edge && sample_index < samples.offset_outside) return ShapeSampleCategory::kEdge;
  if (sample_index >= samples.offset_outside && sample_index < samples.TotalPoints())
    return ShapeSampleCategory::kOutside;
  return ShapeSampleCategory::kUnknown;
}

inline ShapeCheckContext MakeShapeCheckContext(const ShapeContractSampleView &samples, int sample_index,
                                               int convention_bit = -1)
{
  ShapeCheckContext context;
  context.sample_index   = sample_index;
  context.sample_group   = ShapeSampleCategoryForIndex(samples, sample_index);
  context.convention_bit = convention_bit;
  return context;
}

inline int PairedInsideSampleIndex(const ShapeContractSampleView &samples, int outside_sample_index)
{
  if (samples.max_points_inside <= 0 || outside_sample_index < samples.offset_outside ||
      outside_sample_index >= samples.TotalPoints()) {
    return -1;
  }
  const int outside_ordinal = outside_sample_index - samples.offset_outside;
  return samples.offset_inside + (outside_ordinal % samples.max_points_inside);
}

inline std::string ShapeConventionLabel(int convention_bit)
{
  if (convention_bit >= 0 && convention_bit < static_cast<int>(ShapeConventionMessages().size())) {
    return ShapeConventionMessages()[convention_bit];
  }
  return "Non-bit contract diagnostic";
}

inline const char *InsideLabel(vecgeom::EnumInside inside)
{
  switch (inside) {
  case vecgeom::EnumInside::kInside:
    return "inside";
  case vecgeom::EnumInside::kSurface:
    return "surface";
  case vecgeom::EnumInside::kOutside:
    return "outside";
  }
  return "unknown";
}

inline std::string FormatVec(const Vec_t &vec)
{
  std::ostringstream out;
  out << std::setprecision(17);
  out << "(" << vec.x() << ", " << vec.y() << ", " << vec.z() << ")";
  return out.str();
}

inline bool IsNormalConventionBit(int convention_bit)
{
  return convention_bit >= kNormalSurfaceValid && convention_bit < kSurfaceRayNotBothZero;
}

inline bool IsSurfaceConventionBit(int convention_bit)
{
  return convention_bit >= kSurfaceRayNotBothZero && convention_bit <= kSurfaceShallowOutward;
}

inline bool IsDistanceToOutCheckBit(int convention_bit)
{
  return convention_bit >= kDistanceToOutPositive && convention_bit <= kDistanceToOutOnSurface;
}

inline bool IsDistanceToInCheckBit(int convention_bit)
{
  return convention_bit >= kDistanceToInApproachFinite && convention_bit <= kDistanceToInOnSurface;
}

inline bool IsSafetyCheckBit(int convention_bit)
{
  return convention_bit >= kSafetyToOutPositive && convention_bit <= kSafetyToInSafeSphere;
}

inline bool IsHitConsistencyCheckBit(int convention_bit)
{
  return convention_bit >= kHitInsideExitFinite && convention_bit < kShapeConventionBitCount;
}

inline const char *ShapeContractReplayFunctionName(const ShapeCheckContext &context)
{
  if (IsHitConsistencyCheckBit(context.convention_bit)) return "vecgeom::test::ReplayShapeHitConsistencySample";
  if (IsSafetyCheckBit(context.convention_bit)) return "vecgeom::test::ReplayShapeSafetySample";
  if (IsDistanceToInCheckBit(context.convention_bit)) return "vecgeom::test::ReplayShapeDistanceToInSample";
  if (IsDistanceToOutCheckBit(context.convention_bit)) return "vecgeom::test::ReplayShapeDistanceToOutSample";
  if (IsSurfaceConventionBit(context.convention_bit)) return "vecgeom::test::ReplayShapeSurfaceSample";
  if (IsNormalConventionBit(context.convention_bit)) return "vecgeom::test::ReplayShapeNormalSample";
  return "vecgeom::test::ReplayShapeConventionSample";
}

inline const char *ShapeContractEvaluatorFunctionName(const ShapeCheckContext &context)
{
  if (IsHitConsistencyCheckBit(context.convention_bit)) {
    return context.sample_group == ShapeSampleCategory::kInside ? "vecgeom::test::EvaluateInsideHitConsistencySample"
                                                                : "vecgeom::test::EvaluateOutsideHitConsistencySample";
  }
  if (IsSafetyCheckBit(context.convention_bit)) return "vecgeom::test::EvaluateShapeSafetySample";
  if (IsDistanceToInCheckBit(context.convention_bit)) return "vecgeom::test::EvaluateOutsideDistanceToInSample";
  if (IsDistanceToOutCheckBit(context.convention_bit)) return "vecgeom::test::EvaluateInsideDistanceToOutSample";
  if (IsSurfaceConventionBit(context.convention_bit)) return "vecgeom::test::EvaluateSurfacePointSample";
  if (IsNormalConventionBit(context.convention_bit)) {
    switch (context.sample_group) {
    case ShapeSampleCategory::kInside:
      return "vecgeom::test::EvaluateInsideExitNormalSample";
    case ShapeSampleCategory::kSurface:
    case ShapeSampleCategory::kEdge:
      return "vecgeom::test::EvaluateSurfaceNormalSample";
    case ShapeSampleCategory::kOutside:
      return "vecgeom::test::EvaluateOutsideEntryNormalSample";
    default:
      return "vecgeom::test::ReplayShapeNormalSample";
    }
  }

  switch (context.sample_group) {
  case ShapeSampleCategory::kInside:
    return "vecgeom::test::EvaluateInsideConventionSample";
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
    return "vecgeom::test::EvaluateSurfaceConventionSample";
  case ShapeSampleCategory::kOutside:
    return "vecgeom::test::EvaluateOutsideConventionSample";
  default:
    return "vecgeom::test::ReplayShapeConventionSample";
  }
}

inline const char *ShapeContractGeometryFunctionName(const ShapeCheckContext &context)
{
  switch (context.convention_bit) {
  case kSurfaceDistanceToInEntering:
  case kSurfaceDistanceToInExiting:
  case kInsideDistanceToIn:
    return "vecgeom::VPlacedVolume::DistanceToIn";
  case kSurfaceDistanceToOutExiting:
  case kSurfaceDistanceToOutEntering:
  case kInsideDistanceToOut:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kSurfaceSafetyToIn:
  case kInsideSafetyToIn:
  case kOutsideSafetyToIn:
    return "vecgeom::VPlacedVolume::SafetyToIn";
  case kSurfaceSafetyToOut:
  case kInsideSafetyToOut:
  case kOutsideSafetyToOut:
    return "vecgeom::VPlacedVolume::SafetyToOut";
  case kOutsideDistanceToIn:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn";
  case kOutsideDistanceToOut:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kNormalSurfaceValid:
  case kNormalSurfaceUnitLength:
  case kNormalSurfaceOutward:
  case kNormalInsideExitValid:
  case kNormalInsideExitUnitLength:
  case kNormalInsideExitOutward:
  case kNormalOutsideEntryValid:
  case kNormalOutsideEntryUnitLength:
  case kNormalOutsideEntryInward:
    return "vecgeom::VPlacedVolume::Normal";
  case kNormalInsideExitOnSurface:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::Inside";
  case kNormalOutsideEntryOnSurface:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn + "
           "vecgeom::VPlacedVolume::Inside";
  case kSurfaceRayNotBothZero:
  case kSurfaceDistanceToOutFinite:
  case kSurfaceShallowInward:
  case kSurfaceShallowOutward:
    return "vecgeom::VPlacedVolume::DistanceToIn + vecgeom::VPlacedVolume::DistanceToOut";
  case kSurfaceGrazingNotBothZero:
    return "vecgeom::VPlacedVolume::Normal + vecgeom::VPlacedVolume::DistanceToIn + "
           "vecgeom::VPlacedVolume::DistanceToOut";
  case kDistanceToOutPositive:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kDistanceToOutWithinExtent:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::Extent";
  case kDistanceToOutAboveSafety:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::SafetyToOut";
  case kDistanceToOutOnSurface:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::Inside";
  case kDistanceToInApproachFinite:
    return "vecgeom::VUnplacedVolume::ApproachSolid";
  case kDistanceToInPositive:
  case kDistanceToInFinite:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn";
  case kDistanceToInAboveSafety:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn + "
           "vecgeom::VPlacedVolume::SafetyToIn";
  case kDistanceToInWithinTarget:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn + paired Inside target";
  case kDistanceToInOnSurface:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn + "
           "vecgeom::VPlacedVolume::Inside";
  case kSafetyToOutPositive:
    return "vecgeom::VPlacedVolume::SafetyToOut";
  case kSafetyToOutDistanceBound:
    return "vecgeom::VPlacedVolume::SafetyToOut + vecgeom::VPlacedVolume::DistanceToOut";
  case kSafetyToOutSafeSphere:
    return "vecgeom::VPlacedVolume::SafetyToOut + vecgeom::VPlacedVolume::Inside";
  case kSafetyToInPositive:
    return "vecgeom::VPlacedVolume::SafetyToIn";
  case kSafetyToInDistanceBound:
    return "vecgeom::VPlacedVolume::SafetyToIn + vecgeom::VPlacedVolume::DistanceToIn";
  case kSafetyToInSafeSphere:
    return "vecgeom::VPlacedVolume::SafetyToIn + vecgeom::VPlacedVolume::Inside";
  case kHitInsideExitFinite:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kHitInsideExitOnSurface:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::Inside";
  case kHitInsideExitSafetyToIn:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::SafetyToIn";
  case kHitInsideExitSafetyToOut:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::SafetyToOut";
  case kHitInsideExitNormalAgreement:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::Normal + "
           "vecgeom::VPlacedVolume::DistanceToIn";
  case kHitOutsideEntryFinite:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn";
  case kHitOutsideEntryOnSurface:
    return "vecgeom::VUnplacedVolume::ApproachSolid + vecgeom::VPlacedVolume::DistanceToIn + "
           "vecgeom::VPlacedVolume::Inside";
  case kHitOutsideEntrySafetyToIn:
    return "vecgeom::VPlacedVolume::SafetyToIn";
  case kHitOutsideEntrySafetyToOut:
    return "vecgeom::VPlacedVolume::SafetyToOut";
  case kHitOutsideEntryDistanceToIn:
    return "vecgeom::VPlacedVolume::DistanceToIn";
  case kHitOutsideEntryDistanceToOut:
  case kHitOutsideEntryDistanceSign:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kHitOutsideExitOnSurface:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::Inside";
  case kHitOutsideExitSafetyToIn:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::SafetyToIn";
  case kHitOutsideExitSafetyToOut:
    return "vecgeom::VPlacedVolume::DistanceToOut + vecgeom::VPlacedVolume::SafetyToOut";
  default:
    return "vecgeom::VPlacedVolume::Inside";
  }
}

// Report the leaf geometry method name separately so callers can rebuild a
// debugger-friendly concrete breakpoint such as
// PolyconeImplementation<ConeTypes::UniversalCone>::SafetyToOut.
inline const char *ShapeContractGeometryMethodName(const ShapeCheckContext &context)
{
  switch (context.convention_bit) {
  case kSurfaceDistanceToInEntering:
  case kSurfaceDistanceToInExiting:
  case kInsideDistanceToIn:
  case kOutsideDistanceToIn:
    return "DistanceToIn";
  case kSurfaceDistanceToOutExiting:
  case kSurfaceDistanceToOutEntering:
  case kInsideDistanceToOut:
  case kOutsideDistanceToOut:
    return "DistanceToOut";
  case kSurfaceSafetyToIn:
  case kInsideSafetyToIn:
  case kOutsideSafetyToIn:
    return "SafetyToIn";
  case kSurfaceSafetyToOut:
  case kInsideSafetyToOut:
  case kOutsideSafetyToOut:
    return "SafetyToOut";
  case kNormalSurfaceValid:
  case kNormalSurfaceUnitLength:
  case kNormalSurfaceOutward:
  case kNormalInsideExitValid:
  case kNormalInsideExitUnitLength:
  case kNormalInsideExitOutward:
  case kNormalOutsideEntryValid:
  case kNormalOutsideEntryUnitLength:
  case kNormalOutsideEntryInward:
    return "Normal";
  case kNormalInsideExitOnSurface:
    return "DistanceToOut";
  case kNormalOutsideEntryOnSurface:
    return "DistanceToIn";
  case kSurfaceRayNotBothZero:
  case kSurfaceGrazingNotBothZero:
  case kSurfaceShallowInward:
    return "DistanceToIn";
  case kSurfaceDistanceToOutFinite:
  case kSurfaceShallowOutward:
    return "DistanceToOut";
  case kDistanceToOutPositive:
  case kDistanceToOutWithinExtent:
  case kDistanceToOutAboveSafety:
  case kDistanceToOutOnSurface:
    return "DistanceToOut";
  case kDistanceToInApproachFinite:
    return "ApproachSolid";
  case kDistanceToInPositive:
  case kDistanceToInFinite:
  case kDistanceToInAboveSafety:
  case kDistanceToInWithinTarget:
  case kDistanceToInOnSurface:
    return "DistanceToIn";
  case kSafetyToOutPositive:
  case kSafetyToOutDistanceBound:
  case kSafetyToOutSafeSphere:
    return "SafetyToOut";
  case kSafetyToInPositive:
  case kSafetyToInDistanceBound:
  case kSafetyToInSafeSphere:
    return "SafetyToIn";
  case kHitInsideExitFinite:
  case kHitInsideExitOnSurface:
    return "DistanceToOut";
  case kHitInsideExitSafetyToIn:
    return "SafetyToIn";
  case kHitInsideExitSafetyToOut:
    return "SafetyToOut";
  case kHitInsideExitNormalAgreement:
    return "Normal";
  case kHitOutsideEntryFinite:
  case kHitOutsideEntryOnSurface:
  case kHitOutsideEntryDistanceToIn:
    return "DistanceToIn";
  case kHitOutsideEntrySafetyToIn:
    return "SafetyToIn";
  case kHitOutsideEntrySafetyToOut:
    return "SafetyToOut";
  case kHitOutsideEntryDistanceToOut:
  case kHitOutsideEntryDistanceSign:
  case kHitOutsideExitOnSurface:
    return "DistanceToOut";
  case kHitOutsideExitSafetyToIn:
    return "SafetyToIn";
  case kHitOutsideExitSafetyToOut:
    return "SafetyToOut";
  default:
    return "Inside";
  }
}

inline std::string ShapeContractImplementationFunctionName(const char *implementation_type,
                                                           const ShapeCheckContext &context)
{
  if (implementation_type == nullptr || implementation_type[0] == '\0') return "";
  std::ostringstream out;
  out << implementation_type << "::" << ShapeContractGeometryMethodName(context);
  return out.str();
}

inline const char *ShapeContractSupportFunctionName(const ShapeCheckContext &context)
{
  switch (context.convention_bit) {
  case kSurfaceDistanceToInEntering:
  case kSurfaceDistanceToInExiting:
  case kSurfaceDistanceToOutExiting:
  case kSurfaceDistanceToOutEntering:
    return "vecgeom::VPlacedVolume::Normal";
  case kNormalSurfaceOutward:
  case kNormalInsideExitOnSurface:
  case kNormalInsideExitOutward:
  case kNormalOutsideEntryOnSurface:
  case kNormalOutsideEntryInward:
    return "vecgeom::VPlacedVolume::Inside";
  case kSurfaceRayNotBothZero:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kSurfaceGrazingNotBothZero:
  case kSurfaceDistanceToOutFinite:
  case kSurfaceShallowInward:
  case kSurfaceShallowOutward:
    return "vecgeom::VPlacedVolume::Normal + vecgeom::VPlacedVolume::DistanceToOut";
  case kDistanceToInApproachFinite:
    return "vecgeom::VUnplacedVolume::ApproachSolid";
  case kDistanceToOutWithinExtent:
    return "vecgeom::VPlacedVolume::Extent";
  case kDistanceToOutAboveSafety:
    return "vecgeom::VPlacedVolume::SafetyToOut";
  case kDistanceToOutOnSurface:
    return "vecgeom::VPlacedVolume::Inside";
  case kDistanceToInPositive:
  case kDistanceToInFinite:
    return "vecgeom::VPlacedVolume::DistanceToIn";
  case kDistanceToInAboveSafety:
    return "vecgeom::VPlacedVolume::SafetyToIn";
  case kDistanceToInWithinTarget:
    return "paired Inside sample point";
  case kDistanceToInOnSurface:
    return "vecgeom::VPlacedVolume::Inside";
  case kSafetyToOutDistanceBound:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kSafetyToOutSafeSphere:
    return "vecgeom::VPlacedVolume::Inside";
  case kSafetyToInDistanceBound:
    return "vecgeom::VPlacedVolume::DistanceToIn";
  case kSafetyToInSafeSphere:
    return "vecgeom::VPlacedVolume::Inside";
  case kHitInsideExitOnSurface:
    return "vecgeom::VPlacedVolume::Inside";
  case kHitInsideExitSafetyToIn:
  case kHitInsideExitSafetyToOut:
    return "surface exit point";
  case kHitInsideExitNormalAgreement:
    return "vecgeom::VPlacedVolume::DistanceToIn";
  case kHitOutsideEntryOnSurface:
    return "vecgeom::VPlacedVolume::Inside";
  case kHitOutsideEntrySafetyToIn:
  case kHitOutsideEntrySafetyToOut:
    return "surface entry point";
  case kHitOutsideEntryDistanceToIn:
    return "surface entry point";
  case kHitOutsideEntryDistanceToOut:
  case kHitOutsideEntryDistanceSign:
    return "vecgeom::VPlacedVolume::DistanceToOut";
  case kHitOutsideExitOnSurface:
    return "vecgeom::VPlacedVolume::Inside";
  case kHitOutsideExitSafetyToIn:
  case kHitOutsideExitSafetyToOut:
    return "surface exit point";
  default:
    return "";
  }
}

inline std::string DescribeShapeContractDebugHint(const ShapeCheckContext &context)
{
  std::ostringstream out;
  out << "replay=" << ShapeContractReplayFunctionName(context);
  out << " evaluator=" << ShapeContractEvaluatorFunctionName(context);
  out << " geometry=" << ShapeContractGeometryFunctionName(context);
  const char *support = ShapeContractSupportFunctionName(context);
  if (support[0] != '\0') out << " support=" << support;
  return out.str();
}

class ShapeContractViolationSink {
public:
  // Record into the structured result store and optionally mirror the same
  // event to a legacy display/debugging sink owned by ShapeTester.
  ShapeContractViolationSink(ShapeCheckResult &result, int max_display, ShapeContractReporter reporter = {})
      : fResult(result), fMaxDisplay(max_display), fReporter(std::move(reporter))
  {
  }

  void Record(const std::string &message, const Vec_t &point, const Vec_t &direction, Precision distance,
              const ShapeCheckContext &context = {})
  {
    auto decision = fResult.Record(message, point, direction, distance, fMaxDisplay, context);
    if (fReporter) fReporter(decision, context, message, point, direction, distance);
  }

private:
  ShapeCheckResult &fResult;
  int fMaxDisplay;
  ShapeContractReporter fReporter;
};

struct ShapeContractCheckSummary {
  std::uint64_t score        = 0;
  bool surface_points_passed = true;
  bool inside_points_passed  = true;
  bool outside_points_passed = true;

  bool Passed() const { return surface_points_passed && inside_points_passed && outside_points_passed; }
};

struct ShapeContractFailure {
  ShapeCheckContext context;
  std::string contract_label;
  std::string message;
  Precision distance = 0.;
};

struct ShapeContractRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t direction;
  vecgeom::EnumInside inside_result = vecgeom::EnumInside::kOutside;
  bool valid_normal                 = false;
  Vec_t normal;
  Precision normal_dot_direction = 0.;
  Precision distance_to_in       = 0.;
  Precision distance_to_out      = 0.;
  Precision safety_to_in         = 0.;
  Precision safety_to_out        = 0.;
  Precision approach_distance    = 0.;
  Vec_t approach_point;
  Precision shifted_distance_to_in = 0.;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

enum class ShapeSurfaceKind { kUnknown = 0, kSmooth, kEdgeCandidate };

struct ShapeSurfaceCheckOptions {
  bool require_surface_distance_to_out_finite = false;
  bool enable_shallow_surface_rays            = false;
};

inline const ShapeSurfaceCheckOptions &DefaultShapeSurfaceCheckOptions()
{
  static const ShapeSurfaceCheckOptions options;
  return options;
}

struct ShapeTangentialProbeReplay;

template <typename ImplT>
ShapeSurfaceKind DetectSurfaceKind(ImplT const *volume, const Vec_t &point, const Vec_t &normal_unit,
                                   Precision solid_tolerance, std::vector<vecgeom::EnumInside> *probe_results = nullptr,
                                   std::vector<ShapeTangentialProbeReplay> *probe_details = nullptr);

inline const char *ShapeSurfaceKindLabel(ShapeSurfaceKind kind)
{
  switch (kind) {
  case ShapeSurfaceKind::kSmooth:
    return "smooth";
  case ShapeSurfaceKind::kEdgeCandidate:
    return "edge_candidate";
  default:
    return "unknown";
  }
}

struct ShapeNormalCheckSummary {
  std::uint64_t score       = 0;
  bool surface_passed       = true;
  bool inside_exit_passed   = true;
  bool outside_entry_passed = true;

  bool Passed() const { return surface_passed && inside_exit_passed && outside_entry_passed; }
};

struct ShapeTangentialProbeReplay {
  Precision probe_step = 0.;
  Vec_t probe_point;
  vecgeom::EnumInside probe_inside_result = vecgeom::EnumInside::kOutside;
  Precision reprojection_distance         = kInfLength;
  Vec_t boundary_point;
  vecgeom::EnumInside boundary_inside_result = vecgeom::EnumInside::kOutside;
  bool valid_normal                          = false;
  Vec_t normal;
  Precision normal_dot_reference = 0.;
};

struct ShapeNormalRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t direction;
  vecgeom::EnumInside point_inside_result = vecgeom::EnumInside::kOutside;
  Precision travel_distance               = 0.;
  Precision approach_distance             = 0.;
  Vec_t approach_point;
  Vec_t boundary_point;
  vecgeom::EnumInside boundary_inside_result = vecgeom::EnumInside::kOutside;
  bool valid_normal                          = false;
  Vec_t normal;
  Precision normal_magnitude               = 0.;
  Precision normal_dot_direction           = 0.;
  Precision normal_probe_step              = 0.;
  Precision tangential_probe_step          = 0.;
  vecgeom::EnumInside outward_probe_result = vecgeom::EnumInside::kOutside;
  vecgeom::EnumInside inward_probe_result  = vecgeom::EnumInside::kOutside;
  std::vector<vecgeom::EnumInside> tangential_probe_results;
  std::vector<ShapeTangentialProbeReplay> tangential_probe_details;
  ShapeSurfaceKind surface_kind = ShapeSurfaceKind::kUnknown;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

struct ShapeSurfaceCheckSummary {
  std::uint64_t score = 0;
  bool surface_passed = true;
  bool grazing_passed = true;
  bool shallow_passed = true;

  bool Passed() const { return surface_passed && grazing_passed && shallow_passed; }
};

struct ShapeSurfaceRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t direction;
  vecgeom::EnumInside point_inside_result = vecgeom::EnumInside::kOutside;
  bool valid_normal                       = false;
  Vec_t normal;
  ShapeSurfaceKind surface_kind = ShapeSurfaceKind::kUnknown;
  Precision grazing_tolerance   = 0.;
  Vec_t grazing_direction;
  Precision distance_to_in          = 0.;
  Precision distance_to_out         = 0.;
  Precision grazing_distance_to_in  = 0.;
  Precision grazing_distance_to_out = 0.;
  bool checked_shallow_rays         = false;
  Vec_t shallow_inward_direction;
  Vec_t shallow_outward_direction;
  Precision shallow_inward_distance_to_in   = 0.;
  Precision shallow_inward_distance_to_out  = 0.;
  Precision shallow_outward_distance_to_in  = 0.;
  Precision shallow_outward_distance_to_out = 0.;
  std::vector<vecgeom::EnumInside> tangential_probe_results;
  std::vector<ShapeTangentialProbeReplay> tangential_probe_details;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

struct ShapeDistanceToOutCheckSummary {
  std::uint64_t score       = 0;
  bool inside_points_passed = true;

  bool Passed() const { return inside_points_passed; }
};

struct ShapeDistanceToInCheckSummary {
  std::uint64_t score        = 0;
  bool outside_rays_passed   = true;
  int evaluated_outside_rays = 0;

  bool Passed() const { return outside_rays_passed; }
};

struct ShapeSafetyCheckSummary {
  std::uint64_t score          = 0;
  bool inside_points_passed    = true;
  bool outside_points_passed   = true;
  int evaluated_inside_points  = 0;
  int evaluated_outside_points = 0;

  bool Passed() const { return inside_points_passed && outside_points_passed; }
};

struct ShapeHitConsistencyCheckSummary {
  std::uint64_t score          = 0;
  bool inside_hits_passed      = true;
  bool outside_hits_passed     = true;
  int evaluated_inside_points  = 0;
  int evaluated_outside_points = 0;

  bool Passed() const { return inside_hits_passed && outside_hits_passed; }
};

struct ShapeDistanceToOutRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t direction;
  vecgeom::EnumInside point_inside_result = vecgeom::EnumInside::kOutside;
  Precision distance_to_out               = 0.;
  Precision safety_to_out                 = 0.;
  Precision max_extent_distance           = 0.;
  Vec_t boundary_point;
  vecgeom::EnumInside boundary_inside_result = vecgeom::EnumInside::kOutside;
  bool valid_normal                          = false;
  Vec_t normal;
  Precision normal_dot_direction = 0.;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

struct ShapeDistanceToInRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t target_point;
  int target_sample_index                  = -1;
  vecgeom::EnumInside target_inside_result = vecgeom::EnumInside::kOutside;
  Vec_t direction;
  Precision target_distance            = 0.;
  Precision safety_to_in               = 0.;
  Precision raw_approach_distance      = kInfLength;
  Precision adjusted_approach_distance = kInfLength;
  Vec_t approach_point;
  Precision distance_to_in  = kInfLength;
  Precision travel_distance = kInfLength;
  Vec_t boundary_point;
  vecgeom::EnumInside boundary_inside_result = vecgeom::EnumInside::kOutside;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

struct ShapeSafetyRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t direction;
  vecgeom::EnumInside point_inside_result = vecgeom::EnumInside::kOutside;
  Precision safety                        = 0.;
  Precision distance                      = 0.;
  Vec_t sphere_point;
  vecgeom::EnumInside sphere_inside_result = vecgeom::EnumInside::kOutside;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

struct ShapeHitConsistencyRayReplay {
  ShapeCheckContext context;
  Vec_t point;
  Vec_t target_point;
  int target_sample_index                  = -1;
  vecgeom::EnumInside point_inside_result  = vecgeom::EnumInside::kOutside;
  vecgeom::EnumInside target_inside_result = vecgeom::EnumInside::kOutside;
  Vec_t direction;
  Precision raw_approach_distance      = kInfLength;
  Precision adjusted_approach_distance = kInfLength;
  Vec_t approach_point;
  Precision travel_distance = kInfLength;
  Vec_t boundary_point;
  vecgeom::EnumInside boundary_inside_result = vecgeom::EnumInside::kOutside;
  Precision boundary_safety_to_in            = 0.;
  Precision boundary_safety_to_out           = 0.;
  Precision boundary_distance_to_in          = kInfLength;
  Precision boundary_distance_to_out         = kInfLength;
  bool boundary_valid_normal                 = false;
  Vec_t boundary_normal;
  Precision boundary_normal_dot_direction = 0.;
  Vec_t exit_point;
  vecgeom::EnumInside exit_inside_result = vecgeom::EnumInside::kOutside;
  Precision exit_safety_to_in            = 0.;
  Precision exit_safety_to_out           = 0.;
  std::vector<ShapeContractFailure> failures;

  bool Passed() const { return failures.empty(); }
};

inline std::string DescribeShapeCheckResult(const ShapeCheckResult &result, int max_violation_types = 3)
{
  if (result.CountErrors() == 0) return "";

  std::ostringstream out;
  out << "\nRecorded violations:";
  int shown = 0;
  for (auto const &violation : result.Violations()) {
    if (shown++ == max_violation_types) {
      out << "\n... (" << result.CountViolationTypes() - max_violation_types << " more violation types)";
      break;
    }

    out << "\n- message: " << violation.message;
    if (violation.convention_bit >= 0) {
      out << "\n  contract: [" << violation.convention_bit << "] " << ShapeConventionLabel(violation.convention_bit);
    }
    out << "\n  count: " << violation.count;
    for (auto const &occurrence : violation.displayed_occurrences) {
      out << "\n  occurrence: " << FormatShapeCheckContext(occurrence.context)
          << " point=" << FormatVec(occurrence.point) << " direction=" << FormatVec(occurrence.direction)
          << " distance=" << occurrence.distance;
    }
  }
  return out.str();
}

inline std::string DescribeShapeNormalRayReplay(const ShapeNormalRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";
  out << "Inside(point)=" << InsideLabel(replay.point_inside_result) << "\n";
  out << "travel_distance=" << replay.travel_distance << "\n";
  if (replay.context.sample_group == ShapeSampleCategory::kOutside) {
    out << "ApproachSolid=" << replay.approach_distance << "\n";
    out << "ApproachPoint=" << FormatVec(replay.approach_point) << "\n";
  }
  out << "boundary_point=" << FormatVec(replay.boundary_point) << "\n";
  out << "Inside(boundary_point)=" << InsideLabel(replay.boundary_inside_result) << "\n";
  out << "valid_normal=" << (replay.valid_normal ? "true" : "false") << "\n";
  out << "normal=" << FormatVec(replay.normal) << "\n";
  out << "normal_magnitude=" << replay.normal_magnitude << "\n";
  out << "dot(direction, normal)=" << replay.normal_dot_direction << "\n";
  out << "normal_probe_step=" << replay.normal_probe_step << "\n";
  out << "tangential_probe_step=" << replay.tangential_probe_step << "\n";
  out << "Inside(boundary + step * normal)=" << InsideLabel(replay.outward_probe_result) << "\n";
  out << "Inside(boundary - step * normal)=" << InsideLabel(replay.inward_probe_result) << "\n";
  out << "surface_kind=" << ShapeSurfaceKindLabel(replay.surface_kind) << "\n";
  if (!replay.tangential_probe_results.empty()) {
    out << "tangential_probes:";
    for (size_t i = 0; i < replay.tangential_probe_results.size(); ++i) {
      out << "\n  probe[" << i << "]=" << InsideLabel(replay.tangential_probe_results[i]);
    }
    out << "\n";
  }
  if (!replay.tangential_probe_details.empty()) {
    out << "tangential_probe_details:";
    for (size_t i = 0; i < replay.tangential_probe_details.size(); ++i) {
      auto const &probe = replay.tangential_probe_details[i];
      out << "\n  probe[" << i << "] step=" << probe.probe_step << " point=" << FormatVec(probe.probe_point)
          << " Inside(probe)=" << InsideLabel(probe.probe_inside_result)
          << " reprojection_distance=" << probe.reprojection_distance
          << " boundary_point=" << FormatVec(probe.boundary_point)
          << " Inside(boundary)=" << InsideLabel(probe.boundary_inside_result)
          << " valid_normal=" << (probe.valid_normal ? "true" : "false")
          << " dot(reference_normal, probe_normal)=" << probe.normal_dot_reference;
    }
    out << "\n";
  }

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

inline std::string DescribeShapeSurfaceRayReplay(const ShapeSurfaceRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";
  out << "Inside(point)=" << InsideLabel(replay.point_inside_result) << "\n";
  out << "valid_normal=" << (replay.valid_normal ? "true" : "false") << "\n";
  out << "normal=" << FormatVec(replay.normal) << "\n";
  out << "surface_kind=" << ShapeSurfaceKindLabel(replay.surface_kind) << "\n";
  out << "grazing_tolerance=" << replay.grazing_tolerance << "\n";
  out << "DistanceToIn=" << replay.distance_to_in << "\n";
  out << "DistanceToOut=" << replay.distance_to_out << "\n";
  if (replay.valid_normal) {
    out << "grazing_direction=" << FormatVec(replay.grazing_direction) << "\n";
    out << "GrazingDistanceToIn=" << replay.grazing_distance_to_in << "\n";
    out << "GrazingDistanceToOut=" << replay.grazing_distance_to_out << "\n";
  }
  if (replay.checked_shallow_rays) {
    out << "shallow_inward_direction=" << FormatVec(replay.shallow_inward_direction) << "\n";
    out << "ShallowInwardDistanceToIn=" << replay.shallow_inward_distance_to_in << "\n";
    out << "ShallowInwardDistanceToOut=" << replay.shallow_inward_distance_to_out << "\n";
    out << "shallow_outward_direction=" << FormatVec(replay.shallow_outward_direction) << "\n";
    out << "ShallowOutwardDistanceToIn=" << replay.shallow_outward_distance_to_in << "\n";
    out << "ShallowOutwardDistanceToOut=" << replay.shallow_outward_distance_to_out << "\n";
  }
  if (!replay.tangential_probe_results.empty()) {
    out << "tangential_probes:";
    for (size_t i = 0; i < replay.tangential_probe_results.size(); ++i) {
      out << "\n  probe[" << i << "]=" << InsideLabel(replay.tangential_probe_results[i]);
    }
    out << "\n";
  }
  if (!replay.tangential_probe_details.empty()) {
    out << "tangential_probe_details:";
    for (size_t i = 0; i < replay.tangential_probe_details.size(); ++i) {
      auto const &probe = replay.tangential_probe_details[i];
      out << "\n  probe[" << i << "] step=" << probe.probe_step << " point=" << FormatVec(probe.probe_point)
          << " Inside(probe)=" << InsideLabel(probe.probe_inside_result)
          << " reprojection_distance=" << probe.reprojection_distance
          << " boundary_point=" << FormatVec(probe.boundary_point)
          << " Inside(boundary)=" << InsideLabel(probe.boundary_inside_result)
          << " valid_normal=" << (probe.valid_normal ? "true" : "false")
          << " dot(reference_normal, probe_normal)=" << probe.normal_dot_reference;
    }
    out << "\n";
  }

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

inline std::string DescribeShapeDistanceToOutRayReplay(const ShapeDistanceToOutRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";
  out << "Inside(point)=" << InsideLabel(replay.point_inside_result) << "\n";
  out << "DistanceToOut=" << replay.distance_to_out << "\n";
  out << "SafetyToOut=" << replay.safety_to_out << "\n";
  out << "max_extent_distance=" << replay.max_extent_distance << "\n";
  out << "boundary_point=" << FormatVec(replay.boundary_point) << "\n";
  out << "Inside(boundary_point)=" << InsideLabel(replay.boundary_inside_result) << "\n";
  out << "valid_normal=" << (replay.valid_normal ? "true" : "false") << "\n";
  out << "normal=" << FormatVec(replay.normal) << "\n";
  out << "dot(direction, normal)=" << replay.normal_dot_direction << "\n";

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

inline std::string DescribeShapeDistanceToInRayReplay(const ShapeDistanceToInRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "target_point=" << FormatVec(replay.target_point) << "\n";
  out << "target_sample_index=" << replay.target_sample_index << "\n";
  out << "Inside(target_point)=" << InsideLabel(replay.target_inside_result) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";
  out << "target_distance=" << replay.target_distance << "\n";
  out << "SafetyToIn=" << replay.safety_to_in << "\n";
  out << "raw_approach_distance=" << replay.raw_approach_distance << "\n";
  out << "adjusted_approach_distance=" << replay.adjusted_approach_distance << "\n";
  out << "approach_point=" << FormatVec(replay.approach_point) << "\n";
  out << "DistanceToIn=" << replay.distance_to_in << "\n";
  out << "travel_distance=" << replay.travel_distance << "\n";
  out << "boundary_point=" << FormatVec(replay.boundary_point) << "\n";
  out << "Inside(boundary_point)=" << InsideLabel(replay.boundary_inside_result) << "\n";

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

inline std::string DescribeShapeSafetyRayReplay(const ShapeSafetyRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";
  out << "Inside(point)=" << InsideLabel(replay.point_inside_result) << "\n";
  out << "safety=" << replay.safety << "\n";
  out << "paired_distance=" << replay.distance << "\n";
  out << "sphere_point=" << FormatVec(replay.sphere_point) << "\n";
  out << "Inside(sphere_point)=" << InsideLabel(replay.sphere_inside_result) << "\n";

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

inline std::string DescribeShapeHitConsistencyRayReplay(const ShapeHitConsistencyRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "Inside(point)=" << InsideLabel(replay.point_inside_result) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";

  if (replay.context.sample_group == ShapeSampleCategory::kOutside) {
    out << "target_point=" << FormatVec(replay.target_point) << "\n";
    out << "target_sample_index=" << replay.target_sample_index << "\n";
    out << "Inside(target_point)=" << InsideLabel(replay.target_inside_result) << "\n";
    out << "raw_approach_distance=" << replay.raw_approach_distance << "\n";
    out << "adjusted_approach_distance=" << replay.adjusted_approach_distance << "\n";
    out << "approach_point=" << FormatVec(replay.approach_point) << "\n";
  }

  out << "travel_distance=" << replay.travel_distance << "\n";
  out << "boundary_point=" << FormatVec(replay.boundary_point) << "\n";
  out << "Inside(boundary_point)=" << InsideLabel(replay.boundary_inside_result) << "\n";
  out << "boundary_safety_to_in=" << replay.boundary_safety_to_in << "\n";
  out << "boundary_safety_to_out=" << replay.boundary_safety_to_out << "\n";
  out << "boundary_distance_to_in=" << replay.boundary_distance_to_in << "\n";
  out << "boundary_distance_to_out=" << replay.boundary_distance_to_out << "\n";
  out << "boundary_valid_normal=" << (replay.boundary_valid_normal ? "true" : "false") << "\n";
  out << "boundary_normal=" << FormatVec(replay.boundary_normal) << "\n";
  out << "dot(direction, boundary_normal)=" << replay.boundary_normal_dot_direction << "\n";

  if (replay.context.sample_group == ShapeSampleCategory::kOutside) {
    out << "exit_point=" << FormatVec(replay.exit_point) << "\n";
    out << "Inside(exit_point)=" << InsideLabel(replay.exit_inside_result) << "\n";
    out << "exit_safety_to_in=" << replay.exit_safety_to_in << "\n";
    out << "exit_safety_to_out=" << replay.exit_safety_to_out << "\n";
  }

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

inline std::string DescribeShapeContractRayReplay(const ShapeContractRayReplay &replay)
{
  std::ostringstream out;
  out << "Replaying " << FormatShapeCheckContext(replay.context) << "\n";
  out << "point=" << FormatVec(replay.point) << "\n";
  out << "direction=" << FormatVec(replay.direction) << "\n";
  out << "Inside(point)=" << InsideLabel(replay.inside_result) << "\n";
  out << "valid_normal=" << (replay.valid_normal ? "true" : "false") << "\n";
  out << "normal=" << FormatVec(replay.normal) << "\n";
  out << "dot(direction, normal)=" << replay.normal_dot_direction << "\n";
  out << "DistanceToIn=" << replay.distance_to_in << "\n";
  out << "DistanceToOut=" << replay.distance_to_out << "\n";
  out << "SafetyToIn=" << replay.safety_to_in << "\n";
  out << "SafetyToOut=" << replay.safety_to_out << "\n";
  if (replay.context.sample_group == ShapeSampleCategory::kOutside) {
    out << "ApproachSolid=" << replay.approach_distance << "\n";
    out << "ApproachPoint=" << FormatVec(replay.approach_point) << "\n";
    out << "ShiftedDistanceToIn=" << replay.shifted_distance_to_in << "\n";
  }

  if (replay.failures.empty()) {
    out << "failing_contracts=none";
    return out.str();
  }

  out << "failing_contracts:";
  for (auto const &failure : replay.failures) {
    out << "\n- " << FormatShapeCheckContext(failure.context);
    if (!failure.contract_label.empty()) out << "\n  contract: " << failure.contract_label;
    out << "\n  debug_hint: " << DescribeShapeContractDebugHint(failure.context);
    out << "\n  message: " << failure.message;
    out << "\n  observed_distance: " << failure.distance;
  }
  return out.str();
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateSurfaceConventionSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeContractRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context   = context;
    replay->point     = point;
    replay->direction = direction;
  }

  auto inside_result = volume->Inside(point);
  if (replay) replay->inside_result = inside_result;
  if (inside_result != vecgeom::EnumInside::kSurface) {
    record_failure(context, "For Surface point, Inside says that the Point is not on the Surface", 0.);
  }

  Vec_t normal(0., 0., 0.);
  bool valid_normal = volume->Normal(point, normal);
  if (replay) {
    replay->valid_normal         = valid_normal;
    replay->normal               = normal;
    replay->normal_dot_direction = direction.Dot(normal);
  }

  ShapeSurfaceKind surface_kind = ShapeSurfaceKind::kUnknown;
  if (valid_normal) {
    const Precision normal_magnitude = normal.Mag();
    if (normal_magnitude > 0.) {
      Vec_t normal_unit(normal);
      normal_unit /= normal_magnitude;
      surface_kind = DetectSurfaceKind(volume, point, normal_unit, solid_tolerance);
    }
  }

  const Precision normal_dot_direction = direction.Dot(normal);
  Precision distance_to_in             = volume->DistanceToIn(point, direction);

  Vec_t norm(0., 0., 0.);
  Precision distance_to_out = call_distance_to_out(volume, point, direction, norm);
  if (replay) {
    replay->distance_to_in  = distance_to_in;
    replay->distance_to_out = distance_to_out;
  }

  const bool smooth_inward_ray   = surface_kind == ShapeSurfaceKind::kSmooth && normal_dot_direction < 0.;
  const bool accepted_entry      = distance_to_in <= solid_tolerance;
  const bool useful_continuation = distance_to_out > solid_tolerance;

  if (smooth_inward_ray && useful_continuation) {
    bool ok =
        distance_to_in < kInfLength && vecCore::math::Abs(distance_to_in * normal_dot_direction) <= solid_tolerance;
    if (valid_normal && !ok) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSurfaceDistanceToInEntering},
                     "DistanceToIn for Surface Point entering into the Shape should be 0 within tolerance (VecGeom "
                     "convention)",
                     distance_to_in);
    }
  }

  const bool convex_shape = false;
  if (surface_kind == ShapeSurfaceKind::kSmooth && normal_dot_direction > 0. && distance_to_in == kInfLength) {
    if (convex_shape) {
      if (!vecgeom::test::ApproxEqual<Precision>(distance_to_in, static_cast<Precision>(kInfLength))) {
        passed = false;
        record_failure({context.sample_index, context.sample_group, kSurfaceDistanceToInExiting},
                       "DistanceToIn for Surface Point exiting the Shape should be > 0.", distance_to_in);
      }
    } else if (!(distance_to_in > 0.)) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSurfaceDistanceToInExiting},
                     "DistanceToIn for Surface Point exiting the Shape should be > 0.", distance_to_in);
    }
  }

  if (surface_kind == ShapeSurfaceKind::kSmooth && normal_dot_direction > 0. && distance_to_out == kInfLength) {
    bool ok = (distance_to_out * normal_dot_direction) <= solid_tolerance;
    if (!ok) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSurfaceDistanceToOutExiting},
                     "DistanceToOut for Surface Point exiting the shape should be <= tolerance (VecGeom convention)",
                     distance_to_out);
    }
  }

  if (smooth_inward_ray && accepted_entry) {
    if (!useful_continuation) {
      if (valid_normal) {
        passed = false;
        record_failure({context.sample_index, context.sample_group, kSurfaceDistanceToOutEntering},
                       "DistanceToOut for Surface Point entering into the Shape should be > tolerance.",
                       distance_to_out);
      }
    }
  }

  Precision dist = volume->SafetyToIn(point);
  if (replay) replay->safety_to_in = dist;
  if (!(dist <= solid_tolerance)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceSafetyToIn},
                   "SafetyToIn for Surface Point should be <= tolerance (VecGeom convention)", dist);
  }

  dist = volume->SafetyToOut(point);
  if (replay) replay->safety_to_out = dist;
  if (!(dist <= solid_tolerance)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceSafetyToOut},
                   "SafetyToOut for Surface Point should be <= tolerance (VecGeom convention)", dist);
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckSurfaceConventions(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                             DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                             std::uint64_t &score)
{
  bool surface_point_convention_passed = true;
  for (int i = 0; i < samples.max_points_surface + samples.max_points_edge; ++i) {
    const int sample_index = samples.offset_surface + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateSurfaceConventionSample(
        volume, point, direction, solid_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    surface_point_convention_passed = surface_point_convention_passed && sample_passed;
  }

  return surface_point_convention_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateInsideConventionSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, DistanceToOutCaller &&call_distance_to_out,
    const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeContractRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context   = context;
    replay->point     = point;
    replay->direction = direction;
  }

  auto inside_result = volume->Inside(point);
  if (replay) replay->inside_result = inside_result;
  if (inside_result != vecgeom::EnumInside::kInside) {
    record_failure(context, "For Inside point, Inside function says that the Point is not inside", 0.);
  }

  Precision dist = volume->DistanceToIn(point, direction);
  if (replay) replay->distance_to_in = dist;
  if (!(dist < 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kInsideDistanceToIn},
                   "DistanceToIn for Inside Point should be Negative (-1.) (Wrong side)", dist);
  }

  Vec_t norm(0., 0., 0.);
  dist = call_distance_to_out(volume, point, direction, norm);
  if (replay) replay->distance_to_out = dist;
  if (dist == kInfLength) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kInsideDistanceToOut},
                   "DistanceToOut for Inside Point can never be Infinity", dist);
  }

  dist = volume->SafetyToIn(point);
  if (replay) replay->safety_to_in = dist;
  if (!(dist < 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kInsideSafetyToIn},
                   "SafetyToIn for Inside Point should be Negative (-1.) (Wrong side, VecGeom convention)", dist);
  }

  dist = volume->SafetyToOut(point);
  if (replay) replay->safety_to_out = dist;
  if (!(dist > 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kInsideSafetyToOut},
                   "SafetyToOut for Inside Point should be > 0.", dist);
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckInsideConventions(ImplT const *volume, const ShapeContractSampleView &samples,
                            DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                            std::uint64_t &score)
{
  bool inside_point_convention_passed = true;

  for (int i = 0; i < samples.max_points_inside; ++i) {
    const int sample_index = samples.offset_inside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateInsideConventionSample(
        volume, point, direction, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    inside_point_convention_passed = inside_point_convention_passed && sample_passed;
  }

  return inside_point_convention_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateOutsideConventionSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, DistanceToOutCaller &&call_distance_to_out,
    const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeContractRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context   = context;
    replay->point     = point;
    replay->direction = direction;
  }

  auto inside_result = volume->Inside(point);
  if (replay) replay->inside_result = inside_result;
  if (inside_result != vecgeom::EnumInside::kOutside) {
    record_failure(context, "For Outside point, Inside function says that the Point is not Outside", 0.);
  }

  const Vec_t invdir(1. / NonZero(direction.x()), 1. / NonZero(direction.y()), 1. / NonZero(direction.z()));
  Precision dist_bb = volume->GetUnplacedVolume()->ApproachSolid(point, invdir);
  Vec_t point_bb(point);
  Precision shifted_dist_to_in = kInfLength;
  Precision dist               = kInfLength;
  // A missed bounding-box approach is still a valid outside-point contract
  // case: DistanceToIn remains infinite, but we must not propagate the point
  // to infinity and feed non-finite coordinates back into the solid.
  if (dist_bb < kInfLength) {
    point_bb           = point + dist_bb * direction;
    shifted_dist_to_in = volume->DistanceToIn(point_bb, direction);
    dist               = shifted_dist_to_in + dist_bb;
  }
  if (replay) {
    replay->approach_distance      = dist_bb;
    replay->approach_point         = point_bb;
    replay->distance_to_in         = dist;
    replay->shifted_distance_to_in = shifted_dist_to_in;
  }
  if (!(dist > 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kOutsideDistanceToIn},
                   "DistanceToIn for Outside Point should be > 0.", dist);
  }

  Vec_t norm(0., 0., 0.);
  dist = call_distance_to_out(volume, point, direction, norm);
  if (replay) replay->distance_to_out = dist;
  if (!(dist < 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kOutsideDistanceToOut},
                   "DistanceToOut for Outside Point should be Negative (-1.) (Wrong side).", dist);
  }

  dist = volume->SafetyToIn(point);
  if (replay) replay->safety_to_in = dist;
  if (!(dist > 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kOutsideSafetyToIn},
                   "SafetyToIn for Outside Point should be > 0.", dist);
  }

  dist = volume->SafetyToOut(point);
  if (replay) replay->safety_to_out = dist;
  if (!(dist < 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kOutsideSafetyToOut},
                   "SafetyToOut for Outside Point should be Negative (-1) (Wrong side)", dist);
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckOutsideConventions(ImplT const *volume, const ShapeContractSampleView &samples,
                             DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                             std::uint64_t &score)
{
  bool outside_point_convention_passed = true;

  for (int i = 0; i < samples.max_points_outside; ++i) {
    const int sample_index = samples.offset_outside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateOutsideConventionSample(
        volume, point, direction, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    outside_point_convention_passed = outside_point_convention_passed && sample_passed;
  }

  return outside_point_convention_passed;
}

inline Precision ShapeNormalProbeStep(Precision solid_tolerance)
{
  if (solid_tolerance > 0.) return solid_tolerance * static_cast<Precision>(0.25);
  return vecgeom::kTolerance * static_cast<Precision>(0.25);
}

inline Precision ShapeTangentialProbeBaseTolerance(Precision solid_tolerance)
{
  return solid_tolerance > 0. ? solid_tolerance : static_cast<Precision>(vecgeom::kTolerance);
}

inline std::array<Precision, 2> ShapeTangentialProbeSteps(Precision solid_tolerance)
{
  // Use two finite lateral scales. The smaller one stays very local, while the
  // larger one helps distinguish true singular points from purely numerical
  // surface fuzz. Each probe is reprojected onto the surface before its nearby
  // normal is compared against the reference normal so curved smooth surfaces
  // can still classify as non-edge-like.
  const Precision base_tolerance = ShapeTangentialProbeBaseTolerance(solid_tolerance);
  return {base_tolerance * static_cast<Precision>(10.), base_tolerance * static_cast<Precision>(100.)};
}

inline Precision ShapeTangentialProbeStep(Precision solid_tolerance)
{
  return ShapeTangentialProbeSteps(solid_tolerance).back();
}

inline Precision ShapeTangentialProbeDistanceLimit(Precision probe_step)
{
  return probe_step * static_cast<Precision>(4.);
}

inline Precision ShapeSurfaceNegativeDistanceTolerance(Precision solid_tolerance)
{
  return ShapeTangentialProbeDistanceLimit(ShapeTangentialProbeStep(solid_tolerance));
}

inline Precision ShapeTangentialNormalAgreementThreshold() { return static_cast<Precision>(0.95); }

template <typename ImplT>
inline Precision ShapeExtentDistance(ImplT const *volume)
{
  Vec_t minExtent;
  Vec_t maxExtent;
  volume->Extent(minExtent, maxExtent);
  const Precision maxX = std::max(std::fabs(maxExtent.x()), std::fabs(minExtent.x()));
  const Precision maxY = std::max(std::fabs(maxExtent.y()), std::fabs(minExtent.y()));
  const Precision maxZ = std::max(std::fabs(maxExtent.z()), std::fabs(minExtent.z()));
  return static_cast<Precision>(2.) * std::sqrt(maxX * maxX + maxY * maxY + maxZ * maxZ);
}

inline bool BuildTangentialBasis(const Vec_t &normal_unit, Vec_t &tangent_a, Vec_t &tangent_b)
{
  Vec_t reference(1., 0., 0.);
  if (std::fabs(normal_unit.x()) > static_cast<Precision>(0.8)) reference = Vec_t(0., 1., 0.);
  tangent_a = normal_unit.Cross(reference);
  if (tangent_a.Mag2() <= 0.) return false;
  tangent_a.Normalize();
  tangent_b = normal_unit.Cross(tangent_a);
  if (tangent_b.Mag2() <= 0.) return false;
  tangent_b.Normalize();
  return true;
}

inline bool BuildPreferredTangentialDirection(const Vec_t &normal_unit, const Vec_t &preferred_direction,
                                              Vec_t &tangent_direction)
{
  tangent_direction = preferred_direction - preferred_direction.Dot(normal_unit) * normal_unit;
  if (tangent_direction.Mag2() > 0.) {
    tangent_direction.Normalize();
    return true;
  }

  Vec_t tangent_b(0., 0., 0.);
  if (!BuildTangentialBasis(normal_unit, tangent_direction, tangent_b)) return false;
  return true;
}

inline void ApplyGrazingTolerance(const Vec_t &normal_unit, Precision grazing_tolerance, Vec_t &grazing_direction)
{
  if (!(grazing_tolerance > 0.)) return;
  // Keep the exact-grazing rule strict by default, but allow the caller to
  // deterministically tilt the tangential ray into a near-grazing one when a
  // wider scan around the ideal tangent is needed for debugging or sampling.
  grazing_direction += grazing_tolerance * normal_unit;
  grazing_direction.Normalize();
}

inline Precision ShapeShallowSurfaceRayTilt(Precision solid_tolerance)
{
  return ShapeTangentialProbeBaseTolerance(solid_tolerance);
}

inline int ShapeShallowSurfaceRayStride() { return 16; }

inline bool ShouldCheckShallowSurfaceRays(const ShapeCheckContext &context)
{
  return (context.sample_index % ShapeShallowSurfaceRayStride()) == 0;
}

template <typename ImplT>
bool EvaluateTangentialNormalProbe(ImplT const *volume, const Vec_t &probe_point, const Vec_t &reference_normal_unit,
                                   Precision probe_step, ShapeTangentialProbeReplay *probe_replay = nullptr,
                                   vecgeom::EnumInside *probe_inside_result = nullptr)
{
  const Precision max_reprojection_distance = ShapeTangentialProbeDistanceLimit(probe_step);
  auto inside_result                        = volume->Inside(probe_point);
  if (probe_inside_result) *probe_inside_result = inside_result;
  if (probe_replay) {
    probe_replay->probe_step             = probe_step;
    probe_replay->probe_point            = probe_point;
    probe_replay->probe_inside_result    = inside_result;
    probe_replay->reprojection_distance  = kInfLength;
    probe_replay->boundary_inside_result = vecgeom::EnumInside::kOutside;
    probe_replay->valid_normal           = false;
    probe_replay->normal_dot_reference   = static_cast<Precision>(-1.);
  }

  auto try_candidate = [&](const Vec_t &travel_direction, Precision travel_distance, ShapeTangentialProbeReplay &best,
                           bool &found_candidate) {
    if (!(travel_distance >= static_cast<Precision>(0.)) || travel_distance >= kInfLength ||
        travel_distance > max_reprojection_distance) {
      return;
    }

    const Vec_t candidate_point = probe_point + travel_distance * travel_direction;

    ShapeTangentialProbeReplay candidate;
    candidate.probe_step             = probe_step;
    candidate.probe_point            = probe_point;
    candidate.probe_inside_result    = inside_result;
    candidate.reprojection_distance  = travel_distance;
    candidate.boundary_point         = candidate_point;
    candidate.boundary_inside_result = volume->Inside(candidate_point);
    if (candidate.boundary_inside_result != vecgeom::EnumInside::kSurface) return;

    Vec_t normal(0., 0., 0.);
    candidate.valid_normal = volume->Normal(candidate_point, normal);
    candidate.normal       = normal;
    if (!candidate.valid_normal) return;

    const Precision magnitude = normal.Mag();
    if (magnitude <= static_cast<Precision>(0.)) return;
    candidate.normal /= magnitude;
    candidate.normal_dot_reference = candidate.normal.Dot(reference_normal_unit);

    if (!found_candidate || candidate.normal_dot_reference > best.normal_dot_reference ||
        (vecgeom::test::ApproxEqual<Precision>(candidate.normal_dot_reference, best.normal_dot_reference) &&
         candidate.reprojection_distance < best.reprojection_distance)) {
      best            = candidate;
      found_candidate = true;
    }
  };

  if (inside_result == vecgeom::EnumInside::kSurface) {
    ShapeTangentialProbeReplay best;
    bool found_candidate = false;
    try_candidate(Vec_t(0., 0., 0.), static_cast<Precision>(0.), best, found_candidate);
    if (probe_replay && found_candidate) *probe_replay = best;
    return found_candidate && best.normal_dot_reference >= ShapeTangentialNormalAgreementThreshold();
  }

  ShapeTangentialProbeReplay best;
  bool found_candidate = false;
  if (inside_result == vecgeom::EnumInside::kOutside) {
    const Precision inward_distance  = volume->DistanceToIn(probe_point, -reference_normal_unit);
    const Precision outward_distance = volume->DistanceToIn(probe_point, reference_normal_unit);
    try_candidate(-reference_normal_unit, inward_distance, best, found_candidate);
    try_candidate(reference_normal_unit, outward_distance, best, found_candidate);
  } else if (inside_result == vecgeom::EnumInside::kInside) {
    Vec_t scratch_normal(0., 0., 0.);
    const Precision outward_distance = volume->DistanceToOut(probe_point, reference_normal_unit, scratch_normal);
    const Precision inward_distance  = volume->DistanceToOut(probe_point, -reference_normal_unit, scratch_normal);
    try_candidate(reference_normal_unit, outward_distance, best, found_candidate);
    try_candidate(-reference_normal_unit, inward_distance, best, found_candidate);
  }

  if (probe_replay && found_candidate) *probe_replay = best;
  // If none of the local reprojections reaches a nearby boundary point with a
  // stable normal, treat the original hit as edge-like or singular.
  return found_candidate && best.normal_dot_reference >= ShapeTangentialNormalAgreementThreshold();
}

template <typename ImplT>
ShapeSurfaceKind DetectSurfaceKind(ImplT const *volume, const Vec_t &point, const Vec_t &normal_unit,
                                   Precision solid_tolerance, std::vector<vecgeom::EnumInside> *probe_results,
                                   std::vector<ShapeTangentialProbeReplay> *probe_details)
{
  if (probe_results) probe_results->clear();
  if (probe_details) probe_details->clear();

  Vec_t tangent_a(0., 0., 0.);
  Vec_t tangent_b(0., 0., 0.);
  if (!BuildTangentialBasis(normal_unit, tangent_a, tangent_b)) return ShapeSurfaceKind::kUnknown;

  bool all_probes_stable = true;
  for (auto const &probe_step : ShapeTangentialProbeSteps(solid_tolerance)) {
    const std::array<Vec_t, 4> probes = {point + probe_step * tangent_a, point - probe_step * tangent_a,
                                         point + probe_step * tangent_b, point - probe_step * tangent_b};

    for (auto const &probe : probes) {
      ShapeTangentialProbeReplay probe_replay;
      vecgeom::EnumInside probe_inside = vecgeom::EnumInside::kOutside;
      bool probe_passed = EvaluateTangentialNormalProbe(volume, probe, normal_unit, probe_step,
                                                        probe_details ? &probe_replay : nullptr, &probe_inside);
      if (probe_results) probe_results->push_back(probe_inside);
      if (probe_details) probe_details->push_back(probe_replay);
      all_probes_stable = all_probes_stable && probe_passed;
    }
  }
  return all_probes_stable ? ShapeSurfaceKind::kSmooth : ShapeSurfaceKind::kEdgeCandidate;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateSurfacePointSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    Precision grazing_tolerance, DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeSurfaceRayReplay *replay           = nullptr,
    const ShapeSurfaceCheckOptions &options = DefaultShapeSurfaceCheckOptions())
{
  bool passed = true;

  if (replay) {
    replay->context           = context;
    replay->point             = point;
    replay->direction         = direction;
    replay->grazing_tolerance = grazing_tolerance;
  }

  const auto inside_result = volume->Inside(point);
  if (replay) replay->point_inside_result = inside_result;
  if (inside_result != vecgeom::EnumInside::kSurface) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceRayNotBothZero},
                   "Surface point check requires Inside(point) == kSurface.", 0.);
    return passed;
  }

  Vec_t normal(0., 0., 0.);
  const bool valid_normal = volume->Normal(point, normal);
  if (replay) {
    replay->valid_normal = valid_normal;
    replay->normal       = normal;
  }

  Vec_t distance_normal(0., 0., 0.);
  const Precision distance_to_in  = volume->DistanceToIn(point, direction);
  const Precision distance_to_out = call_distance_to_out(volume, point, direction, distance_normal);
  if (replay) {
    replay->distance_to_in  = distance_to_in;
    replay->distance_to_out = distance_to_out;
  }
  if (distance_to_in <= vecgeom::kTolerance && distance_to_out <= vecgeom::kTolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceRayNotBothZero},
                   "DistanceToIn and DistanceToOut cannot both be zero for Surface ray.", distance_to_out);
  }
  auto check_distance_to_out_finite = [&](Precision distance, Precision negative_distance_tolerance) {
    if (distance >= -negative_distance_tolerance && distance < kInfLength) return;
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceDistanceToOutFinite},
                   "DistanceToOut for Surface ray must be finite.", distance);
  };
  if (options.require_surface_distance_to_out_finite) check_distance_to_out_finite(distance_to_out, solid_tolerance);

  if (!valid_normal) return passed;

  const Precision normal_magnitude = normal.Mag();
  if (!(normal_magnitude > 0.)) return passed;

  Vec_t normal_unit(normal);
  normal_unit /= normal_magnitude;

  auto surface_kind = DetectSurfaceKind(volume, point, normal_unit, solid_tolerance,
                                        replay ? &replay->tangential_probe_results : nullptr,
                                        replay ? &replay->tangential_probe_details : nullptr);
  if (replay) replay->surface_kind = surface_kind;
  if (surface_kind != ShapeSurfaceKind::kSmooth) return passed;

  Vec_t tangent_direction(0., 0., 0.);
  if (!BuildPreferredTangentialDirection(normal_unit, direction, tangent_direction)) return passed;
  Vec_t grazing_direction(tangent_direction);
  ApplyGrazingTolerance(normal_unit, grazing_tolerance, grazing_direction);
  if (replay) replay->grazing_direction = grazing_direction;

  const Precision grazing_distance_to_in  = volume->DistanceToIn(point, grazing_direction);
  const Precision grazing_distance_to_out = call_distance_to_out(volume, point, grazing_direction, distance_normal);
  if (replay) {
    replay->grazing_distance_to_in  = grazing_distance_to_in;
    replay->grazing_distance_to_out = grazing_distance_to_out;
  }
  if (grazing_distance_to_in <= vecgeom::kTolerance && grazing_distance_to_out <= vecgeom::kTolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceGrazingNotBothZero},
                   "DistanceToIn and DistanceToOut cannot both be zero for grazing Surface ray.",
                   grazing_distance_to_out);
  }
  if (options.require_surface_distance_to_out_finite) {
    check_distance_to_out_finite(grazing_distance_to_out, solid_tolerance);
  }

  if (!options.enable_shallow_surface_rays || !ShouldCheckShallowSurfaceRays(context)) return passed;

  const Precision shallow_tilt = ShapeShallowSurfaceRayTilt(solid_tolerance);
  Vec_t shallow_inward_direction(tangent_direction - shallow_tilt * normal_unit);
  shallow_inward_direction.Normalize();
  Vec_t shallow_outward_direction(tangent_direction + shallow_tilt * normal_unit);
  shallow_outward_direction.Normalize();

  const Precision shallow_inward_distance_to_in = volume->DistanceToIn(point, shallow_inward_direction);
  const Precision shallow_inward_distance_to_out =
      call_distance_to_out(volume, point, shallow_inward_direction, distance_normal);
  const Precision shallow_outward_distance_to_in = volume->DistanceToIn(point, shallow_outward_direction);
  const Precision shallow_outward_distance_to_out =
      call_distance_to_out(volume, point, shallow_outward_direction, distance_normal);
  if (replay) {
    replay->checked_shallow_rays            = true;
    replay->shallow_inward_direction        = shallow_inward_direction;
    replay->shallow_outward_direction       = shallow_outward_direction;
    replay->shallow_inward_distance_to_in   = shallow_inward_distance_to_in;
    replay->shallow_inward_distance_to_out  = shallow_inward_distance_to_out;
    replay->shallow_outward_distance_to_in  = shallow_outward_distance_to_in;
    replay->shallow_outward_distance_to_out = shallow_outward_distance_to_out;
  }
  if (options.require_surface_distance_to_out_finite) {
    const Precision shallow_negative_distance_tolerance = ShapeSurfaceNegativeDistanceTolerance(solid_tolerance);
    check_distance_to_out_finite(shallow_inward_distance_to_out, shallow_negative_distance_tolerance);
    check_distance_to_out_finite(shallow_outward_distance_to_out, shallow_negative_distance_tolerance);
  }

  const Precision shallow_inward_projection = shallow_inward_direction.Dot(normal);
  if (shallow_inward_distance_to_out > vecgeom::kTolerance) {
    const bool shallow_inward_ok =
        shallow_inward_distance_to_in < kInfLength &&
        vecCore::math::Abs(shallow_inward_distance_to_in * shallow_inward_projection) <= solid_tolerance;
    if (!shallow_inward_ok) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSurfaceShallowInward},
                     "DistanceToIn for shallow inward Surface ray should be 0 within tolerance.",
                     shallow_inward_distance_to_in);
    }
  }

  const Precision shallow_outward_projection          = shallow_outward_direction.Dot(normal);
  const Precision shallow_negative_distance_tolerance = ShapeSurfaceNegativeDistanceTolerance(solid_tolerance);
  const bool shallow_outward_ok =
      shallow_outward_distance_to_out >= -shallow_negative_distance_tolerance &&
      shallow_outward_distance_to_out < kInfLength &&
      vecCore::math::Abs(shallow_outward_distance_to_out * shallow_outward_projection) <= solid_tolerance;
  if (!shallow_outward_ok) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kSurfaceShallowOutward},
                   "DistanceToOut for shallow outward Surface ray should be 0 within tolerance.",
                   shallow_outward_distance_to_out);
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckSurfacePoints(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                        Precision grazing_tolerance, DistanceToOutCaller &&call_distance_to_out,
                        ShapeContractViolationSink &sink, std::uint64_t &score,
                        const ShapeSurfaceCheckOptions &options = DefaultShapeSurfaceCheckOptions())
{
  bool surface_points_passed = true;
  for (int i = 0; i < samples.max_points_surface + samples.max_points_edge; ++i) {
    const int sample_index = samples.offset_surface + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateSurfacePointSample(
        volume, point, direction, solid_tolerance, grazing_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        },
        nullptr, options);
    surface_points_passed = surface_points_passed && sample_passed;
  }
  return surface_points_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateInsideDistanceToOutSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    Precision max_extent_distance, DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeDistanceToOutRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context             = context;
    replay->point               = point;
    replay->direction           = direction;
    replay->max_extent_distance = max_extent_distance;
  }

  const auto inside_result = volume->Inside(point);
  if (replay) replay->point_inside_result = inside_result;
  if (inside_result != vecgeom::EnumInside::kInside) {
    passed = false;
    record_failure(context, "DistanceToOut check requires Inside(point) == kInside.", 0.);
    return passed;
  }

  const Precision safe_distance = volume->SafetyToOut(point);
  if (replay) replay->safety_to_out = safe_distance;

  Vec_t propagated_normal(0., 0., 0.);
  const Precision dist = call_distance_to_out(volume, point, direction, propagated_normal);
  if (replay) replay->distance_to_out = dist;

  if (!(dist > 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToOutPositive},
                   "DistanceToOut for Inside Point should be > 0.", dist);
  }

  if (!(dist < kInfLength) || !(dist <= max_extent_distance)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToOutWithinExtent},
                   "DistanceToOut for Inside Point should be finite and not exceed the solid extent.", dist);
  }

  if (safe_distance > 0. && dist < safe_distance - solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToOutAboveSafety},
                   "DistanceToOut for Inside Point should be >= SafetyToOut(point) within tolerance.", safe_distance);
  }

  if (dist > 0. && dist < kInfLength && dist <= max_extent_distance) {
    const Vec_t boundary_point = point + dist * direction;
    const auto boundary_inside = volume->Inside(boundary_point);
    if (replay) {
      replay->boundary_point         = boundary_point;
      replay->boundary_inside_result = boundary_inside;
      replay->normal                 = propagated_normal;
      replay->normal_dot_direction   = propagated_normal.Dot(direction);
      replay->valid_normal           = volume->Normal(boundary_point, replay->normal);
      replay->normal_dot_direction   = replay->normal.Dot(direction);
    }

    if (boundary_inside != vecgeom::EnumInside::kSurface) {
      bool effectively_on_surface = false;
      if (boundary_inside == vecgeom::EnumInside::kInside) {
        effectively_on_surface = volume->SafetyToOut(boundary_point) <= solid_tolerance;
      } else if (boundary_inside == vecgeom::EnumInside::kOutside) {
        effectively_on_surface = volume->SafetyToIn(boundary_point) <= solid_tolerance;
      }
      if (!effectively_on_surface) {
        passed = false;
        const std::string message =
            boundary_inside == vecgeom::EnumInside::kInside
                ? "DistanceToOut for Inside Point undershoots: propagated exit point should be on the Surface."
                : "DistanceToOut for Inside Point overshoots: propagated exit point should be on the Surface.";
        record_failure({context.sample_index, context.sample_group, kDistanceToOutOnSurface}, message, dist);
      }
    }
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckInsideDistanceToOutSamples(ImplT const *volume, const ShapeContractSampleView &samples,
                                     Precision solid_tolerance, Precision max_extent_distance,
                                     DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                                     std::uint64_t &score)
{
  bool inside_points_passed = true;
  for (int i = 0; i < samples.max_points_inside; ++i) {
    const int sample_index = samples.offset_inside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateInsideDistanceToOutSample(
        volume, point, direction, solid_tolerance, max_extent_distance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    inside_points_passed = inside_points_passed && sample_passed;
  }
  return inside_points_passed;
}

template <typename ImplT>
bool EvaluateOutsideDistanceToInSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &target_point, Precision solid_tolerance,
    const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeDistanceToInRayReplay *replay = nullptr)
{
  bool passed = true;

  const Vec_t target_delta        = target_point - point;
  const Precision target_distance = target_delta.Mag();
  const Vec_t direction           = target_distance > 0. ? target_delta.Unit() : Vec_t(0., 0., 0.);

  if (replay) {
    replay->context         = context;
    replay->point           = point;
    replay->target_point    = target_point;
    replay->direction       = direction;
    replay->target_distance = target_distance;
  }

  const auto target_inside = volume->Inside(target_point);
  if (replay) replay->target_inside_result = target_inside;
  if (target_inside != vecgeom::EnumInside::kInside) {
    passed = false;
    record_failure(context, "DistanceToIn check requires the paired target point to be Inside.", 0.);
    return passed;
  }

  if (!(target_distance > 0.)) {
    passed = false;
    record_failure(context, "DistanceToIn check requires distinct Outside and Inside points.", target_distance);
    return passed;
  }

  const Precision safe_distance = volume->SafetyToIn(point);
  if (replay) replay->safety_to_in = safe_distance;

  const Vec_t invdir(1. / NonZero(direction.x()), 1. / NonZero(direction.y()), 1. / NonZero(direction.z()));
  const Precision raw_dist_bb  = volume->GetUnplacedVolume()->ApproachSolid(point, invdir);
  const Precision tolerance_bb = static_cast<Precision>(10.) * solid_tolerance;
  const Precision dist_bb =
      raw_dist_bb < kInfLength ? (raw_dist_bb > tolerance_bb ? raw_dist_bb - tolerance_bb : 0.) : kInfLength;

  if (replay) {
    replay->raw_approach_distance      = raw_dist_bb;
    replay->adjusted_approach_distance = dist_bb;
    replay->approach_point             = raw_dist_bb < kInfLength ? point + dist_bb * direction : point;
  }

  if (!(raw_dist_bb < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToInApproachFinite},
                   "ApproachSolid for Outside -> Inside ray should be finite.", raw_dist_bb);
    return passed;
  }

  const Vec_t approach_point = point + dist_bb * direction;
  const Precision dist_in    = volume->DistanceToIn(approach_point, direction);
  const Precision total_dist = dist_bb + dist_in;
  if (replay) {
    replay->approach_point  = approach_point;
    replay->distance_to_in  = dist_in;
    replay->travel_distance = total_dist;
  }

  if (!(dist_in > 0.)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToInPositive},
                   "DistanceToIn for Outside -> Inside ray should be > 0.", dist_in);
  }

  if (!(dist_in < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToInFinite},
                   "DistanceToIn for Outside -> Inside ray should be finite.", dist_in);
  }

  if (safe_distance > 0. && total_dist < safe_distance - solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kDistanceToInAboveSafety},
                   "DistanceToIn for Outside -> Inside ray should be >= SafetyToIn(point) within tolerance.",
                   safe_distance);
  }

  if (dist_in > 0. && dist_in < kInfLength) {
    if (total_dist > target_distance + solid_tolerance) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kDistanceToInWithinTarget},
                     "DistanceToIn for Outside -> Inside ray should reach the Surface before the paired Inside target.",
                     total_dist);
    }

    const Vec_t boundary_point = approach_point + dist_in * direction;
    const auto boundary_inside = volume->Inside(boundary_point);
    if (replay) {
      replay->boundary_point         = boundary_point;
      replay->boundary_inside_result = boundary_inside;
    }

    if (boundary_inside != vecgeom::EnumInside::kSurface) {
      passed = false;
      const std::string message =
          boundary_inside == vecgeom::EnumInside::kOutside
              ? "DistanceToIn for Outside -> Inside ray undershoots: propagated entry point should be on the Surface."
              : "DistanceToIn for Outside -> Inside ray overshoots: propagated entry point should be on the Surface.";
      record_failure({context.sample_index, context.sample_group, kDistanceToInOnSurface}, message, total_dist);
    }
  }

  return passed;
}

template <typename ImplT>
bool CheckOutsideDistanceToInSamples(ImplT const *volume, const ShapeContractSampleView &samples,
                                     Precision solid_tolerance, ShapeContractViolationSink &sink, std::uint64_t &score,
                                     int &evaluated_outside_rays)
{
  bool outside_rays_passed = true;
  if (samples.max_points_inside <= 0) return outside_rays_passed;

  for (int i = 0; i < samples.max_points_outside; ++i) {
    const int sample_index        = samples.offset_outside + i;
    const int target_sample_index = PairedInsideSampleIndex(samples, sample_index);
    if (target_sample_index < 0) continue;
    ++evaluated_outside_rays;

    const Vec_t &point        = samples.Point(sample_index);
    const Vec_t &target_point = samples.Point(target_sample_index);
    const auto context        = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed        = EvaluateOutsideDistanceToInSample(
        volume, point, target_point, solid_tolerance, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0)
            score |= (static_cast<std::uint64_t>(1) << failure_context.convention_bit);
          sink.Record(message, point, target_point - point, distance, failure_context);
        });
    outside_rays_passed = outside_rays_passed && sample_passed;
  }
  return outside_rays_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateShapeSafetySample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeSafetyRayReplay *replay = nullptr)
{
  bool passed              = true;
  const auto inside_result = volume->Inside(point);

  if (replay) {
    replay->context             = context;
    replay->point               = point;
    replay->direction           = direction;
    replay->point_inside_result = inside_result;
  }

  if (context.sample_group == ShapeSampleCategory::kInside) {
    const Precision safety = volume->SafetyToOut(point);
    Vec_t propagated_normal(0., 0., 0.);
    const Precision dist     = call_distance_to_out(volume, point, direction, propagated_normal);
    const Vec_t sphere_point = point + safety * direction;
    const auto sphere_inside = volume->Inside(sphere_point);

    if (replay) {
      replay->safety               = safety;
      replay->distance             = dist;
      replay->sphere_point         = sphere_point;
      replay->sphere_inside_result = sphere_inside;
    }

    if (!(safety > 0.)) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSafetyToOutPositive},
                     "SafetyToOut for Inside Point should be > 0.", safety);
      return passed;
    }

    if (dist < safety - solid_tolerance) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSafetyToOutDistanceBound},
                     "SafetyToOut for Inside Point should not exceed DistanceToOut(point, direction) within tolerance.",
                     dist);
    }

    if (sphere_inside == vecgeom::EnumInside::kOutside) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSafetyToOutSafeSphere},
                     "Point on the SafetyToOut sphere should stay Inside or on the Surface.", safety);
    }
    return passed;
  }

  if (context.sample_group == ShapeSampleCategory::kOutside) {
    const Precision safety   = volume->SafetyToIn(point);
    const Precision dist     = volume->DistanceToIn(point, direction);
    const Vec_t sphere_point = point + safety * direction;
    const auto sphere_inside = volume->Inside(sphere_point);

    if (replay) {
      replay->safety               = safety;
      replay->distance             = dist;
      replay->sphere_point         = sphere_point;
      replay->sphere_inside_result = sphere_inside;
    }

    if (!(safety > 0.)) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSafetyToInPositive},
                     "SafetyToIn for Outside Point should be > 0.", safety);
      return passed;
    }

    if (dist < safety - solid_tolerance) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSafetyToInDistanceBound},
                     "SafetyToIn for Outside Point should not exceed DistanceToIn(point, direction) within tolerance.",
                     dist);
    }

    if (sphere_inside == vecgeom::EnumInside::kInside) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kSafetyToInSafeSphere},
                     "Point on the SafetyToIn sphere should stay Outside or on the Surface.", safety);
    }
    return passed;
  }

  passed = false;
  record_failure(context, "Safety checks require an Inside or Outside sample.", 0.);
  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckInsideSafetySamples(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                              DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                              std::uint64_t &score, int &evaluated_inside_points)
{
  bool inside_points_passed = true;
  for (int i = 0; i < samples.max_points_inside; ++i) {
    const int sample_index = samples.offset_inside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    ++evaluated_inside_points;
    bool sample_passed = EvaluateShapeSafetySample(
        volume, point, direction, solid_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0)
            score |= (static_cast<std::uint64_t>(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    inside_points_passed = inside_points_passed && sample_passed;
  }
  return inside_points_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckOutsideSafetySamples(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                               DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                               std::uint64_t &score, int &evaluated_outside_points)
{
  bool outside_points_passed = true;
  for (int i = 0; i < samples.max_points_outside; ++i) {
    const int sample_index = samples.offset_outside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    ++evaluated_outside_points;
    bool sample_passed = EvaluateShapeSafetySample(
        volume, point, direction, solid_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0)
            score |= (static_cast<std::uint64_t>(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    outside_points_passed = outside_points_passed && sample_passed;
  }
  return outside_points_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateInsideHitConsistencySample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeHitConsistencyRayReplay *replay = nullptr)
{
  bool passed = true;

  const auto inside_result = volume->Inside(point);
  if (replay) {
    replay->context             = context;
    replay->point               = point;
    replay->direction           = direction;
    replay->point_inside_result = inside_result;
  }

  if (inside_result != vecgeom::EnumInside::kInside) {
    passed = false;
    record_failure(context, "Hit-consistency check requires Inside(point) == kInside.", 0.);
    return passed;
  }

  Vec_t propagated_normal(0., 0., 0.);
  const Precision dist = call_distance_to_out(volume, point, direction, propagated_normal);
  if (replay) replay->travel_distance = dist;
  if (!(dist > 0.) || !(dist < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitInsideExitFinite},
                   "DistanceToOut for Inside consistency ray should be finite and > 0.", dist);
    return passed;
  }

  const Vec_t boundary_point = point + dist * direction;
  const auto boundary_inside = volume->Inside(boundary_point);
  if (replay) {
    replay->boundary_point         = boundary_point;
    replay->boundary_inside_result = boundary_inside;
  }
  if (boundary_inside != vecgeom::EnumInside::kSurface) {
    bool effectively_on_surface = false;
    if (boundary_inside == vecgeom::EnumInside::kInside) {
      effectively_on_surface = volume->SafetyToOut(boundary_point) <= solid_tolerance;
    } else if (boundary_inside == vecgeom::EnumInside::kOutside) {
      effectively_on_surface = volume->SafetyToIn(boundary_point) <= solid_tolerance;
    }
    if (!effectively_on_surface) {
      passed = false;
      const std::string message =
          boundary_inside == vecgeom::EnumInside::kInside
              ? "Inside consistency ray undershoots: propagated exit point should be on the Surface."
              : "Inside consistency ray overshoots: propagated exit point should be on the Surface.";
      record_failure({context.sample_index, context.sample_group, kHitInsideExitOnSurface}, message, dist);
      return passed;
    }
  }

  const Precision safety_to_in  = volume->SafetyToIn(boundary_point);
  const Precision safety_to_out = volume->SafetyToOut(boundary_point);
  Vec_t boundary_normal(0., 0., 0.);
  bool valid_boundary_normal = volume->Normal(boundary_point, boundary_normal);
  if (replay) {
    replay->boundary_safety_to_in         = safety_to_in;
    replay->boundary_safety_to_out        = safety_to_out;
    replay->boundary_normal               = boundary_normal;
    replay->boundary_valid_normal         = valid_boundary_normal;
    replay->boundary_normal_dot_direction = boundary_normal.Dot(direction);
  }

  if (safety_to_in > solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitInsideExitSafetyToIn},
                   "SafetyToIn at the propagated exit point should stay within tolerance.", safety_to_in);
  }

  if (safety_to_out > solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitInsideExitSafetyToOut},
                   "SafetyToOut at the propagated exit point should stay within tolerance.", safety_to_out);
  }

  if (valid_boundary_normal && boundary_normal.Dot(direction) < 0.) {
    const Precision boundary_distance_to_in = volume->DistanceToIn(boundary_point, direction);
    if (replay) replay->boundary_distance_to_in = boundary_distance_to_in;
    if (std::fabs(boundary_distance_to_in) > solid_tolerance) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kHitInsideExitNormalAgreement},
                     "Normal and DistanceToIn should agree at the propagated exit point for Inside consistency rays.",
                     boundary_distance_to_in);
    }
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckInsideHitConsistencySamples(ImplT const *volume, const ShapeContractSampleView &samples,
                                      Precision solid_tolerance, DistanceToOutCaller &&call_distance_to_out,
                                      ShapeContractViolationSink &sink, std::uint64_t &score,
                                      int &evaluated_inside_points)
{
  bool inside_hits_passed = true;
  for (int i = 0; i < samples.max_points_inside; ++i) {
    const int sample_index = samples.offset_inside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    ++evaluated_inside_points;
    bool sample_passed = EvaluateInsideHitConsistencySample(
        volume, point, direction, solid_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0)
            score |= (static_cast<std::uint64_t>(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    inside_hits_passed = inside_hits_passed && sample_passed;
  }
  return inside_hits_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateOutsideHitConsistencySample(
    ImplT const *volume, const Vec_t &point, const Vec_t &target_point, int target_sample_index,
    Precision solid_tolerance, DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeHitConsistencyRayReplay *replay = nullptr)
{
  bool passed = true;

  const auto point_inside       = volume->Inside(point);
  const auto target_inside      = volume->Inside(target_point);
  const Vec_t target_delta      = target_point - point;
  const Precision target_length = target_delta.Mag();
  const Vec_t direction         = target_length > 0. ? target_delta.Unit() : Vec_t(0., 0., 0.);

  if (replay) {
    replay->context              = context;
    replay->point                = point;
    replay->target_point         = target_point;
    replay->target_sample_index  = target_sample_index;
    replay->point_inside_result  = point_inside;
    replay->target_inside_result = target_inside;
    replay->direction            = direction;
  }

  if (point_inside != vecgeom::EnumInside::kOutside || target_inside != vecgeom::EnumInside::kInside ||
      !(target_length > 0.)) {
    passed = false;
    record_failure(context, "Hit-consistency outside check requires paired Outside and Inside points.", target_length);
    return passed;
  }

  const Vec_t invdir(1. / NonZero(direction.x()), 1. / NonZero(direction.y()), 1. / NonZero(direction.z()));
  const Precision raw_dist_bb  = volume->GetUnplacedVolume()->ApproachSolid(point, invdir);
  const Precision tolerance_bb = static_cast<Precision>(10.) * solid_tolerance;
  const Precision dist_bb =
      raw_dist_bb < kInfLength ? (raw_dist_bb > tolerance_bb ? raw_dist_bb - tolerance_bb : 0.) : kInfLength;

  if (replay) {
    replay->raw_approach_distance      = raw_dist_bb;
    replay->adjusted_approach_distance = dist_bb;
    replay->approach_point             = raw_dist_bb < kInfLength ? point + dist_bb * direction : point;
  }

  if (!(raw_dist_bb < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntryFinite},
                   "DistanceToIn for Outside consistency ray should be finite after bounding-box approach.",
                   raw_dist_bb);
    return passed;
  }

  const Vec_t approach_point = point + dist_bb * direction;
  const Precision dist_in    = volume->DistanceToIn(approach_point, direction);
  if (replay) {
    replay->approach_point          = approach_point;
    replay->boundary_distance_to_in = dist_in;
    replay->travel_distance         = dist_bb + dist_in;
  }

  if (!(dist_in > 0.) || !(dist_in < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntryFinite},
                   "DistanceToIn for Outside consistency ray should be finite and > 0.", dist_in);
    return passed;
  }

  const Vec_t boundary_point = approach_point + dist_in * direction;
  const auto boundary_inside = volume->Inside(boundary_point);
  if (replay) {
    replay->boundary_point         = boundary_point;
    replay->boundary_inside_result = boundary_inside;
  }
  if (boundary_inside != vecgeom::EnumInside::kSurface) {
    bool effectively_on_surface = false;
    if (boundary_inside == vecgeom::EnumInside::kInside) {
      effectively_on_surface = volume->SafetyToOut(boundary_point) <= solid_tolerance;
    } else if (boundary_inside == vecgeom::EnumInside::kOutside) {
      effectively_on_surface = volume->SafetyToIn(boundary_point) <= solid_tolerance;
    }
    if (!effectively_on_surface) {
      passed = false;
      const std::string message =
          boundary_inside == vecgeom::EnumInside::kOutside
              ? "Outside consistency ray undershoots: propagated entry point should be on the Surface."
              : "Outside consistency ray overshoots: propagated entry point should be on the Surface.";
      record_failure({context.sample_index, context.sample_group, kHitOutsideEntryOnSurface}, message,
                     dist_bb + dist_in);
      return passed;
    }
  }

  const Precision boundary_safety_to_in   = volume->SafetyToIn(boundary_point);
  const Precision boundary_safety_to_out  = volume->SafetyToOut(boundary_point);
  const Precision boundary_distance_to_in = volume->DistanceToIn(boundary_point, direction);
  Vec_t exit_normal(0., 0., 0.);
  const Precision boundary_distance_to_out = call_distance_to_out(volume, boundary_point, direction, exit_normal);

  if (replay) {
    replay->boundary_safety_to_in         = boundary_safety_to_in;
    replay->boundary_safety_to_out        = boundary_safety_to_out;
    replay->boundary_distance_to_in       = boundary_distance_to_in;
    replay->boundary_distance_to_out      = boundary_distance_to_out;
    replay->boundary_normal               = exit_normal;
    replay->boundary_valid_normal         = volume->Normal(boundary_point, replay->boundary_normal);
    replay->boundary_normal_dot_direction = replay->boundary_normal.Dot(direction);
  }

  if (boundary_safety_to_in > solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntrySafetyToIn},
                   "SafetyToIn at the propagated entry point should stay within tolerance.", boundary_safety_to_in);
  }

  if (boundary_safety_to_out > solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntrySafetyToOut},
                   "SafetyToOut at the propagated entry point should stay within tolerance.", boundary_safety_to_out);
  }

  if (!(boundary_distance_to_in < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntryDistanceToIn},
                   "DistanceToIn at the propagated entry point should stay finite.", boundary_distance_to_in);
  }

  if (!(boundary_distance_to_out < kInfLength)) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntryDistanceToOut},
                   "DistanceToOut at the propagated entry point should stay finite.", boundary_distance_to_out);
    return passed;
  }

  if (boundary_distance_to_out < -solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideEntryDistanceSign},
                   "DistanceToOut at the propagated entry point should not be negative beyond tolerance.",
                   boundary_distance_to_out);
    return passed;
  }

  const Precision exit_distance = boundary_distance_to_out > 0. ? boundary_distance_to_out : static_cast<Precision>(0.);
  const Vec_t exit_point        = boundary_point + exit_distance * direction;
  const auto exit_inside        = volume->Inside(exit_point);
  const Precision exit_safety_to_in  = volume->SafetyToIn(exit_point);
  const Precision exit_safety_to_out = volume->SafetyToOut(exit_point);
  if (replay) {
    replay->exit_point         = exit_point;
    replay->exit_inside_result = exit_inside;
    replay->exit_safety_to_in  = exit_safety_to_in;
    replay->exit_safety_to_out = exit_safety_to_out;
  }

  if (exit_inside != vecgeom::EnumInside::kSurface) {
    bool effectively_on_surface = false;
    if (exit_inside == vecgeom::EnumInside::kInside) {
      effectively_on_surface = volume->SafetyToOut(exit_point) <= solid_tolerance;
    } else if (exit_inside == vecgeom::EnumInside::kOutside) {
      effectively_on_surface = volume->SafetyToIn(exit_point) <= solid_tolerance;
    }
    if (!effectively_on_surface) {
      passed = false;
      const std::string message =
          exit_inside == vecgeom::EnumInside::kInside
              ? "DistanceToOut from the propagated entry point undershoots: exit point should be on the Surface."
              : "DistanceToOut from the propagated entry point overshoots: exit point should be on the Surface.";
      record_failure({context.sample_index, context.sample_group, kHitOutsideExitOnSurface}, message, exit_distance);
      return passed;
    }
  }

  if (exit_safety_to_in > solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideExitSafetyToIn},
                   "SafetyToIn at the propagated exit point should stay within tolerance.", exit_safety_to_in);
  }

  if (exit_safety_to_out > solid_tolerance) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kHitOutsideExitSafetyToOut},
                   "SafetyToOut at the propagated exit point should stay within tolerance.", exit_safety_to_out);
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckOutsideHitConsistencySamples(ImplT const *volume, const ShapeContractSampleView &samples,
                                       Precision solid_tolerance, DistanceToOutCaller &&call_distance_to_out,
                                       ShapeContractViolationSink &sink, std::uint64_t &score,
                                       int &evaluated_outside_points)
{
  bool outside_hits_passed = true;
  if (samples.max_points_inside <= 0) return outside_hits_passed;

  for (int i = 0; i < samples.max_points_outside; ++i) {
    const int sample_index        = samples.offset_outside + i;
    const int target_sample_index = PairedInsideSampleIndex(samples, sample_index);
    if (target_sample_index < 0) continue;
    ++evaluated_outside_points;

    const Vec_t &point        = samples.Point(sample_index);
    const Vec_t &target_point = samples.Point(target_sample_index);
    const auto context        = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed        = EvaluateOutsideHitConsistencySample(
        volume, point, target_point, target_sample_index, solid_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0)
            score |= (static_cast<std::uint64_t>(1) << failure_context.convention_bit);
          sink.Record(message, point, target_point - point, distance, failure_context);
        });
    outside_hits_passed = outside_hits_passed && sample_passed;
  }
  return outside_hits_passed;
}

template <typename ImplT>
bool EvaluateSurfaceNormalSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeNormalRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context        = context;
    replay->point          = point;
    replay->direction      = direction;
    replay->boundary_point = point;
  }

  auto inside_result = volume->Inside(point);
  if (replay) {
    replay->point_inside_result    = inside_result;
    replay->boundary_inside_result = inside_result;
  }
  if (inside_result != vecgeom::EnumInside::kSurface) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalSurfaceValid},
                   "Surface normal check requires Inside(point) == kSurface.", 0.);
    return passed;
  }

  Vec_t normal(0., 0., 0.);
  bool valid_normal = volume->Normal(point, normal);
  if (replay) {
    replay->valid_normal          = valid_normal;
    replay->normal                = normal;
    replay->normal_magnitude      = normal.Mag();
    replay->normal_dot_direction  = normal.Dot(direction);
    replay->normal_probe_step     = ShapeNormalProbeStep(solid_tolerance);
    replay->tangential_probe_step = ShapeTangentialProbeStep(solid_tolerance);
  }
  if (!valid_normal) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalSurfaceValid},
                   "Normal for Surface Point should be valid.", 0.);
    return passed;
  }

  const Precision magnitude = normal.Mag();
  if (!vecgeom::test::ApproxEqual<Precision>(magnitude, static_cast<Precision>(1.0))) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalSurfaceUnitLength},
                   "Normal for Surface Point should have unit length.", magnitude);
  }

  Vec_t normal_unit(normal);
  if (magnitude > 0.) normal_unit /= magnitude;
  std::vector<vecgeom::EnumInside> tangential_probe_results;
  std::vector<ShapeTangentialProbeReplay> tangential_probe_details;
  auto surface_kind = DetectSurfaceKind(volume, point, normal_unit, solid_tolerance,
                                        replay ? &replay->tangential_probe_results : &tangential_probe_results,
                                        replay ? &replay->tangential_probe_details : &tangential_probe_details);
  if (replay) replay->surface_kind = surface_kind;

  const Vec_t outward_probe = point + ShapeNormalProbeStep(solid_tolerance) * normal_unit;
  const Vec_t inward_probe  = point - ShapeNormalProbeStep(solid_tolerance) * normal_unit;
  auto outward_inside       = volume->Inside(outward_probe);
  auto inward_inside        = volume->Inside(inward_probe);
  if (replay) {
    replay->outward_probe_result = outward_inside;
    replay->inward_probe_result  = inward_inside;
  }

  if (surface_kind == ShapeSurfaceKind::kSmooth) {
    if (outward_inside == vecgeom::EnumInside::kInside || inward_inside == vecgeom::EnumInside::kOutside) {
      passed = false;
      record_failure({context.sample_index, context.sample_group, kNormalSurfaceOutward},
                     "Normal for Surface Point should point topologically outwards.", normal.Dot(direction));
    }
  }

  return passed;
}

template <typename ImplT>
bool CheckSurfaceNormals(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                         ShapeContractViolationSink &sink, std::uint64_t &score)
{
  bool surface_normals_passed = true;
  for (int i = 0; i < samples.max_points_surface + samples.max_points_edge; ++i) {
    const int sample_index = samples.offset_surface + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateSurfaceNormalSample(
        volume, point, direction, solid_tolerance, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    surface_normals_passed = surface_normals_passed && sample_passed;
  }
  return surface_normals_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool EvaluateInsideExitNormalSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    DistanceToOutCaller &&call_distance_to_out, const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeNormalRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context   = context;
    replay->point     = point;
    replay->direction = direction;
  }

  auto inside_result = volume->Inside(point);
  if (replay) replay->point_inside_result = inside_result;

  Vec_t propagated_normal(0., 0., 0.);
  Precision dist = call_distance_to_out(volume, point, direction, propagated_normal);
  if (replay) replay->travel_distance = dist;
  if (!(dist >= 0.) || dist >= kInfLength) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalInsideExitOnSurface},
                   "Inside exit normal check requires a finite DistanceToOut.", dist);
    return passed;
  }

  const Vec_t boundary_point = point + dist * direction;
  auto boundary_inside       = volume->Inside(boundary_point);
  if (replay) {
    replay->boundary_point         = boundary_point;
    replay->boundary_inside_result = boundary_inside;
  }
  if (boundary_inside != vecgeom::EnumInside::kSurface) {
    if (boundary_inside == vecgeom::EnumInside::kInside) {
      const Precision boundary_safety_out = volume->SafetyToOut(boundary_point);
      if (boundary_safety_out <= solid_tolerance) return passed;
    } else if (boundary_inside == vecgeom::EnumInside::kOutside) {
      const Precision boundary_safety_in = volume->SafetyToIn(boundary_point);
      if (boundary_safety_in <= solid_tolerance) return passed;
    }
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalInsideExitOnSurface},
                   "Inside exit point for normal check should be on the Surface.", dist);
    return passed;
  }

  Vec_t normal(0., 0., 0.);
  bool valid_normal = volume->Normal(boundary_point, normal);
  if (replay) {
    replay->valid_normal          = valid_normal;
    replay->normal                = normal;
    replay->normal_magnitude      = normal.Mag();
    replay->normal_dot_direction  = normal.Dot(direction);
    replay->normal_probe_step     = ShapeNormalProbeStep(solid_tolerance);
    replay->tangential_probe_step = ShapeTangentialProbeStep(solid_tolerance);
  }
  if (!valid_normal) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalInsideExitValid},
                   "Normal at exit point from Inside ray should be valid.", dist);
    return passed;
  }

  const Precision magnitude = normal.Mag();
  if (!vecgeom::test::ApproxEqual<Precision>(magnitude, static_cast<Precision>(1.0))) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalInsideExitUnitLength},
                   "Normal at exit point from Inside ray should have unit length.", magnitude);
  }

  Vec_t normal_unit(normal);
  if (magnitude > 0.) normal_unit /= magnitude;
  auto surface_kind = DetectSurfaceKind(volume, boundary_point, normal_unit, solid_tolerance,
                                        replay ? &replay->tangential_probe_results : nullptr,
                                        replay ? &replay->tangential_probe_details : nullptr);
  if (replay) replay->surface_kind = surface_kind;

  const Vec_t outward_probe = boundary_point + ShapeNormalProbeStep(solid_tolerance) * normal_unit;
  const Vec_t inward_probe  = boundary_point - ShapeNormalProbeStep(solid_tolerance) * normal_unit;
  auto outward_inside       = volume->Inside(outward_probe);
  auto inward_inside        = volume->Inside(inward_probe);
  if (replay) {
    replay->outward_probe_result = outward_inside;
    replay->inward_probe_result  = inward_inside;
  }

  if (surface_kind == ShapeSurfaceKind::kSmooth && normal_unit.Dot(direction) < 0.) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalInsideExitOutward},
                   "Normal at exit point from Inside ray should not oppose the exiting direction.",
                   normal.Dot(direction));
  }

  return passed;
}

template <typename ImplT, typename DistanceToOutCaller>
bool CheckInsideExitNormals(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                            DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
                            std::uint64_t &score)
{
  bool inside_exit_passed = true;
  for (int i = 0; i < samples.max_points_inside; ++i) {
    const int sample_index = samples.offset_inside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateInsideExitNormalSample(
        volume, point, direction, solid_tolerance, call_distance_to_out, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    inside_exit_passed = inside_exit_passed && sample_passed;
  }
  return inside_exit_passed;
}

template <typename ImplT>
bool EvaluateOutsideEntryNormalSample(
    ImplT const *volume, const Vec_t &point, const Vec_t &direction, Precision solid_tolerance,
    const ShapeCheckContext &context,
    const std::function<void(const ShapeCheckContext &, const std::string &, Precision)> &record_failure,
    ShapeNormalRayReplay *replay = nullptr)
{
  bool passed = true;

  if (replay) {
    replay->context   = context;
    replay->point     = point;
    replay->direction = direction;
  }

  auto inside_result = volume->Inside(point);
  if (replay) replay->point_inside_result = inside_result;

  const Vec_t invdir(1. / NonZero(direction.x()), 1. / NonZero(direction.y()), 1. / NonZero(direction.z()));
  Precision dist_bb = volume->GetUnplacedVolume()->ApproachSolid(point, invdir);
  if (replay) {
    replay->approach_distance = dist_bb;
    replay->approach_point    = point;
  }
  if (!(dist_bb < kInfLength)) return passed;

  const Vec_t approach_point = point + dist_bb * direction;
  Precision dist_in          = volume->DistanceToIn(approach_point, direction);
  if (replay) {
    replay->approach_point  = approach_point;
    replay->travel_distance = dist_bb + dist_in;
  }
  if (!(dist_in > 0.) || dist_in >= kInfLength) return passed;

  const Vec_t boundary_point = approach_point + dist_in * direction;
  auto boundary_inside       = volume->Inside(boundary_point);
  if (replay) {
    replay->boundary_point         = boundary_point;
    replay->boundary_inside_result = boundary_inside;
  }
  if (boundary_inside != vecgeom::EnumInside::kSurface) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalOutsideEntryOnSurface},
                   "Outside entry point for normal check should be on the Surface.", dist_in + dist_bb);
    return passed;
  }

  Vec_t normal(0., 0., 0.);
  bool valid_normal = volume->Normal(boundary_point, normal);
  if (replay) {
    replay->valid_normal          = valid_normal;
    replay->normal                = normal;
    replay->normal_magnitude      = normal.Mag();
    replay->normal_dot_direction  = normal.Dot(direction);
    replay->normal_probe_step     = ShapeNormalProbeStep(solid_tolerance);
    replay->tangential_probe_step = ShapeTangentialProbeStep(solid_tolerance);
  }
  if (!valid_normal) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalOutsideEntryValid},
                   "Normal at entry point from Outside ray should be valid.", dist_in + dist_bb);
    return passed;
  }

  const Precision magnitude = normal.Mag();
  if (!vecgeom::test::ApproxEqual<Precision>(magnitude, static_cast<Precision>(1.0))) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalOutsideEntryUnitLength},
                   "Normal at entry point from Outside ray should have unit length.", magnitude);
  }

  Vec_t normal_unit(normal);
  if (magnitude > 0.) normal_unit /= magnitude;
  auto surface_kind = DetectSurfaceKind(volume, boundary_point, normal_unit, solid_tolerance,
                                        replay ? &replay->tangential_probe_results : nullptr,
                                        replay ? &replay->tangential_probe_details : nullptr);
  if (replay) replay->surface_kind = surface_kind;

  const Vec_t outward_probe = boundary_point + ShapeNormalProbeStep(solid_tolerance) * normal_unit;
  const Vec_t inward_probe  = boundary_point - ShapeNormalProbeStep(solid_tolerance) * normal_unit;
  auto outward_inside       = volume->Inside(outward_probe);
  auto inward_inside        = volume->Inside(inward_probe);
  if (replay) {
    replay->outward_probe_result = outward_inside;
    replay->inward_probe_result  = inward_inside;
  }

  if (surface_kind == ShapeSurfaceKind::kSmooth && normal_unit.Dot(direction) > 0.) {
    passed = false;
    record_failure({context.sample_index, context.sample_group, kNormalOutsideEntryInward},
                   "Normal at entry point from Outside ray should oppose the entering direction.",
                   normal.Dot(direction));
  }

  return passed;
}

template <typename ImplT>
bool CheckOutsideEntryNormals(ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance,
                              ShapeContractViolationSink &sink, std::uint64_t &score)
{
  bool outside_entry_passed = true;
  for (int i = 0; i < samples.max_points_outside; ++i) {
    const int sample_index = samples.offset_outside + i;
    const Vec_t &point     = samples.Point(sample_index);
    const Vec_t &direction = samples.Direction(sample_index);
    const auto context     = MakeShapeCheckContext(samples, sample_index);
    bool sample_passed     = EvaluateOutsideEntryNormalSample(
        volume, point, direction, solid_tolerance, context,
        [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
          if (failure_context.convention_bit >= 0) score |= (std::uint64_t(1) << failure_context.convention_bit);
          sink.Record(message, point, direction, distance, failure_context);
        });
    outside_entry_passed = outside_entry_passed && sample_passed;
  }
  return outside_entry_passed;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeSurfaceCheckSummary RunShapeSurfaceChecks(
    ImplT const *volume, const ShapeContractSampleView &samples, Precision solid_tolerance, Precision grazing_tolerance,
    DistanceToOutCaller &&call_distance_to_out, ShapeContractViolationSink &sink,
    const ShapeSurfaceCheckOptions &options = DefaultShapeSurfaceCheckOptions())
{
  ShapeSurfaceCheckSummary summary;
  CheckSurfacePoints(volume, samples, solid_tolerance, grazing_tolerance, call_distance_to_out, sink, summary.score,
                     options);
  const std::uint64_t surface_mask =
      (std::uint64_t(1) << kSurfaceRayNotBothZero) | (std::uint64_t(1) << kSurfaceDistanceToOutFinite);
  const std::uint64_t shallow_mask =
      (std::uint64_t(1) << kSurfaceShallowInward) | (std::uint64_t(1) << kSurfaceShallowOutward);
  summary.surface_passed = (summary.score & surface_mask) == 0;
  summary.grazing_passed = (summary.score & (std::uint64_t(1) << kSurfaceGrazingNotBothZero)) == 0;
  summary.shallow_passed = (summary.score & shallow_mask) == 0;
  return summary;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeDistanceToOutCheckSummary RunShapeDistanceToOutChecks(ImplT const *volume, const ShapeContractSampleView &samples,
                                                           Precision solid_tolerance,
                                                           DistanceToOutCaller &&call_distance_to_out,
                                                           ShapeContractViolationSink &sink)
{
  ShapeDistanceToOutCheckSummary summary;
  summary.inside_points_passed = CheckInsideDistanceToOutSamples(
      volume, samples, solid_tolerance, ShapeExtentDistance(volume), call_distance_to_out, sink, summary.score);
  return summary;
}

template <typename ImplT>
ShapeDistanceToInCheckSummary RunShapeDistanceToInChecks(ImplT const *volume, const ShapeContractSampleView &samples,
                                                         Precision solid_tolerance, ShapeContractViolationSink &sink)
{
  ShapeDistanceToInCheckSummary summary;
  summary.outside_rays_passed = CheckOutsideDistanceToInSamples(volume, samples, solid_tolerance, sink, summary.score,
                                                                summary.evaluated_outside_rays);
  return summary;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeSafetyCheckSummary RunShapeSafetyChecks(ImplT const *volume, const ShapeContractSampleView &samples,
                                             Precision solid_tolerance, DistanceToOutCaller &&call_distance_to_out,
                                             ShapeContractViolationSink &sink)
{
  ShapeSafetyCheckSummary summary;
  summary.inside_points_passed  = CheckInsideSafetySamples(volume, samples, solid_tolerance, call_distance_to_out, sink,
                                                           summary.score, summary.evaluated_inside_points);
  summary.outside_points_passed = CheckOutsideSafetySamples(volume, samples, solid_tolerance, call_distance_to_out,
                                                            sink, summary.score, summary.evaluated_outside_points);
  return summary;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeHitConsistencyCheckSummary RunShapeHitConsistencyChecks(ImplT const *volume,
                                                             const ShapeContractSampleView &samples,
                                                             Precision solid_tolerance,
                                                             DistanceToOutCaller &&call_distance_to_out,
                                                             ShapeContractViolationSink &sink)
{
  ShapeHitConsistencyCheckSummary summary;
  summary.inside_hits_passed  = CheckInsideHitConsistencySamples(volume, samples, solid_tolerance, call_distance_to_out,
                                                                 sink, summary.score, summary.evaluated_inside_points);
  summary.outside_hits_passed = CheckOutsideHitConsistencySamples(
      volume, samples, solid_tolerance, call_distance_to_out, sink, summary.score, summary.evaluated_outside_points);
  return summary;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeSurfaceRayReplay ReplayShapeSurfaceSample(
    ImplT const *volume, const ShapeContractSampleView &samples, int sample_index, Precision solid_tolerance,
    Precision grazing_tolerance, DistanceToOutCaller &&call_distance_to_out,
    const ShapeSurfaceCheckOptions &options = DefaultShapeSurfaceCheckOptions())
{
  ShapeSurfaceRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
    replay.failures.push_back(
        {failure_context, ShapeConventionLabel(failure_context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
    EvaluateSurfacePointSample(volume, point, direction, solid_tolerance, grazing_tolerance, call_distance_to_out,
                               replay.context, record_failure, &replay, options);
    break;
  case ShapeSampleCategory::kInside:
  case ShapeSampleCategory::kOutside:
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeSafetyRayReplay ReplayShapeSafetySample(ImplT const *volume, const ShapeContractSampleView &samples,
                                             int sample_index, Precision solid_tolerance,
                                             DistanceToOutCaller &&call_distance_to_out)
{
  ShapeSafetyRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
    replay.failures.push_back(
        {failure_context, ShapeConventionLabel(failure_context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kInside:
  case ShapeSampleCategory::kOutside:
    EvaluateShapeSafetySample(volume, point, direction, solid_tolerance, call_distance_to_out, replay.context,
                              record_failure, &replay);
    break;
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeHitConsistencyRayReplay ReplayShapeHitConsistencySample(ImplT const *volume,
                                                             const ShapeContractSampleView &samples, int sample_index,
                                                             Precision solid_tolerance,
                                                             DistanceToOutCaller &&call_distance_to_out)
{
  ShapeHitConsistencyRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
    replay.failures.push_back(
        {failure_context, ShapeConventionLabel(failure_context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kInside:
    EvaluateInsideHitConsistencySample(volume, point, direction, solid_tolerance, call_distance_to_out, replay.context,
                                       record_failure, &replay);
    break;
  case ShapeSampleCategory::kOutside: {
    const int target_sample_index = PairedInsideSampleIndex(samples, sample_index);
    if (target_sample_index >= 0) {
      const Vec_t target_point = samples.Point(target_sample_index);
      EvaluateOutsideHitConsistencySample(volume, point, target_point, target_sample_index, solid_tolerance,
                                          call_distance_to_out, replay.context, record_failure, &replay);
    }
    break;
  }
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

template <typename ImplT>
ShapeDistanceToInRayReplay ReplayShapeDistanceToInSample(ImplT const *volume, const ShapeContractSampleView &samples,
                                                         int sample_index, Precision solid_tolerance)
{
  ShapeDistanceToInRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
    replay.failures.push_back(
        {failure_context, ShapeConventionLabel(failure_context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kOutside: {
    const int target_sample_index = PairedInsideSampleIndex(samples, sample_index);
    if (target_sample_index >= 0) {
      replay.target_sample_index = target_sample_index;
      const Vec_t target_point   = samples.Point(target_sample_index);
      EvaluateOutsideDistanceToInSample(volume, point, target_point, solid_tolerance, replay.context, record_failure,
                                        &replay);
    }
    break;
  }
  case ShapeSampleCategory::kInside:
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeDistanceToOutRayReplay ReplayShapeDistanceToOutSample(ImplT const *volume, const ShapeContractSampleView &samples,
                                                           int sample_index, Precision solid_tolerance,
                                                           DistanceToOutCaller &&call_distance_to_out)
{
  ShapeDistanceToOutRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
    replay.failures.push_back(
        {failure_context, ShapeConventionLabel(failure_context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kInside:
    EvaluateInsideDistanceToOutSample(volume, point, direction, solid_tolerance, ShapeExtentDistance(volume),
                                      call_distance_to_out, replay.context, record_failure, &replay);
    break;
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
  case ShapeSampleCategory::kOutside:
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeNormalCheckSummary RunShapeNormalChecks(ImplT const *volume, const ShapeContractSampleView &samples,
                                             Precision solid_tolerance, DistanceToOutCaller &&call_distance_to_out,
                                             ShapeContractViolationSink &sink)
{
  ShapeNormalCheckSummary summary;
  summary.surface_passed = CheckSurfaceNormals(volume, samples, solid_tolerance, sink, summary.score);
  summary.inside_exit_passed =
      CheckInsideExitNormals(volume, samples, solid_tolerance, call_distance_to_out, sink, summary.score);
  summary.outside_entry_passed = CheckOutsideEntryNormals(volume, samples, solid_tolerance, sink, summary.score);
  return summary;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeNormalRayReplay ReplayShapeNormalSample(ImplT const *volume, const ShapeContractSampleView &samples,
                                             int sample_index, Precision solid_tolerance,
                                             DistanceToOutCaller &&call_distance_to_out)
{
  ShapeNormalRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &failure_context, const std::string &message, Precision distance) {
    replay.failures.push_back(
        {failure_context, ShapeConventionLabel(failure_context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kInside:
    EvaluateInsideExitNormalSample(volume, point, direction, solid_tolerance, call_distance_to_out, replay.context,
                                   record_failure, &replay);
    break;
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
    EvaluateSurfaceNormalSample(volume, point, direction, solid_tolerance, replay.context, record_failure, &replay);
    break;
  case ShapeSampleCategory::kOutside:
    EvaluateOutsideEntryNormalSample(volume, point, direction, solid_tolerance, replay.context, record_failure,
                                     &replay);
    break;
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeContractCheckSummary RunShapeConventionChecks(ImplT const *volume, const ShapeContractSampleView &samples,
                                                   Precision solid_tolerance,
                                                   DistanceToOutCaller &&call_distance_to_out,
                                                   ShapeContractViolationSink &sink)
{
  // Run the three legacy convention families independently so callers can
  // assert on coarse-grained status as well as on the detailed bitset.
  ShapeContractCheckSummary summary;
  summary.surface_points_passed =
      CheckSurfaceConventions(volume, samples, solid_tolerance, call_distance_to_out, sink, summary.score);
  summary.inside_points_passed  = CheckInsideConventions(volume, samples, call_distance_to_out, sink, summary.score);
  summary.outside_points_passed = CheckOutsideConventions(volume, samples, call_distance_to_out, sink, summary.score);
  return summary;
}

template <typename ImplT, typename DistanceToOutCaller>
ShapeContractRayReplay ReplayShapeConventionSample(ImplT const *volume, const ShapeContractSampleView &samples,
                                                   int sample_index, Precision solid_tolerance,
                                                   DistanceToOutCaller &&call_distance_to_out)
{
  ShapeContractRayReplay replay;
  const Vec_t point     = samples.Point(sample_index);
  const Vec_t direction = samples.Direction(sample_index);
  replay.context        = MakeShapeCheckContext(samples, sample_index);
  replay.point          = point;
  replay.direction      = direction;

  auto record_failure = [&](const ShapeCheckContext &context, const std::string &message, Precision distance) {
    replay.failures.push_back({context, ShapeConventionLabel(context.convention_bit), message, distance});
  };

  switch (replay.context.sample_group) {
  case ShapeSampleCategory::kInside:
    EvaluateInsideConventionSample(volume, point, direction, call_distance_to_out, replay.context, record_failure,
                                   &replay);
    break;
  case ShapeSampleCategory::kSurface:
  case ShapeSampleCategory::kEdge:
    EvaluateSurfaceConventionSample(volume, point, direction, solid_tolerance, call_distance_to_out, replay.context,
                                    record_failure, &replay);
    break;
  case ShapeSampleCategory::kOutside:
    EvaluateOutsideConventionSample(volume, point, direction, call_distance_to_out, replay.context, record_failure,
                                    &replay);
    break;
  case ShapeSampleCategory::kUnknown:
    break;
  }

  return replay;
}

} // namespace test
} // namespace vecgeom

#endif
