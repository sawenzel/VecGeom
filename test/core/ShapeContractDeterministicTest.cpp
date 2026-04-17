// Purpose: Deterministic helper-level tests for extracted shape contract logic.

#undef NDEBUG

#include "VecGeom/base/FpeEnable.h"

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "VecGeom/base/Assert.h"
#include "VecGeomTest/ApproxEqual.h"
#include "VecGeomTest/ShapeContractChecks.h"
#include "VecGeomTest/ShapeSampleSet.h"
#include "VecGeomTest/TestCaseSphere.h"
#include "VecGeomTest/TestCaseTube.h"

using vecgeom::Precision;
using Vec_t = vecgeom::Vector3D<Precision>;

namespace {

struct ScriptedProfile {
  vecgeom::EnumInside inside_result = vecgeom::EnumInside::kSurface;
  bool valid_normal                 = true;
  Vec_t normal                      = Vec_t(1., 0., 0.);
  Precision distance_to_in          = 0.;
  Precision distance_to_out         = -1.;
  Precision safety_to_in            = 0.;
  Precision safety_to_out           = 0.;
  Precision approach_solid          = 0.;
};

// This scripted stand-in lets the helper tests assert exact convention-bit and
// violation behavior without duplicating the many geometry-specific ray tables
// already maintained in TestBox, TestTube, and related unit tests.
class ScriptedContractVolume {
public:
  class FakeUnplacedVolume {
  public:
    explicit FakeUnplacedVolume(ScriptedContractVolume const &owner) : fOwner(owner) {}

    Precision ApproachSolid(const Vec_t &point, const Vec_t &) const { return fOwner.ProfileFor(point).approach_solid; }

  private:
    ScriptedContractVolume const &fOwner;
  };

  ScriptedContractVolume() : fUnplaced(*this) {}

  void SetProfile(int key, const ScriptedProfile &profile)
  {
    VECGEOM_ASSERT(key >= 0);
    VECGEOM_ASSERT(key < static_cast<int>(fProfiles.size()));
    fProfiles[key] = profile;
  }

  vecgeom::EnumInside Inside(const Vec_t &point) const { return ProfileFor(point).inside_result; }

  bool Normal(const Vec_t &point, Vec_t &normal) const
  {
    auto const &profile = ProfileFor(point);
    normal              = profile.normal;
    return profile.valid_normal;
  }

  Precision DistanceToIn(const Vec_t &point, const Vec_t &) const { return ProfileFor(point).distance_to_in; }

  Precision SafetyToIn(const Vec_t &point) const { return ProfileFor(point).safety_to_in; }

  Precision SafetyToOut(const Vec_t &point) const { return ProfileFor(point).safety_to_out; }

  Precision DistanceToOut(const Vec_t &point, const Vec_t &, Vec_t &normal) const
  {
    auto const &profile = ProfileFor(point);
    normal              = profile.normal;
    return profile.distance_to_out;
  }

  void Extent(Vec_t &min, Vec_t &max) const
  {
    min = Vec_t(-2., -2., -2.);
    max = Vec_t(2., 2., 2.);
  }

  FakeUnplacedVolume const *GetUnplacedVolume() const { return &fUnplaced; }

private:
  ScriptedProfile const &ProfileFor(const Vec_t &point) const
  {
    const int key = static_cast<int>(point.x());
    VECGEOM_ASSERT(key >= 0);
    VECGEOM_ASSERT(key < static_cast<int>(fProfiles.size()));
    return fProfiles[key];
  }

  std::array<ScriptedProfile, 8> fProfiles;
  FakeUnplacedVolume fUnplaced;
};

class OutsideBoundingBoxMissVolume {
public:
  class FakeUnplacedVolume {
  public:
    explicit FakeUnplacedVolume(OutsideBoundingBoxMissVolume const &) {}

    Precision ApproachSolid(const Vec_t &, const Vec_t &) const { return vecgeom::kInfLength; }
  };

  OutsideBoundingBoxMissVolume() : fUnplaced(*this) {}

  vecgeom::EnumInside Inside(const Vec_t &) const { return vecgeom::EnumInside::kOutside; }

  bool Normal(const Vec_t &, Vec_t &) const { return false; }

  Precision DistanceToIn(const Vec_t &point, const Vec_t &) const
  {
    ++fDistanceToInCalls;
    VECGEOM_ASSERT(std::isfinite(point.x()));
    VECGEOM_ASSERT(std::isfinite(point.y()));
    VECGEOM_ASSERT(std::isfinite(point.z()));
    return vecgeom::kInfLength;
  }

  Precision SafetyToIn(const Vec_t &) const { return 1.; }

  Precision SafetyToOut(const Vec_t &) const { return -1.; }

  Precision DistanceToOut(const Vec_t &, const Vec_t &, Vec_t &) const { return -1.; }

  void Extent(Vec_t &min, Vec_t &max) const
  {
    min = Vec_t(-2., -2., -2.);
    max = Vec_t(2., 2., 2.);
  }

  FakeUnplacedVolume const *GetUnplacedVolume() const { return &fUnplaced; }

  int DistanceToInCalls() const { return fDistanceToInCalls; }

private:
  mutable int fDistanceToInCalls = 0;
  FakeUnplacedVolume fUnplaced;
};

class GrazingSurfaceVolume {
public:
  explicit GrazingSurfaceVolume(Precision zero_band = static_cast<Precision>(vecgeom::kTolerance))
      : fZeroBand(zero_band)
  {
  }

  vecgeom::EnumInside Inside(const Vec_t &) const { return vecgeom::EnumInside::kSurface; }

  bool Normal(const Vec_t &, Vec_t &normal) const
  {
    normal = Vec_t(1., 0., 0.);
    return true;
  }

  Precision DistanceToIn(const Vec_t &, const Vec_t &direction) const
  {
    return std::fabs(direction.x()) <= fZeroBand ? 0. : 1.;
  }

  Precision SafetyToIn(const Vec_t &) const { return 0.; }

  Precision SafetyToOut(const Vec_t &) const { return 0.; }

  Precision DistanceToOut(const Vec_t &, const Vec_t &direction, Vec_t &normal) const
  {
    normal = Vec_t(1., 0., 0.);
    return std::fabs(direction.x()) <= fZeroBand ? 0. : 2.;
  }

  void Extent(Vec_t &min, Vec_t &max) const
  {
    min = Vec_t(-2., -2., -2.);
    max = Vec_t(2., 2., 2.);
  }

private:
  Precision fZeroBand = static_cast<Precision>(vecgeom::kTolerance);
};

auto MakeDistanceToOutCaller()
{
  return [](ScriptedContractVolume const *volume, const Vec_t &point, const Vec_t &direction, Vec_t &normal) {
    return volume->DistanceToOut(point, direction, normal);
  };
}

auto MakeBoundingBoxMissDistanceToOutCaller()
{
  return [](OutsideBoundingBoxMissVolume const *volume, const Vec_t &point, const Vec_t &direction, Vec_t &normal) {
    return volume->DistanceToOut(point, direction, normal);
  };
}

auto MakeGrazingSurfaceDistanceToOutCaller()
{
  return [](GrazingSurfaceVolume const *volume, const Vec_t &point, const Vec_t &direction, Vec_t &normal) {
    return volume->DistanceToOut(point, direction, normal);
  };
}

vecgeom::test::ShapeSampleSet MakeManualSamples(const std::vector<std::pair<Vec_t, Vec_t>> &inside_samples,
                                                const std::vector<std::pair<Vec_t, Vec_t>> &surface_samples,
                                                const std::vector<std::pair<Vec_t, Vec_t>> &outside_samples)
{
  vecgeom::test::ShapeSampleSet samples;
  samples.max_points_inside  = static_cast<int>(inside_samples.size());
  samples.max_points_surface = static_cast<int>(surface_samples.size());
  samples.max_points_edge    = 0;
  samples.max_points_outside = static_cast<int>(outside_samples.size());

  samples.offset_inside  = 0;
  samples.offset_surface = samples.max_points_inside;
  samples.offset_edge    = samples.offset_surface + samples.max_points_surface;
  samples.offset_outside = samples.offset_edge + samples.max_points_edge;

  samples.points.reserve(samples.max_points_inside + samples.max_points_surface + samples.max_points_outside);
  samples.directions.reserve(samples.points.capacity());

  auto append_samples = [&samples](const std::vector<std::pair<Vec_t, Vec_t>> &entries) {
    for (auto const &entry : entries) {
      samples.points.push_back(entry.first);
      samples.directions.push_back(entry.second);
    }
  };

  append_samples(inside_samples);
  append_samples(surface_samples);
  append_samples(outside_samples);
  return samples;
}

const vecgeom::test::ShapeViolation *FindViolation(const vecgeom::test::ShapeCheckResult &result, const char *message)
{
  for (auto const &violation : result.Violations()) {
    if (violation.message == message) return &violation;
  }
  return nullptr;
}

void CheckSurfaceDistanceToInEnteringBit()
{
  ScriptedContractVolume volume;
  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.normal          = Vec_t(1., 0., 0.);
  surface_profile.distance_to_in  = 1.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  surface_profile.valid_normal    = true;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({}, {{Vec_t(1., 0., 0.), Vec_t(-1., 0., 0.)}}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckSurfaceConventions(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(),
                                                       sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kSurfaceDistanceToInEntering));
  VECGEOM_ASSERT(result.CountErrors() == 1);
  VECGEOM_ASSERT(result.CountViolationTypes() == 1);

  auto const *violation = FindViolation(
      result, "DistanceToIn for Surface Point entering into the Shape should be 0 within tolerance (VecGeom "
              "convention)");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSurfaceDistanceToInEntering);
  VECGEOM_ASSERT(violation->count == 1);
  VECGEOM_ASSERT(violation->displayed_occurrences.size() == 1);
  VECGEOM_ASSERT(violation->displayed_occurrences.front().context.sample_index == samples.offset_surface);
  VECGEOM_ASSERT(violation->displayed_occurrences.front().context.sample_group ==
                 vecgeom::test::ShapeSampleCategory::kSurface);
  VECGEOM_ASSERT(vecgeom::test::ApproxEqual(violation->displayed_occurrences.front().point, Vec_t(1., 0., 0.)));
  VECGEOM_ASSERT(vecgeom::test::ApproxEqual(violation->displayed_occurrences.front().direction, Vec_t(-1., 0., 0.)));
  VECGEOM_ASSERT(vecgeom::test::ApproxEqual<Precision>(violation->displayed_occurrences.front().distance, 1.));

  auto replay = vecgeom::test::ReplayShapeConventionSample(&volume, view, samples.offset_surface, vecgeom::kTolerance,
                                                           MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.context.sample_index == samples.offset_surface);
  VECGEOM_ASSERT(replay.context.sample_group == vecgeom::test::ShapeSampleCategory::kSurface);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kSurfaceDistanceToInEntering);
  VECGEOM_ASSERT(replay.failures.front().message ==
                 "DistanceToIn for Surface Point entering into the Shape should be 0 within tolerance (VecGeom "
                 "convention)");
}

void CheckInsideDistanceToInBit()
{
  ScriptedContractVolume volume;
  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = 0.;
  inside_profile.distance_to_out = 2.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 1.;
  volume.SetProfile(0, inside_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(0., 1., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckInsideConventions(&volume, view, MakeDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kInsideDistanceToIn));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "DistanceToIn for Inside Point should be Negative (-1.) (Wrong side)");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->count == 1);
}

void CheckOutsideDistanceToInBit()
{
  ScriptedContractVolume volume;
  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 0.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 1.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  auto samples = MakeManualSamples({}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckOutsideConventions(&volume, view, MakeDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kOutsideDistanceToIn));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "DistanceToIn for Outside Point should be > 0.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->count == 1);
}

void CheckAggregatedSummaryAcrossFamilies()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = 0.;
  inside_profile.distance_to_out = 2.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 1.;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.normal          = Vec_t(1., 0., 0.);
  surface_profile.distance_to_in  = 1.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  surface_profile.valid_normal    = true;
  volume.SetProfile(1, surface_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 0.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 1.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(0., 1., 0.)}}, {{Vec_t(1., 0., 0.), Vec_t(-1., 0., 0.)}},
                                   {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeConventionChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(!summary.surface_points_passed);
  VECGEOM_ASSERT(!summary.inside_points_passed);
  VECGEOM_ASSERT(!summary.outside_points_passed);

  const int expected_score = (1 << vecgeom::test::kSurfaceDistanceToInEntering) |
                             (1 << vecgeom::test::kInsideDistanceToIn) | (1 << vecgeom::test::kOutsideDistanceToIn);
  VECGEOM_ASSERT(summary.score == expected_score);
  VECGEOM_ASSERT(result.CountErrors() == 3);
  VECGEOM_ASSERT(result.CountViolationTypes() == 3);
}

void CheckSurfaceRayNotBothZeroBit()
{
  ScriptedContractVolume volume;
  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = false;
  surface_profile.normal          = Vec_t(1., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 0.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({}, {{Vec_t(1., 0., 0.), Vec_t(0., 1., 0.)}}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed =
      vecgeom::test::CheckSurfacePoints(&volume, view, vecgeom::kTolerance, 0., MakeDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kSurfaceRayNotBothZero));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "DistanceToIn and DistanceToOut cannot both be zero for Surface ray.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSurfaceRayNotBothZero);

  auto replay = vecgeom::test::ReplayShapeSurfaceSample(&volume, view, samples.offset_surface, vecgeom::kTolerance, 0.,
                                                        MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(!replay.valid_normal);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kSurfaceRayNotBothZero);
}

void CheckSurfaceGrazingNotBothZeroBit()
{
  GrazingSurfaceVolume volume;

  auto samples = MakeManualSamples({}, {{Vec_t(1., 0., 0.), Vec_t(1., 1., 0.).Unit()}}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckSurfacePoints(&volume, view, vecgeom::kTolerance, 0.,
                                                  MakeGrazingSurfaceDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kSurfaceGrazingNotBothZero));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "DistanceToIn and DistanceToOut cannot both be zero for grazing Surface ray.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSurfaceGrazingNotBothZero);

  auto replay = vecgeom::test::ReplayShapeSurfaceSample(&volume, view, samples.offset_surface, vecgeom::kTolerance, 0.,
                                                        MakeGrazingSurfaceDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.valid_normal);
  VECGEOM_ASSERT(replay.surface_kind == vecgeom::test::ShapeSurfaceKind::kSmooth);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kSurfaceGrazingNotBothZero);
}

void CheckSurfaceGrazingToleranceTiltsRay()
{
  constexpr Precision kZeroBand         = 1.e-5;
  constexpr Precision kGrazingTolerance = 1.e-4;
  GrazingSurfaceVolume volume(kZeroBand);

  auto samples = MakeManualSamples({}, {{Vec_t(1., 0., 0.), Vec_t(1., 1., 0.).Unit()}}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckSurfacePoints(&volume, view, vecgeom::kTolerance, kGrazingTolerance,
                                                  MakeGrazingSurfaceDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(passed);
  VECGEOM_ASSERT(score == 0);
  VECGEOM_ASSERT(result.CountErrors() == 0);

  auto replay = vecgeom::test::ReplayShapeSurfaceSample(&volume, view, samples.offset_surface, vecgeom::kTolerance,
                                                        kGrazingTolerance, MakeGrazingSurfaceDistanceToOutCaller());
  VECGEOM_ASSERT(replay.Passed());
  VECGEOM_ASSERT(replay.grazing_tolerance == kGrazingTolerance);
  VECGEOM_ASSERT(std::fabs(replay.grazing_direction.x()) > kZeroBand);
}

void CheckCurvedSurfaceDetectorReprojectsSphereAndTubeTangentialProbes()
{
  auto check_curved_surface = [](const vecgeom::VPlacedVolume &volume, const Vec_t &surface_point) {
    VECGEOM_ASSERT(volume.Inside(surface_point) == vecgeom::EnumInside::kSurface);

    Vec_t normal(0., 0., 0.);
    VECGEOM_ASSERT(volume.Normal(surface_point, normal));
    VECGEOM_ASSERT(normal.Mag2() > 0.);
    normal.Normalize();

    std::vector<vecgeom::EnumInside> probe_results;
    std::vector<vecgeom::test::ShapeTangentialProbeReplay> probe_details;
    auto kind = vecgeom::test::DetectSurfaceKind(&volume, surface_point, normal, vecgeom::kTolerance, &probe_results,
                                                 &probe_details);

    VECGEOM_ASSERT(kind == vecgeom::test::ShapeSurfaceKind::kSmooth);
    VECGEOM_ASSERT(probe_details.size() == 8);

    for (auto const &probe : probe_details) {
      VECGEOM_ASSERT(probe.probe_inside_result != vecgeom::EnumInside::kInside);
      VECGEOM_ASSERT(probe.reprojection_distance < vecgeom::kInfLength);
      VECGEOM_ASSERT(probe.boundary_inside_result == vecgeom::EnumInside::kSurface);
      VECGEOM_ASSERT(probe.valid_normal);
      VECGEOM_ASSERT(probe.normal_dot_reference > static_cast<Precision>(0.95));
    }
  };

  auto sphere = vecgeom::test::MakeSphereThinShellTestSolid();
  check_curved_surface(*sphere, Vec_t(100., 0., 0.));

  auto tube = vecgeom::test::MakeTubeFullPhiTestSolid();
  check_curved_surface(*tube, Vec_t(10., 0., 0.));
}

void CheckSurfaceNormalValidityBit()
{
  ScriptedContractVolume volume;
  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = false;
  surface_profile.normal          = Vec_t(1., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({}, {{Vec_t(1., 0., 0.), Vec_t(-1., 0., 0.)}}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckSurfaceNormals(&volume, view, vecgeom::kTolerance, sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kNormalSurfaceValid));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "Normal for Surface Point should be valid.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kNormalSurfaceValid);

  auto replay = vecgeom::test::ReplayShapeNormalSample(&volume, view, samples.offset_surface, vecgeom::kTolerance,
                                                       MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(!replay.valid_normal);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kNormalSurfaceValid);
}

void CheckSurfaceNormalUnitLengthBit()
{
  ScriptedContractVolume volume;
  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = true;
  surface_profile.normal          = Vec_t(2., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({}, {{Vec_t(1., 0., 0.), Vec_t(-1., 0., 0.)}}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckSurfaceNormals(&volume, view, vecgeom::kTolerance, sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kNormalSurfaceUnitLength));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "Normal for Surface Point should have unit length.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kNormalSurfaceUnitLength);

  auto replay = vecgeom::test::ReplayShapeNormalSample(&volume, view, samples.offset_surface, vecgeom::kTolerance,
                                                       MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(vecgeom::test::ApproxEqual<Precision>(replay.normal_magnitude, 2.));
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kNormalSurfaceUnitLength);
}

void CheckInsideExitNormalOrientationBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = -1.;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 1.;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = true;
  surface_profile.normal          = Vec_t(-1., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed =
      vecgeom::test::CheckInsideExitNormals(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kNormalInsideExitOutward));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "Normal at exit point from Inside ray should not oppose the exiting direction.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kNormalInsideExitOutward);

  auto replay = vecgeom::test::ReplayShapeNormalSample(&volume, view, samples.offset_inside, vecgeom::kTolerance,
                                                       MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.boundary_inside_result == vecgeom::EnumInside::kSurface);
  VECGEOM_ASSERT(replay.surface_kind == vecgeom::test::ShapeSurfaceKind::kSmooth);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kNormalInsideExitOutward);
}

void CheckOutsideEntryNormalOrientationBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 1.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 1.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = true;
  surface_profile.normal          = Vec_t(-1., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({}, {}, {{Vec_t(2., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed = vecgeom::test::CheckOutsideEntryNormals(&volume, view, vecgeom::kTolerance, sink, score);

  VECGEOM_ASSERT(!passed);
  VECGEOM_ASSERT(score == (1 << vecgeom::test::kNormalOutsideEntryInward));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "Normal at entry point from Outside ray should oppose the entering direction.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kNormalOutsideEntryInward);

  auto replay = vecgeom::test::ReplayShapeNormalSample(&volume, view, samples.offset_outside, vecgeom::kTolerance,
                                                       MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.boundary_inside_result == vecgeom::EnumInside::kSurface);
  VECGEOM_ASSERT(replay.surface_kind == vecgeom::test::ShapeSurfaceKind::kSmooth);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kNormalOutsideEntryInward);
}

void CheckOutsideBoundingBoxMissSkipsInfinitePropagation()
{
  OutsideBoundingBoxMissVolume volume;

  auto samples = MakeManualSamples({}, {}, {{Vec_t(10., 0., 0.), Vec_t(0., 1., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  int score = 0;

  bool passed =
      vecgeom::test::CheckOutsideConventions(&volume, view, MakeBoundingBoxMissDistanceToOutCaller(), sink, score);

  VECGEOM_ASSERT(passed);
  VECGEOM_ASSERT(score == 0);
  VECGEOM_ASSERT(result.CountErrors() == 0);
  VECGEOM_ASSERT(volume.DistanceToInCalls() == 0);

  auto replay = vecgeom::test::ReplayShapeConventionSample(&volume, view, samples.offset_outside, vecgeom::kTolerance,
                                                           MakeBoundingBoxMissDistanceToOutCaller());
  VECGEOM_ASSERT(replay.Passed());
  VECGEOM_ASSERT(replay.approach_distance == vecgeom::kInfLength);
  VECGEOM_ASSERT(replay.distance_to_in == vecgeom::kInfLength);
  VECGEOM_ASSERT(replay.shifted_distance_to_in == vecgeom::kInfLength);
  VECGEOM_ASSERT(vecgeom::test::ApproxEqual(replay.approach_point, Vec_t(10., 0., 0.)));
  VECGEOM_ASSERT(volume.DistanceToInCalls() == 0);
}

void CheckDistanceToOutPositiveBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = -1.;
  inside_profile.distance_to_out = 0.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 0.;
  volume.SetProfile(0, inside_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeDistanceToOutChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (1 << vecgeom::test::kDistanceToOutPositive));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "DistanceToOut for Inside Point should be > 0.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToOutPositive);

  auto replay = vecgeom::test::ReplayShapeDistanceToOutSample(&volume, view, samples.offset_inside, vecgeom::kTolerance,
                                                              MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kDistanceToOutPositive);
}

void CheckDistanceToOutSafetyBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = -1.;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 2.;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = true;
  surface_profile.normal          = Vec_t(1., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeDistanceToOutChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (1 << vecgeom::test::kDistanceToOutAboveSafety));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "DistanceToOut for Inside Point should be >= SafetyToOut(point) within tolerance.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToOutAboveSafety);
}

void CheckDistanceToOutOnSurfaceBitAndReplay()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = -1.;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.valid_normal    = true;
  outside_profile.normal          = Vec_t(1., 0., 0.);
  outside_profile.distance_to_in  = 1.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 1.;
  outside_profile.safety_to_out   = -1.;
  volume.SetProfile(1, outside_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeDistanceToOutChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (1 << vecgeom::test::kDistanceToOutOnSurface));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(
      result, "DistanceToOut for Inside Point overshoots: propagated exit point should be on the Surface.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToOutOnSurface);

  auto replay = vecgeom::test::ReplayShapeDistanceToOutSample(&volume, view, samples.offset_inside, vecgeom::kTolerance,
                                                              MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.boundary_inside_result == vecgeom::EnumInside::kOutside);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kDistanceToOutOnSurface);
}

void CheckDistanceToOutHappyPath()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_in  = -1.;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_in    = -1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result   = vecgeom::EnumInside::kSurface;
  surface_profile.valid_normal    = true;
  surface_profile.normal          = Vec_t(1., 0., 0.);
  surface_profile.distance_to_in  = 0.;
  surface_profile.distance_to_out = 1.;
  surface_profile.safety_to_in    = 0.;
  surface_profile.safety_to_out   = 0.;
  volume.SetProfile(1, surface_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeDistanceToOutChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(summary.Passed());
  VECGEOM_ASSERT(summary.score == 0);
  VECGEOM_ASSERT(result.CountErrors() == 0);

  auto replay = vecgeom::test::ReplayShapeDistanceToOutSample(&volume, view, samples.offset_inside, vecgeom::kTolerance,
                                                              MakeDistanceToOutCaller());
  VECGEOM_ASSERT(replay.Passed());
  VECGEOM_ASSERT(replay.boundary_inside_result == vecgeom::EnumInside::kSurface);
  VECGEOM_ASSERT(replay.valid_normal);
}

void CheckDistanceToInApproachBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 1.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 1.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = vecgeom::kInfLength;
  volume.SetProfile(2, outside_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.evaluated_outside_rays == 1);
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kDistanceToInApproachFinite));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "ApproachSolid for Outside -> Inside ray should be finite.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToInApproachFinite);

  auto replay =
      vecgeom::test::ReplayShapeDistanceToInSample(&volume, view, samples.offset_outside, vecgeom::kTolerance);
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.target_sample_index == samples.offset_inside);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kDistanceToInApproachFinite);
}

void CheckDistanceToInPositiveBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 0.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 0.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kDistanceToInPositive));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "DistanceToIn for Outside -> Inside ray should be > 0.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToInPositive);
}

void CheckDistanceToInFiniteBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = vecgeom::kInfLength;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 0.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kDistanceToInFinite));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "DistanceToIn for Outside -> Inside ray should be finite.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToInFinite);
}

void CheckDistanceToInSafetyBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result = vecgeom::EnumInside::kSurface;
  volume.SetProfile(1, surface_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 1.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 2.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kDistanceToInAboveSafety));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "DistanceToIn for Outside -> Inside ray should be >= SafetyToIn(point) within tolerance.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToInAboveSafety);
}

void CheckDistanceToInWithinTargetBitAndReplay()
{
  ScriptedContractVolume volume;

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 2.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 0.;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  ScriptedProfile inside_target_profile;
  inside_target_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(3, inside_target_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result = vecgeom::EnumInside::kSurface;
  volume.SetProfile(4, surface_profile);

  auto samples =
      MakeManualSamples({{Vec_t(3., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kDistanceToInWithinTarget));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(
      result, "DistanceToIn for Outside -> Inside ray should reach the Surface before the paired Inside target.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToInWithinTarget);

  auto replay =
      vecgeom::test::ReplayShapeDistanceToInSample(&volume, view, samples.offset_outside, vecgeom::kTolerance);
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.target_sample_index == samples.offset_inside);
  VECGEOM_ASSERT(replay.boundary_inside_result == vecgeom::EnumInside::kSurface);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kDistanceToInWithinTarget);
}

void CheckDistanceToInOnSurfaceBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 1.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 0.5;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  ScriptedProfile overshoot_profile;
  overshoot_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(1, overshoot_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kDistanceToInOnSurface));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(
      result, "DistanceToIn for Outside -> Inside ray overshoots: propagated entry point should be on the Surface.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kDistanceToInOnSurface);
}

void CheckDistanceToInHappyPath()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile surface_profile;
  surface_profile.inside_result = vecgeom::EnumInside::kSurface;
  volume.SetProfile(1, surface_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result   = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in  = 1.;
  outside_profile.distance_to_out = -1.;
  outside_profile.safety_to_in    = 0.5;
  outside_profile.safety_to_out   = -1.;
  outside_profile.approach_solid  = 0.;
  volume.SetProfile(2, outside_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(2., 0., 0.), Vec_t(1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary = vecgeom::test::RunShapeDistanceToInChecks(&volume, view, vecgeom::kTolerance, sink);

  VECGEOM_ASSERT(summary.Passed());
  VECGEOM_ASSERT(summary.evaluated_outside_rays == 1);
  VECGEOM_ASSERT(summary.score == 0);
  VECGEOM_ASSERT(result.CountErrors() == 0);

  auto replay =
      vecgeom::test::ReplayShapeDistanceToInSample(&volume, view, samples.offset_outside, vecgeom::kTolerance);
  VECGEOM_ASSERT(replay.Passed());
  VECGEOM_ASSERT(replay.target_sample_index == samples.offset_inside);
  VECGEOM_ASSERT(replay.boundary_inside_result == vecgeom::EnumInside::kSurface);
}

void CheckSafetyToOutPositiveBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.;
  volume.SetProfile(0, inside_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kSafetyToOutPositive));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "SafetyToOut for Inside Point should be > 0.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSafetyToOutPositive);
}

void CheckSafetyToOutDistanceBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 2.;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile sphere_profile;
  sphere_profile.inside_result = vecgeom::EnumInside::kSurface;
  volume.SetProfile(2, sphere_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kSafetyToOutDistanceBound));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(
      result, "SafetyToOut for Inside Point should not exceed DistanceToOut(point, direction) within tolerance.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSafetyToOutDistanceBound);
}

void CheckSafetyToOutSphereBitAndReplay()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 2.;
  inside_profile.safety_to_out   = 1.;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile sphere_profile;
  sphere_profile.inside_result = vecgeom::EnumInside::kOutside;
  volume.SetProfile(1, sphere_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kSafetyToOutSafeSphere));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "Point on the SafetyToOut sphere should stay Inside or on the Surface.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSafetyToOutSafeSphere);

  auto replay = vecgeom::test::ReplayShapeSafetySample(&volume, view, samples.offset_inside, vecgeom::kTolerance,
                                                       MakeDistanceToOutCaller());
  VECGEOM_ASSERT(!replay.Passed());
  VECGEOM_ASSERT(replay.sphere_inside_result == vecgeom::EnumInside::kOutside);
  VECGEOM_ASSERT(replay.failures.size() == 1);
  VECGEOM_ASSERT(replay.failures.front().context.convention_bit == vecgeom::test::kSafetyToOutSafeSphere);
}

void CheckSafetyToInPositiveBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 1.;
  outside_profile.safety_to_in   = 0.;
  volume.SetProfile(4, outside_profile);

  auto samples = MakeManualSamples({}, {}, {{Vec_t(4., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kSafetyToInPositive));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(result, "SafetyToIn for Outside Point should be > 0.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSafetyToInPositive);
}

void CheckSafetyToInDistanceBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 0.5;
  outside_profile.safety_to_in   = 1.;
  volume.SetProfile(4, outside_profile);

  ScriptedProfile sphere_profile;
  sphere_profile.inside_result = vecgeom::EnumInside::kOutside;
  volume.SetProfile(3, sphere_profile);

  auto samples = MakeManualSamples({}, {}, {{Vec_t(4., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kSafetyToInDistanceBound));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation = FindViolation(
      result, "SafetyToIn for Outside Point should not exceed DistanceToIn(point, direction) within tolerance.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSafetyToInDistanceBound);
}

void CheckSafetyToInSphereBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 2.;
  outside_profile.safety_to_in   = 1.;
  volume.SetProfile(4, outside_profile);

  ScriptedProfile sphere_profile;
  sphere_profile.inside_result = vecgeom::EnumInside::kInside;
  volume.SetProfile(3, sphere_profile);

  auto samples = MakeManualSamples({}, {}, {{Vec_t(4., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kSafetyToInSafeSphere));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "Point on the SafetyToIn sphere should stay Outside or on the Surface.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kSafetyToInSafeSphere);
}

void CheckSafetyHappyPath()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 2.;
  inside_profile.safety_to_out   = 1.;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile inside_sphere_profile;
  inside_sphere_profile.inside_result = vecgeom::EnumInside::kSurface;
  volume.SetProfile(1, inside_sphere_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 2.;
  outside_profile.safety_to_in   = 1.;
  volume.SetProfile(4, outside_profile);

  ScriptedProfile outside_sphere_profile;
  outside_sphere_profile.inside_result = vecgeom::EnumInside::kSurface;
  volume.SetProfile(3, outside_sphere_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(4., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeSafetyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(summary.Passed());
  VECGEOM_ASSERT(summary.evaluated_inside_points == 1);
  VECGEOM_ASSERT(summary.evaluated_outside_points == 1);
  VECGEOM_ASSERT(summary.score == 0);
  VECGEOM_ASSERT(result.CountErrors() == 0);

  auto inside_replay = vecgeom::test::ReplayShapeSafetySample(&volume, view, samples.offset_inside, vecgeom::kTolerance,
                                                              MakeDistanceToOutCaller());
  VECGEOM_ASSERT(inside_replay.Passed());
  VECGEOM_ASSERT(inside_replay.sphere_inside_result == vecgeom::EnumInside::kSurface);

  auto outside_replay = vecgeom::test::ReplayShapeSafetySample(&volume, view, samples.offset_outside,
                                                               vecgeom::kTolerance, MakeDistanceToOutCaller());
  VECGEOM_ASSERT(outside_replay.Passed());
  VECGEOM_ASSERT(outside_replay.sphere_inside_result == vecgeom::EnumInside::kSurface);
}

void CheckHitConsistencyInsideExitSurfaceBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile exit_profile;
  exit_profile.inside_result = vecgeom::EnumInside::kOutside;
  volume.SetProfile(1, exit_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeHitConsistencyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kHitInsideExitOnSurface));
  VECGEOM_ASSERT(result.CountErrors() == 1);

  auto const *violation =
      FindViolation(result, "Inside consistency ray overshoots: propagated exit point should be on the Surface.");
  VECGEOM_ASSERT(violation != nullptr);
  VECGEOM_ASSERT(violation->convention_bit == vecgeom::test::kHitInsideExitOnSurface);
}

void CheckHitConsistencyInsideBoundaryBits()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile boundary_profile;
  boundary_profile.inside_result   = vecgeom::EnumInside::kSurface;
  boundary_profile.distance_to_in  = 0.;
  boundary_profile.distance_to_out = 1.;
  boundary_profile.safety_to_in    = 0.2;
  boundary_profile.safety_to_out   = 0.3;
  boundary_profile.valid_normal    = true;
  boundary_profile.normal          = Vec_t(1., 0., 0.);
  volume.SetProfile(1, boundary_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 2);
  auto summary =
      vecgeom::test::RunShapeHitConsistencyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  const std::uint64_t expected = (static_cast<std::uint64_t>(1) << vecgeom::test::kHitInsideExitSafetyToIn) |
                                 (static_cast<std::uint64_t>(1) << vecgeom::test::kHitInsideExitSafetyToOut);
  VECGEOM_ASSERT(summary.score == expected);
  VECGEOM_ASSERT(result.CountErrors() == 2);
}

void CheckHitConsistencyInsideNormalAgreementBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile boundary_profile;
  boundary_profile.inside_result   = vecgeom::EnumInside::kSurface;
  boundary_profile.distance_to_in  = 1.;
  boundary_profile.distance_to_out = 1.;
  boundary_profile.safety_to_in    = 0.;
  boundary_profile.safety_to_out   = 0.;
  boundary_profile.valid_normal    = true;
  boundary_profile.normal          = Vec_t(-1., 0., 0.);
  volume.SetProfile(1, boundary_profile);

  auto samples = MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {});
  auto view    = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeHitConsistencyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kHitInsideExitNormalAgreement));
  VECGEOM_ASSERT(result.CountErrors() == 1);
}

void CheckHitConsistencyOutsideEntrySurfaceBit()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile inside_exit_profile;
  inside_exit_profile.inside_result   = vecgeom::EnumInside::kSurface;
  inside_exit_profile.distance_to_in  = 0.;
  inside_exit_profile.distance_to_out = 1.;
  inside_exit_profile.safety_to_in    = 0.;
  inside_exit_profile.safety_to_out   = 0.;
  inside_exit_profile.valid_normal    = true;
  inside_exit_profile.normal          = Vec_t(1., 0., 0.);
  volume.SetProfile(1, inside_exit_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 4.;
  outside_profile.safety_to_in   = 1.;
  outside_profile.safety_to_out  = -1.;
  volume.SetProfile(7, outside_profile);

  ScriptedProfile boundary_profile;
  boundary_profile.inside_result = vecgeom::EnumInside::kOutside;
  volume.SetProfile(3, boundary_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(7., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeHitConsistencyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  VECGEOM_ASSERT(summary.score == (static_cast<std::uint64_t>(1) << vecgeom::test::kHitOutsideEntryOnSurface));
  VECGEOM_ASSERT(result.CountErrors() == 1);
}

void CheckHitConsistencyOutsideBoundaryBits()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile inside_exit_profile;
  inside_exit_profile.inside_result   = vecgeom::EnumInside::kSurface;
  inside_exit_profile.distance_to_in  = 0.;
  inside_exit_profile.distance_to_out = 1.;
  inside_exit_profile.safety_to_in    = 0.;
  inside_exit_profile.safety_to_out   = 0.;
  inside_exit_profile.valid_normal    = true;
  inside_exit_profile.normal          = Vec_t(1., 0., 0.);
  volume.SetProfile(1, inside_exit_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 4.;
  outside_profile.safety_to_in   = 1.;
  outside_profile.safety_to_out  = -1.;
  volume.SetProfile(7, outside_profile);

  ScriptedProfile boundary_profile;
  boundary_profile.inside_result   = vecgeom::EnumInside::kSurface;
  boundary_profile.distance_to_in  = vecgeom::kInfLength;
  boundary_profile.distance_to_out = static_cast<Precision>(-2.);
  boundary_profile.safety_to_in    = 0.2;
  boundary_profile.safety_to_out   = 0.3;
  boundary_profile.valid_normal    = true;
  boundary_profile.normal          = Vec_t(1., 0., 0.);
  volume.SetProfile(3, boundary_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(7., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 4);
  auto summary =
      vecgeom::test::RunShapeHitConsistencyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(!summary.Passed());
  const std::uint64_t expected = (static_cast<std::uint64_t>(1) << vecgeom::test::kHitOutsideEntrySafetyToIn) |
                                 (static_cast<std::uint64_t>(1) << vecgeom::test::kHitOutsideEntrySafetyToOut) |
                                 (static_cast<std::uint64_t>(1) << vecgeom::test::kHitOutsideEntryDistanceToIn) |
                                 (static_cast<std::uint64_t>(1) << vecgeom::test::kHitOutsideEntryDistanceSign);
  VECGEOM_ASSERT(summary.score == expected);
}

void CheckHitConsistencyHappyPathAndReplay()
{
  ScriptedContractVolume volume;

  ScriptedProfile inside_profile;
  inside_profile.inside_result   = vecgeom::EnumInside::kInside;
  inside_profile.distance_to_out = 1.;
  inside_profile.safety_to_out   = 0.5;
  volume.SetProfile(0, inside_profile);

  ScriptedProfile inside_exit_profile;
  inside_exit_profile.inside_result   = vecgeom::EnumInside::kSurface;
  inside_exit_profile.distance_to_in  = 0.;
  inside_exit_profile.distance_to_out = 1.;
  inside_exit_profile.safety_to_in    = 0.;
  inside_exit_profile.safety_to_out   = 0.;
  inside_exit_profile.valid_normal    = true;
  inside_exit_profile.normal          = Vec_t(1., 0., 0.);
  volume.SetProfile(1, inside_exit_profile);

  ScriptedProfile outside_profile;
  outside_profile.inside_result  = vecgeom::EnumInside::kOutside;
  outside_profile.distance_to_in = 4.;
  outside_profile.safety_to_in   = 1.;
  outside_profile.safety_to_out  = -1.;
  volume.SetProfile(7, outside_profile);

  ScriptedProfile entry_profile;
  entry_profile.inside_result   = vecgeom::EnumInside::kSurface;
  entry_profile.distance_to_in  = 0.;
  entry_profile.distance_to_out = 1.;
  entry_profile.safety_to_in    = 0.;
  entry_profile.safety_to_out   = 0.;
  entry_profile.valid_normal    = true;
  entry_profile.normal          = Vec_t(1., 0., 0.);
  volume.SetProfile(3, entry_profile);

  ScriptedProfile exit_profile;
  exit_profile.inside_result = vecgeom::EnumInside::kSurface;
  exit_profile.safety_to_in  = 0.;
  exit_profile.safety_to_out = 0.;
  volume.SetProfile(2, exit_profile);

  auto samples =
      MakeManualSamples({{Vec_t(0., 0., 0.), Vec_t(1., 0., 0.)}}, {}, {{Vec_t(7., 0., 0.), Vec_t(-1., 0., 0.)}});
  auto view = vecgeom::test::MakeShapeContractSampleView(samples);

  vecgeom::test::ShapeCheckResult result;
  vecgeom::test::ShapeContractViolationSink sink(result, 1);
  auto summary =
      vecgeom::test::RunShapeHitConsistencyChecks(&volume, view, vecgeom::kTolerance, MakeDistanceToOutCaller(), sink);

  VECGEOM_ASSERT(summary.Passed());
  VECGEOM_ASSERT(summary.evaluated_inside_points == 1);
  VECGEOM_ASSERT(summary.evaluated_outside_points == 1);
  VECGEOM_ASSERT(summary.score == 0);
  VECGEOM_ASSERT(result.CountErrors() == 0);

  auto inside_replay = vecgeom::test::ReplayShapeHitConsistencySample(&volume, view, samples.offset_inside,
                                                                      vecgeom::kTolerance, MakeDistanceToOutCaller());
  VECGEOM_ASSERT(inside_replay.Passed());
  VECGEOM_ASSERT(inside_replay.boundary_inside_result == vecgeom::EnumInside::kSurface);

  auto outside_replay = vecgeom::test::ReplayShapeHitConsistencySample(&volume, view, samples.offset_outside,
                                                                       vecgeom::kTolerance, MakeDistanceToOutCaller());
  VECGEOM_ASSERT(outside_replay.Passed());
  VECGEOM_ASSERT(outside_replay.boundary_inside_result == vecgeom::EnumInside::kSurface);
  VECGEOM_ASSERT(outside_replay.exit_inside_result == vecgeom::EnumInside::kSurface);
}

} // namespace

int main()
{
  CheckSurfaceDistanceToInEnteringBit();
  CheckInsideDistanceToInBit();
  CheckOutsideDistanceToInBit();
  CheckAggregatedSummaryAcrossFamilies();
  CheckSurfaceRayNotBothZeroBit();
  CheckSurfaceGrazingNotBothZeroBit();
  CheckSurfaceGrazingToleranceTiltsRay();
  CheckCurvedSurfaceDetectorReprojectsSphereAndTubeTangentialProbes();
  CheckSurfaceNormalValidityBit();
  CheckSurfaceNormalUnitLengthBit();
  CheckInsideExitNormalOrientationBit();
  CheckOutsideEntryNormalOrientationBit();
  CheckOutsideBoundingBoxMissSkipsInfinitePropagation();
  CheckDistanceToOutPositiveBit();
  CheckDistanceToOutSafetyBit();
  CheckDistanceToOutOnSurfaceBitAndReplay();
  CheckDistanceToOutHappyPath();
  CheckDistanceToInApproachBit();
  CheckDistanceToInPositiveBit();
  CheckDistanceToInFiniteBit();
  CheckDistanceToInSafetyBit();
  CheckDistanceToInWithinTargetBitAndReplay();
  CheckDistanceToInOnSurfaceBit();
  CheckDistanceToInHappyPath();
  CheckSafetyToOutPositiveBit();
  CheckSafetyToOutDistanceBit();
  CheckSafetyToOutSphereBitAndReplay();
  CheckSafetyToInPositiveBit();
  CheckSafetyToInDistanceBit();
  CheckSafetyToInSphereBit();
  CheckSafetyHappyPath();
  CheckHitConsistencyInsideExitSurfaceBit();
  CheckHitConsistencyInsideBoundaryBits();
  CheckHitConsistencyInsideNormalAgreementBit();
  CheckHitConsistencyOutsideEntrySurfaceBit();
  CheckHitConsistencyOutsideBoundaryBits();
  CheckHitConsistencyHappyPathAndReplay();
  return 0;
}
