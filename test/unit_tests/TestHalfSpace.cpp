// Purpose: Unit tests for the half-space primitive and finite Boolean clipping.

#undef NDEBUG
#include "VecGeom/base/FpeEnable.h"

#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/BooleanVolume.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeom/volumes/HalfSpace.h"
#include "VecGeom/volumes/MultiUnion.h"
#include "VecGeom/volumes/ScaledShape.h"
#include "VecGeom/volumes/Tube.h"
#include "ApproxEqual.h"
#include "VecGeomTest/VolumePointers.h"

#include <cmath>
#include <stdexcept>

using vecgeom::kInfLength;
using vecgeom::Precision;
using Vec_t = vecgeom::Vector3D<Precision>;

// Bypass AddNode's bbox update for the intentionally unbounded guard test.
class HalfSpaceGuardMultiUnion : public vecgeom::UnplacedMultiUnion {
public:
  void AddGuardNode(vecgeom::VPlacedVolume const *volume) { fMultiUnion.fVolumes.push_back(volume); }
};

int TestHalfSpace()
{
  Vec_t up(0., 0., 1.);
  Vec_t down(0., 0., -1.);
  bool rejectedZeroNormal = false;
  try {
    vecgeom::HalfSpaceStruct<Precision> invalid(Vec_t(0., 0., 0.), Vec_t(0., 0., 0.));
  } catch (std::runtime_error const &) {
    rejectedZeroNormal = true;
  }
  VECGEOM_ASSERT(rejectedZeroNormal);

  vecgeom::UnplacedHalfSpace halfspace(Vec_t(0., 0., 0.), Vec_t(0., 0., 2.));

  VECGEOM_ASSERT(ApproxEqual(halfspace.GetNormal(), up));
  VECGEOM_ASSERT(halfspace.Contains(Vec_t(0., 0., -1.)));
  VECGEOM_ASSERT(halfspace.Contains(Vec_t(0., 0., 0.)));
  VECGEOM_ASSERT(!halfspace.Contains(Vec_t(0., 0., 1.)));

  VECGEOM_ASSERT(halfspace.Inside(Vec_t(0., 0., -1.)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(halfspace.Inside(Vec_t(0., 0., 0.)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(halfspace.Inside(Vec_t(0., 0., 1.)) == vecgeom::EInside::kOutside);

  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.DistanceToOut(Vec_t(0., 0., -2.), up), 2.));
  VECGEOM_ASSERT(halfspace.DistanceToOut(Vec_t(0., 0., -2.), down) == kInfLength);
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.DistanceToOut(Vec_t(0., 0., 0.), up), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.DistanceToOut(Vec_t(0., 0., 1.), up), -1.));

  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.DistanceToIn(Vec_t(0., 0., 2.), down), 2.));
  VECGEOM_ASSERT(halfspace.DistanceToIn(Vec_t(0., 0., 2.), up) == kInfLength);
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.DistanceToIn(Vec_t(0., 0., 0.), down), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.DistanceToIn(Vec_t(0., 0., -1.), down), -1.));

  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.SafetyToOut(Vec_t(0., 0., -3.)), 3.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.SafetyToOut(Vec_t(0., 0., 0.)), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.SafetyToOut(Vec_t(0., 0., 3.)), -1.));

  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.SafetyToIn(Vec_t(0., 0., 3.)), 3.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.SafetyToIn(Vec_t(0., 0., 0.)), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(halfspace.SafetyToIn(Vec_t(0., 0., -3.)), -1.));

  Vec_t normal;
  VECGEOM_ASSERT(halfspace.Normal(Vec_t(1., 0., 0.), normal));
  VECGEOM_ASSERT(ApproxEqual(normal, up));
  VECGEOM_ASSERT(!halfspace.Normal(Vec_t(0., 0., 1.), normal));

  vecgeom::LogicalVolume translatedLogical("translated_halfspace", &halfspace);
  vecgeom::Transformation3D translation(0., 0., 5.);
  auto *translated = translatedLogical.Place(&translation);
  VECGEOM_ASSERT(translated->Contains(Vec_t(0., 0., 4.)));
  VECGEOM_ASSERT(!translated->Contains(Vec_t(0., 0., 6.)));

  vecgeom::UnplacedBox hostBox(10., 10., 10.);
  vecgeom::LogicalVolume hostLogical("host", &hostBox);
  vecgeom::LogicalVolume cutterLogical("cutter", &halfspace);
  auto *host   = hostLogical.Place();
  auto *cutter = cutterLogical.Place();
  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> clipped(vecgeom::kSubtraction, host, cutter);
  vecgeom::LogicalVolume clippedLogical("clipped", &clipped);
  auto *placedClipped = clippedLogical.Place();

  VECGEOM_ASSERT(placedClipped->Contains(Vec_t(0., 0., 1.)));
  VECGEOM_ASSERT(!placedClipped->Contains(Vec_t(0., 0., -1.)));
  VECGEOM_ASSERT(!placedClipped->Contains(Vec_t(11., 0., 1.)));
  VECGEOM_ASSERT(placedClipped->Inside(Vec_t(0., 0., 0.)) == vecgeom::EInside::kSurface);

  VECGEOM_ASSERT(placedClipped->Normal(Vec_t(0., 0., 0.), normal));
  VECGEOM_ASSERT(ApproxEqual(normal, down));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedClipped->DistanceToOut(Vec_t(0., 0., 5.), down), 5.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedClipped->DistanceToOut(Vec_t(0., 0., 5.), up), 5.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedClipped->DistanceToOut(Vec_t(0., 0., 0.), down), 0.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedClipped->DistanceToIn(Vec_t(0., 0., -5.), up), 5.));
  VECGEOM_ASSERT(ApproxEqual<Precision>(placedClipped->DistanceToIn(Vec_t(0., 0., 12.), down), 2.));
#ifdef VECGEOM_GEANT4
  VECGEOM_ASSERT(placedClipped->ConvertToGeant4() == nullptr);
#endif

  vecgeom::Transformation3D shiftedHalfSpaceTransform(0., 0., 2.);
  auto *shiftedCutter = cutterLogical.Place(&shiftedHalfSpaceTransform);
  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> shiftedClipped(vecgeom::kSubtraction, host, shiftedCutter);
  VECGEOM_ASSERT(shiftedClipped.Contains(Vec_t(0., 0., 3.)));
  VECGEOM_ASSERT(!shiftedClipped.Contains(Vec_t(0., 0., 1.)));

  vecgeom::RNG::SeedStream(778899);
  int shiftedPlaneSamples = 0;
  for (int i = 0; i < 200; ++i) {
    Vec_t point = shiftedClipped.SamplePointOnSurface();
    VECGEOM_ASSERT(shiftedClipped.Inside(point) == vecgeom::EInside::kSurface);
    if (std::fabs(point.z() - 2.) < 100. * vecgeom::kTolerance && std::fabs(point.x()) < 10. + vecgeom::kTolerance &&
        std::fabs(point.y()) < 10. + vecgeom::kTolerance) {
      ++shiftedPlaneSamples;
    }
  }
  VECGEOM_ASSERT(shiftedPlaneSamples > 0);

  Vec_t obliquePoint(0., 0., 1.);
  Vec_t obliqueNormal(0., 1., 1.);
  obliqueNormal.Normalize();
  vecgeom::UnplacedHalfSpace obliqueHalfSpace(obliquePoint, Vec_t(0., 1., 1.));
  vecgeom::LogicalVolume obliqueCutterLogical("oblique_cutter", &obliqueHalfSpace);
  auto *obliqueCutter = obliqueCutterLogical.Place();
  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> obliqueClipped(vecgeom::kSubtraction, host, obliqueCutter);
  VECGEOM_ASSERT(obliqueClipped.Contains(Vec_t(0., 1., 1.)));
  VECGEOM_ASSERT(!obliqueClipped.Contains(Vec_t(0., 0., 0.)));
  VECGEOM_ASSERT(
      ApproxEqual<Precision>(obliqueClipped.DistanceToOut(Vec_t(0., 1., 1.), -obliqueNormal), 1. / std::sqrt(2.)));
  VECGEOM_ASSERT(ApproxEqual<Precision>(obliqueClipped.DistanceToIn(obliquePoint - obliqueNormal, obliqueNormal), 1.));
  VECGEOM_ASSERT(obliqueClipped.Normal(obliquePoint, normal));
  VECGEOM_ASSERT(ApproxEqual(normal, -obliqueNormal));

  vecgeom::RNG::SeedStream(778900);
  int obliquePlaneSamples = 0;
  for (int i = 0; i < 200; ++i) {
    Vec_t point = obliqueClipped.SamplePointOnSurface();
    VECGEOM_ASSERT(obliqueClipped.Inside(point) == vecgeom::EInside::kSurface);
    if (std::fabs((point - obliquePoint).Dot(obliqueNormal)) < 100. * vecgeom::kTolerance) {
      ++obliquePlaneSamples;
    }
  }
  VECGEOM_ASSERT(obliquePlaneSamples > 0);

  bool constructedUnion = true;
  try {
    vecgeom::UnplacedBooleanVolume<vecgeom::kUnion> validIntermediate(vecgeom::kUnion, host, cutter);
  } catch (std::runtime_error const &) {
    constructedUnion = false;
  }
  VECGEOM_ASSERT(constructedUnion);

  vecgeom::UnplacedBox invalidWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume invalidWorldLogical("invalid_world", &invalidWorldBox);
  bool rejectedPositiveHalfSpaceGeometry = false;
  try {
    vecgeom::UnplacedBooleanVolume<vecgeom::kUnion> invalidPlacedUnion(vecgeom::kUnion, host, cutter);
    vecgeom::LogicalVolume invalidPlacedUnionLogical("invalid_union", &invalidPlacedUnion);
    invalidWorldLogical.PlaceDaughter(&invalidPlacedUnionLogical, &vecgeom::Transformation3D::kIdentity);
  } catch (std::runtime_error const &) {
    rejectedPositiveHalfSpaceGeometry = true;
  }
  VECGEOM_ASSERT(rejectedPositiveHalfSpaceGeometry);

  vecgeom::UnplacedBox nestedHostBox(10., 10., 10.);
  vecgeom::GenericUnplacedTube tube(0., 3., 10., 0., vecgeom::kTwoPi);
  vecgeom::UnplacedHalfSpace nestedHalfSpace(Vec_t(0., 0., 0.), up);
  vecgeom::LogicalVolume nestedHostLogical("nested_host", &nestedHostBox);
  vecgeom::LogicalVolume tubeLogical("tube", &tube);
  vecgeom::LogicalVolume nestedHalfSpaceLogical("nested_halfspace", &nestedHalfSpace);
  auto *nestedHost      = nestedHostLogical.Place();
  auto *placedTube      = tubeLogical.Place();
  auto *placedHalfSpace = nestedHalfSpaceLogical.Place();
  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> tubeCut(vecgeom::kSubtraction, placedTube, placedHalfSpace);
  vecgeom::LogicalVolume tubeCutLogical("tube_cut", &tubeCut);
  auto *placedTubeCut = tubeCutLogical.Place();

  vecgeom::RNG::SeedStream(112233);
  int tubeCutPlaneSamples = 0;
  int tubeCutSideSamples  = 0;
  for (int i = 0; i < 500; ++i) {
    Vec_t point = tubeCut.SamplePointOnSurface();
    VECGEOM_ASSERT(tubeCut.Inside(point) == vecgeom::EInside::kSurface);
    const Precision r2 = point.x() * point.x() + point.y() * point.y();
    if (std::fabs(point.z()) < 100. * vecgeom::kTolerance && r2 < 9.) ++tubeCutPlaneSamples;
    if (std::fabs(r2 - 9.) < 1.e-6 && point.z() > 0.) ++tubeCutSideSamples;
  }
  VECGEOM_ASSERT(tubeCutPlaneSamples > 0);
  VECGEOM_ASSERT(tubeCutSideSamples > 0);

  vecgeom::UnplacedBox tubeCutWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume tubeCutWorldLogical("tube_cut_world", &tubeCutWorldBox);
  auto *placedTubeCutDaughter =
      tubeCutWorldLogical.PlaceDaughter(&tubeCutLogical, &vecgeom::Transformation3D::kIdentity);
  VECGEOM_ASSERT(placedTubeCutDaughter->Contains(Vec_t(0., 0., 1.)));
  VECGEOM_ASSERT(!placedTubeCutDaughter->Contains(Vec_t(0., 0., -1.)));
  VECGEOM_ASSERT(!placedTubeCutDaughter->Contains(Vec_t(4., 0., 1.)));

  vecgeom::UnplacedBooleanVolume<vecgeom::kUnion> validFiniteUnion(vecgeom::kUnion, placedTubeCut, nestedHost);
  vecgeom::LogicalVolume validFiniteUnionLogical("valid_finite_union", &validFiniteUnion);
  vecgeom::UnplacedBox validFiniteUnionWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume validFiniteUnionWorldLogical("valid_finite_union_world", &validFiniteUnionWorldBox);
  validFiniteUnionWorldLogical.PlaceDaughter(&validFiniteUnionLogical, &vecgeom::Transformation3D::kIdentity);

  vecgeom::UnplacedBox invalidUnboundedWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume invalidUnboundedWorldLogical("invalid_unbounded_world", &invalidUnboundedWorldBox);
  bool rejectedUnboundedHalfSpaceSubtraction = false;
  try {
    vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> invalidUnboundedSubtraction(vecgeom::kSubtraction,
                                                                                      placedHalfSpace, placedTube);
    vecgeom::LogicalVolume invalidUnboundedSubtractionLogical("invalid_unbounded_subtraction",
                                                              &invalidUnboundedSubtraction);
    invalidUnboundedWorldLogical.PlaceDaughter(&invalidUnboundedSubtractionLogical,
                                               &vecgeom::Transformation3D::kIdentity);
  } catch (std::runtime_error const &) {
    rejectedUnboundedHalfSpaceSubtraction = true;
  }
  VECGEOM_ASSERT(rejectedUnboundedHalfSpaceSubtraction);

  vecgeom::UnplacedScaledShape scaledHalfSpace(&nestedHalfSpace, 1., 1., 1.);
  vecgeom::LogicalVolume scaledHalfSpaceLogical("scaled_halfspace", &scaledHalfSpace);
  vecgeom::UnplacedBox scaledHalfSpaceWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume scaledHalfSpaceWorldLogical("scaled_halfspace_world", &scaledHalfSpaceWorldBox);
  bool rejectedScaledHalfSpaceGeometry = false;
  try {
    scaledHalfSpaceWorldLogical.PlaceDaughter(&scaledHalfSpaceLogical, &vecgeom::Transformation3D::kIdentity);
  } catch (std::runtime_error const &) {
    rejectedScaledHalfSpaceGeometry = true;
  }
  VECGEOM_ASSERT(rejectedScaledHalfSpaceGeometry);

  vecgeom::UnplacedScaledShape scaledTubeCut(&tubeCut, 1., 1., 1.);
  vecgeom::LogicalVolume scaledTubeCutLogical("scaled_tube_cut", &scaledTubeCut);
  vecgeom::UnplacedBox scaledTubeCutWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume scaledTubeCutWorldLogical("scaled_tube_cut_world", &scaledTubeCutWorldBox);
  scaledTubeCutWorldLogical.PlaceDaughter(&scaledTubeCutLogical, &vecgeom::Transformation3D::kIdentity);

  HalfSpaceGuardMultiUnion halfSpaceMultiUnion;
  halfSpaceMultiUnion.AddGuardNode(placedHalfSpace);
  vecgeom::LogicalVolume halfSpaceMultiUnionLogical("halfspace_multiunion", &halfSpaceMultiUnion);
  vecgeom::UnplacedBox halfSpaceMultiUnionWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume halfSpaceMultiUnionWorldLogical("halfspace_multiunion_world", &halfSpaceMultiUnionWorldBox);
  bool rejectedHalfSpaceMultiUnionGeometry = false;
  try {
    halfSpaceMultiUnionWorldLogical.PlaceDaughter(&halfSpaceMultiUnionLogical, &vecgeom::Transformation3D::kIdentity);
  } catch (std::runtime_error const &) {
    rejectedHalfSpaceMultiUnionGeometry = true;
  }
  VECGEOM_ASSERT(rejectedHalfSpaceMultiUnionGeometry);

  vecgeom::UnplacedMultiUnion finiteMultiUnion;
  finiteMultiUnion.AddNode(placedTubeCut);
  finiteMultiUnion.AddNode(nestedHost);
  finiteMultiUnion.Close();
  vecgeom::LogicalVolume finiteMultiUnionLogical("finite_multiunion", &finiteMultiUnion);
  vecgeom::UnplacedBox finiteMultiUnionWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume finiteMultiUnionWorldLogical("finite_multiunion_world", &finiteMultiUnionWorldBox);
  auto const *placedFiniteMultiUnion =
      finiteMultiUnionWorldLogical.PlaceDaughter(&finiteMultiUnionLogical, &vecgeom::Transformation3D::kIdentity);
  (void)placedFiniteMultiUnion;
#ifdef VECGEOM_GEANT4
  VECGEOM_ASSERT(placedFiniteMultiUnion->ConvertToGeant4() == nullptr);
#endif

#ifdef VECGEOM_ROOT
  vecgeom::UnplacedScaledShape scaledFiniteUnion(&validFiniteUnion, 1., 1., 1.);
  vecgeom::LogicalVolume scaledFiniteUnionLogical("scaled_finite_union", &scaledFiniteUnion);
  auto *placedScaledFiniteUnion = scaledFiniteUnionLogical.Place();
  VECGEOM_ASSERT(vecgeom::BooleanHelper::HasRootUnsupportedHalfSpaceUnion(&scaledFiniteUnion));
  vecgeom::VolumePointers scaledFiniteUnionPointers(placedScaledFiniteUnion);
  VECGEOM_ASSERT(scaledFiniteUnionPointers.ROOT() == nullptr);
#endif

  HalfSpaceGuardMultiUnion unboundedCutterMultiUnion;
  unboundedCutterMultiUnion.AddGuardNode(placedTube);
  unboundedCutterMultiUnion.AddGuardNode(placedHalfSpace);
  vecgeom::LogicalVolume unboundedCutterMultiUnionLogical("unbounded_cutter_multiunion", &unboundedCutterMultiUnion);
  auto *placedUnboundedCutterMultiUnion = unboundedCutterMultiUnionLogical.Place();
  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> validMultiUnionCutter(vecgeom::kSubtraction, nestedHost,
                                                                              placedUnboundedCutterMultiUnion);
  vecgeom::LogicalVolume validMultiUnionCutterLogical("valid_multiunion_cutter", &validMultiUnionCutter);
  vecgeom::UnplacedBox validMultiUnionCutterWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume validMultiUnionCutterWorldLogical("valid_multiunion_cutter_world",
                                                           &validMultiUnionCutterWorldBox);
  validMultiUnionCutterWorldLogical.PlaceDaughter(&validMultiUnionCutterLogical, &vecgeom::Transformation3D::kIdentity);

  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> validParityResult(vecgeom::kSubtraction, nestedHost,
                                                                          placedTubeCut);
  vecgeom::LogicalVolume validParityLogical("valid_parity", &validParityResult);
  auto *placedValidParity = validParityLogical.Place();
  vecgeom::UnplacedBox validParityWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume validParityWorldLogical("valid_parity_world", &validParityWorldBox);
  validParityWorldLogical.PlaceDaughter(&validParityLogical, &vecgeom::Transformation3D::kIdentity);
  VECGEOM_ASSERT(placedValidParity->Contains(Vec_t(4., 0., -1.)));
  VECGEOM_ASSERT(placedValidParity->Contains(Vec_t(4., 0., 1.)));
  VECGEOM_ASSERT(!placedValidParity->Contains(Vec_t(0., 0., 1.)));

  vecgeom::UnplacedBox finiteMinusUnboundedUnionWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume finiteMinusUnboundedUnionWorldLogical("finite_minus_unbounded_union_world",
                                                               &finiteMinusUnboundedUnionWorldBox);
  bool rejectedFiniteMinusUnboundedUnion = false;
  try {
    vecgeom::UnplacedBooleanVolume<vecgeom::kUnion> finitePlusInfinite(vecgeom::kUnion, placedTube, placedHalfSpace);
    vecgeom::LogicalVolume finitePlusInfiniteLogical("finite_plus_infinite", &finitePlusInfinite);
    auto *placedFinitePlusInfinite = finitePlusInfiniteLogical.Place();
    vecgeom::UnplacedBooleanVolume<vecgeom::kUnion> unboundedUnionCutter(vecgeom::kUnion, placedTubeCut,
                                                                         placedFinitePlusInfinite);
    vecgeom::LogicalVolume unboundedUnionCutterLogical("unbounded_union_cutter", &unboundedUnionCutter);
    auto *placedUnboundedUnionCutter = unboundedUnionCutterLogical.Place();
    vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> finiteMinusUnboundedUnion(vecgeom::kSubtraction, nestedHost,
                                                                                    placedUnboundedUnionCutter);
    vecgeom::LogicalVolume finiteMinusUnboundedUnionLogical("finite_minus_unbounded_union", &finiteMinusUnboundedUnion);
    finiteMinusUnboundedUnionWorldLogical.PlaceDaughter(&finiteMinusUnboundedUnionLogical,
                                                        &vecgeom::Transformation3D::kIdentity);
  } catch (std::runtime_error const &) {
    rejectedFiniteMinusUnboundedUnion = true;
  }
  VECGEOM_ASSERT(!rejectedFiniteMinusUnboundedUnion);

  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> hostCut(vecgeom::kSubtraction, nestedHost, placedHalfSpace);
  vecgeom::LogicalVolume hostCutLogical("host_cut", &hostCut);
  auto *placedHostCut = hostCutLogical.Place();
  vecgeom::UnplacedBooleanVolume<vecgeom::kSubtraction> finiteResult(vecgeom::kSubtraction, placedHostCut, placedTube);
  vecgeom::LogicalVolume finiteResultLogical("finite_result", &finiteResult);
  auto *placedFiniteResult = finiteResultLogical.Place();
#ifdef VECGEOM_GEANT4
  VECGEOM_ASSERT(placedFiniteResult->ConvertToGeant4() == nullptr);
#endif

  vecgeom::UnplacedBox acceptedParityWorldBox(20., 20., 20.);
  vecgeom::LogicalVolume acceptedParityWorldLogical("accepted_parity_world", &acceptedParityWorldBox);
  acceptedParityWorldLogical.PlaceDaughter(&finiteResultLogical, &vecgeom::Transformation3D::kIdentity);

  VECGEOM_ASSERT(placedFiniteResult->Contains(Vec_t(4., 0., 1.)));
  VECGEOM_ASSERT(!placedFiniteResult->Contains(Vec_t(0., 0., 1.)));
  VECGEOM_ASSERT(!placedFiniteResult->Contains(Vec_t(4., 0., -1.)));
  VECGEOM_ASSERT(!placedFiniteResult->Contains(Vec_t(11., 0., 1.)));
  VECGEOM_ASSERT(placedFiniteResult->Inside(Vec_t(4., 0., 0.)) == vecgeom::EInside::kSurface);

  Vec_t lower, upper;
  finiteResult.Extent(lower, upper);
  VECGEOM_ASSERT(ApproxEqual(lower, Vec_t(-10., -10., -10.)));
  VECGEOM_ASSERT(ApproxEqual(upper, Vec_t(10., 10., 10.)));

  vecgeom::RNG::SeedStream(12345);
  const Precision expectedCapacity = 4000. - vecgeom::kPi * 3. * 3. * 10.;
  const Precision capacity         = finiteResult.Capacity();
  VECGEOM_ASSERT(std::isfinite(capacity));
  VECGEOM_ASSERT(std::fabs(capacity - expectedCapacity) / expectedCapacity < 0.05);

  const Precision surfaceArea = finiteResult.EstimateSurfaceArea(50000);
  VECGEOM_ASSERT(std::isfinite(surfaceArea));
  VECGEOM_ASSERT(surfaceArea > 0.);

  int acceptedPlaneSamples    = 0;
  int acceptedTubeSideSamples = 0;
  for (int i = 0; i < 1000; ++i) {
    Vec_t point = finiteResult.SamplePointOnSurface();
    VECGEOM_ASSERT(finiteResult.Inside(point) == vecgeom::EInside::kSurface);
    const Precision r2 = point.x() * point.x() + point.y() * point.y();
    if (std::fabs(point.z()) < 100. * vecgeom::kTolerance && r2 > 9.) ++acceptedPlaneSamples;
    if (std::fabs(r2 - 9.) < 1.e-6 && point.z() > 0.) ++acceptedTubeSideSamples;
  }
  VECGEOM_ASSERT(acceptedPlaneSamples > 0);
  VECGEOM_ASSERT(acceptedTubeSideSamples > 0);

  return 0;
}

int main(int, char **)
{
  TestHalfSpace();
  return 0;
}
