/// @file GenericPolyconeImplementation.h
/// @brief Generic polycone kernel helpers and public implementation entry points.
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_GENERICPOLYCONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_GENERICPOLYCONEIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/GenericPolyconeStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>
#include "VecGeom/volumes/kernel/CoaxialConesImplementation.h"
#include "VecGeom/volumes/kernel/shapetypes/ConeTypes.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct GenericPolyconeImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, GenericPolyconeImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedGenericPolycone;
template <typename T>
struct GenericPolyconeStruct;
class UnplacedGenericPolycone;

/// @brief Kernel implementation for generic polycones.
/// @details A generic polycone is represented as ordered z sections. Each
/// section owns a shifted `CoaxialConesStruct` that describes the radial
/// contours in that z interval. This implementation combines the per-section
/// cone queries to provide point classification, distances, safeties, normals,
/// and extent calculations for the full piecewise-linear `r(z)` solid.
struct GenericPolyconeImplementation {

  using PlacedShape_t    = PlacedGenericPolycone;
  using UnplacedStruct_t = GenericPolyconeStruct<Precision>;
  using UnplacedVolume_t = UnplacedGenericPolycone;

  /// @brief Compute the signed distance to a cone outer side surface.
  /// @details Uses the original outer radii stored in the cone struct rather
  /// than radii widened for degenerate cone handling. The caller passes
  /// precomputed `rho` so strict-surface scans do not repeatedly calculate the
  /// same cylindrical radius.
  /// @param cone Cone section to test.
  /// @param point Section-local query point.
  /// @param rho Cylindrical radius of `point`.
  /// @return Signed distance to the outer side surface.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Precision StrictOuterSurfaceDistance(
      ConeStruct<Precision> const &cone, Vector3D<Precision> const &point, Precision rho)
  {
    if (vecCore::math::Abs(cone._frmax1 - cone._frmax2) < cone.fOuterTolerance) {
      return rho - cone._frmax2;
    }

    const Precision projectedRadius = rho - point.z() * cone.fTanRMax;
    const Precision referenceRadius = cone.fRmax2 - cone.fDz * cone.fTanRMax;
    return (projectedRadius - referenceRadius) / cone.fSecRMax;
  }

  /// @brief Compute the signed distance to a cone inner side surface.
  /// @details Uses the original inner radii stored in the cone struct rather
  /// than radii widened for degenerate cone handling. The result is used only
  /// as a strict surface ownership test, so callers compare its absolute value
  /// with tolerance.
  /// @param cone Cone section to test.
  /// @param point Section-local query point.
  /// @param rho Cylindrical radius of `point`.
  /// @return Signed distance to the inner side surface.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Precision StrictInnerSurfaceDistance(
      ConeStruct<Precision> const &cone, Vector3D<Precision> const &point, Precision rho)
  {
    if (vecCore::math::Abs(cone._frmin1 - cone._frmin2) < cone.fInnerTolerance) {
      return rho - cone._frmin2;
    }

    const Precision projectedRadius = rho - point.z() * cone.fTanRMin;
    const Precision referenceRadius = cone.fRmin2 - cone.fDz * cone.fTanRMin;
    return (projectedRadius - referenceRadius) / cone.fSecRMin;
  }

  /// @brief Test whether a point belongs to a cone z-plane annulus.
  /// @details This is a strict bounded-plane test: the point must be on the
  /// requested z plane and inside the original radial interval of that plane.
  /// It is used to distinguish real endcap ownership from artifacts of widened
  /// degenerate radii.
  /// @param cone Cone section to test.
  /// @param point Section-local query point.
  /// @param upperPlane Selects the upper z plane when true, lower when false.
  /// @param rho Cylindrical radius of `point`.
  /// @return True if the point is on the selected bounded z plane.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsWithinStrictZPlaneBand(ConeStruct<Precision> const &cone,
                                                                                    Vector3D<Precision> const &point,
                                                                                    bool upperPlane, Precision rho)
  {
    const Precision planeZ      = upperPlane ? cone.fDz : -cone.fDz;
    const Precision surfaceRMin = upperPlane ? cone._frmin2 : cone._frmin1;
    const Precision surfaceRMax = upperPlane ? cone._frmax2 : cone._frmax1;
    return vecCore::math::Abs(point.z() - planeZ) < kTolerance && rho > (surfaceRMin - kTolerance) &&
           rho < (surfaceRMax + kTolerance);
  }

  /// @brief Test whether a point is on any strictly owned cone surface.
  /// @details Checks z-plane annuli, inner/outer conical surfaces, and phi
  /// planes using the original cone dimensions. This is deliberately stricter
  /// than the normal cone classification because generic-polycone convention
  /// repair must not treat artificial widened surfaces as real boundaries.
  /// @param cone Cone section to test.
  /// @param point Section-local query point.
  /// @param rho Cylindrical radius of `point`.
  /// @return True if the point lies on a real boundary of `cone`.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnStrictConeSurface(ConeStruct<Precision> const &cone,
                                                                                 Vector3D<Precision> const &point,
                                                                                 Precision rho)
  {
    if (vecCore::math::Abs(point.z()) > (cone.fDz + kTolerance)) {
      return false;
    }

    if (IsWithinStrictZPlaneBand(cone, point, false, rho) || IsWithinStrictZPlaneBand(cone, point, true, rho)) {
      return true;
    }

    if (vecCore::math::Abs(StrictOuterSurfaceDistance(cone, point, rho)) < kTolerance) {
      return true;
    }

    if ((cone._frmin1 > Precision(0.) || cone._frmin2 > Precision(0.)) &&
        vecCore::math::Abs(StrictInnerSurfaceDistance(cone, point, rho)) < kTolerance) {
      return true;
    }

    if (!cone.IsFullPhi() &&
        (ConeUtilities::IsOnStartPhi<Precision>(cone, point) || ConeUtilities::IsOnEndPhi<Precision>(cone, point))) {
      return true;
    }

    return false;
  }

  /// @brief Test whether a point is on any strictly owned cone surface.
  /// @details Convenience overload that computes the cylindrical radius once
  /// for callers that do not already have it.
  /// @param cone Cone section to test.
  /// @param point Section-local query point.
  /// @return True if the point lies on a real boundary of `cone`.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnStrictConeSurface(ConeStruct<Precision> const &cone,
                                                                                 Vector3D<Precision> const &point)
  {
    return IsOnStrictConeSurface(cone, point, point.Perp());
  }

  /// @brief Reconstruct a normal from strictly owned section surfaces.
  /// @details Multiple coaxial cone pieces may meet at the same section-local
  /// point. Normals from all strictly owning cone pieces are accumulated and
  /// normalized so edge/corner points can still produce a stable direction.
  /// @param section Generic polycone section to test.
  /// @param localPoint Section-local query point.
  /// @param normal Output normal when a strict owner is found.
  /// @return True if at least one strict surface contributed a valid normal.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool TryGetStrictSectionSurfaceNormal(
      GenericPolyconeSection const &section, Vector3D<Precision> const &localPoint, Vector3D<Precision> &normal)
  {
    normal.Set(0.);
    bool valid = false;
    for (auto const *cone : section.fCoaxialCones->fConeStructVector) {
      Vector3D<Precision> coneNormal;
      if (cone->Normal(localPoint, coneNormal) && IsOnStrictConeSurface(*cone, localPoint)) {
        normal += coneNormal;
        valid = true;
      }
    }

    if (!valid || normal.Mag2() < kToleranceSquared) {
      normal.Set(0.);
      return false;
    }

    normal.Normalize();
    return true;
  }

  /// @brief Cheaply reject points that cannot be on a section surface.
  /// @details This protects the positive-distance `DistanceToIn` convention
  /// repair from scanning all cone pieces for ordinary outside rays. The first
  /// test uses squared radial bounds and the section z range; only plausible
  /// surface starts pay the square root and strict per-cone surface checks.
  /// @param section Generic polycone section to test.
  /// @param localPoint Section-local query point.
  /// @return True if the point is close enough to require strict surface tests.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsNearSectionSurface(GenericPolyconeSection const &section,
                                                                                Vector3D<Precision> const &localPoint)
  {
    const Precision rawMinRadius = section.fCoaxialCones->fMinR - kConeTolerance;
    const Precision minRadius    = rawMinRadius > Precision(0.) ? rawMinRadius : Precision(0.);
    const Precision maxRadius    = section.fCoaxialCones->fMaxR + kConeTolerance;
    const Precision rho2         = localPoint.Perp2();
    if (vecCore::math::Abs(localPoint.z()) > section.fCoaxialCones->fDz + kTolerance || rho2 < minRadius * minRadius ||
        rho2 > maxRadius * maxRadius) {
      return false;
    }

    const Precision rho = vecCore::math::Sqrt(rho2);
    for (auto const *cone : section.fCoaxialCones->fConeStructVector) {
      if (IsOnStrictConeSurface(*cone, localPoint, rho)) return true;
    }

    return vecCore::math::Abs(rho - section.fCoaxialCones->fMinR) < kConeTolerance ||
           vecCore::math::Abs(rho - section.fCoaxialCones->fMaxR) < kConeTolerance;
  }

  /// @brief Check whether a section z boundary is exposed to the polycone exterior.
  /// @details A section-local end plane is not always a real polycone boundary:
  /// another section can occupy the same global z plane. The boundary is exposed
  /// at the first/last section, when the neighboring section classifies the
  /// point as outside, or when the point is on a radial ring that remains a true
  /// surface after the section hand-off.
  /// @param polycone Generic polycone storage.
  /// @param index Index of `section` in `polycone`.
  /// @param section Section owning `localPoint`.
  /// @param globalPoint Polycone-local query point.
  /// @param localPoint Section-local query point.
  /// @return True if the z boundary is a real exposed surface.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsExposedSectionZBoundary(
      UnplacedStruct_t const &polycone, int index, GenericPolyconeSection const &section,
      Vector3D<Precision> const &globalPoint, Vector3D<Precision> const &localPoint)
  {
    const bool onLowerPlane = vecCore::math::Abs(localPoint.z() + section.fCoaxialCones->fDz) < kTolerance;
    const bool onUpperPlane = vecCore::math::Abs(localPoint.z() - section.fCoaxialCones->fDz) < kTolerance;

    if (!onLowerPlane && !onUpperPlane) {
      return false;
    }

    if (onLowerPlane) {
      if (index == 0) {
        return true;
      }

      GenericPolyconeSection const &lowerSection = polycone.GetSection(index - 1);
      const auto lowerLocalPoint                 = globalPoint - Vector3D<Precision>(0, 0, lowerSection.fShift);
      Inside_t lowerInside                       = EInside::kOutside;
      CoaxialConesImplementation::template Inside<Precision>(*lowerSection.fCoaxialCones, lowerLocalPoint, lowerInside);
      if (lowerInside == EInside::kOutside) {
        return true;
      }

      return CoaxialConesImplementation::template IsOnRing<Precision, true>(*section.fCoaxialCones, localPoint) ||
             CoaxialConesImplementation::template IsOnRing<Precision, false>(*lowerSection.fCoaxialCones,
                                                                             lowerLocalPoint);
    }

    if (index == polycone.GetNSections() - 1) {
      return true;
    }

    GenericPolyconeSection const &upperSection = polycone.GetSection(index + 1);
    const auto upperLocalPoint                 = globalPoint - Vector3D<Precision>(0, 0, upperSection.fShift);
    Inside_t upperInside                       = EInside::kOutside;
    CoaxialConesImplementation::template Inside<Precision>(*upperSection.fCoaxialCones, upperLocalPoint, upperInside);
    if (upperInside == EInside::kOutside) {
      return true;
    }

    return CoaxialConesImplementation::template IsOnRing<Precision, false>(*section.fCoaxialCones, localPoint) ||
           CoaxialConesImplementation::template IsOnRing<Precision, true>(*upperSection.fCoaxialCones, upperLocalPoint);
  }

  /// @brief Check radial ownership of a coaxial-cones z plane using squared radius.
  /// @details This is the cheap equivalent of asking whether a point on a
  /// section transition is covered by a neighbor section. It avoids the full
  /// coaxial-cones classifier and reports radial ring edges separately because
  /// those remain exposed polycone surfaces.
  /// @param coaxialCones Coaxial-cones section helper.
  /// @param localPoint Section-local point on the queried z plane.
  /// @param lowerPlane Select lower z plane when true, upper z plane otherwise.
  /// @param r2 Squared radial coordinate of `localPoint`.
  /// @param onRing Output true if the point is on an inner/outer radial ring.
  /// @return True if the z-plane annulus contains the point.
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsOnZPlaneBand(
      CoaxialConesStruct<Precision> const &coaxialCones, Vector3D<Precision> const &localPoint, bool lowerPlane,
      Precision r2, bool &onRing)
  {
    bool inBand = false;
    onRing      = false;
    for (auto const *cone : coaxialCones.fConeStructVector) {
      const Precision planeZ = lowerPlane ? -cone->fDz : cone->fDz;
      const bool onPlane = vecCore::math::Abs(localPoint.z() - planeZ) < kTolerance;
      if (!onPlane) continue;

      const Precision rmin      = lowerPlane ? cone->_frmin1 : cone->_frmin2;
      const Precision rmax      = lowerPlane ? cone->_frmax1 : cone->_frmax2;
      const Precision minRadius = Max(Precision(0.), rmin - kTolerance);
      const Precision maxRadius = rmax + kTolerance;
      const bool aboveInner     = rmin < kTolerance || r2 > minRadius * minRadius;

      const Precision rminLow  = Max(Precision(0.), rmin - kTolerance);
      const Precision rminHigh = rmin + kTolerance;
      const Precision rmaxLow  = Max(Precision(0.), rmax - kTolerance);
      const Precision rmaxHigh = rmax + kTolerance;
      const bool inRadialBand  = aboveInner && r2 < maxRadius * maxRadius;
      const bool onRadialRing =
          (r2 > rminLow * rminLow && r2 < rminHigh * rminHigh) || (r2 > rmaxLow * rmaxLow && r2 < rmaxHigh * rmaxHigh);
      if (!inRadialBand && !onRadialRing) continue;

      // Shared z-plane ownership is valid only inside the phi wedge; otherwise
      // partial-phi outside points would be treated as material hand-offs.
      if (!cone->fPhiWedge.ContainsWithBoundary(localPoint)) continue;

      if (inRadialBand) inBand = true;
      onRing |= onRadialRing;
    }
    return inBand;
  }

  /// @brief Test whether a section-local z-plane point is an internal hand-off.
  /// @details Section cone helpers report their own z end planes as surfaces.
  /// Generic polycones must ignore that zero when the same global z plane is
  /// covered by a neighboring section and is therefore not exposed.
  VECCORE_ATT_HOST_DEVICE static bool IsNonExposedSectionZBoundary(
      UnplacedStruct_t const &polycone, int index, GenericPolyconeSection const &section,
      Vector3D<Precision> const &globalPoint, Vector3D<Precision> const &localPoint)
  {
    const bool onLowerPlane = vecCore::math::Abs(localPoint.z() + section.fCoaxialCones->fDz) < kTolerance;
    const bool onUpperPlane = vecCore::math::Abs(localPoint.z() - section.fCoaxialCones->fDz) < kTolerance;
    if (!onLowerPlane && !onUpperPlane) return false;

    const int neighborIndex = onLowerPlane ? index - 1 : index + 1;
    if (neighborIndex < 0 || neighborIndex >= polycone.GetNSections()) return false;

    GenericPolyconeSection const &neighborSection = polycone.GetSection(neighborIndex);
    const auto neighborLocalPoint                 = globalPoint - Vector3D<Precision>(0, 0, neighborSection.fShift);
    const Precision r2                            = localPoint.Perp2();

    bool currentRing  = false;
    bool neighborRing = false;
    const bool currentBand = IsOnZPlaneBand(*section.fCoaxialCones, localPoint, onLowerPlane, r2, currentRing);
    const bool neighborBand =
        IsOnZPlaneBand(*neighborSection.fCoaxialCones, neighborLocalPoint, !onLowerPlane, r2, neighborRing);
    return currentBand && neighborBand && !currentRing && !neighborRing;
  }

  /// @brief Compute inside safety to section lateral/phi boundaries only.
  /// @details Used to replace a section-local zero z safety when that z plane
  /// is an internal hand-off, not a real generic-polycone boundary.
  VECCORE_ATT_HOST_DEVICE static bool ComputeSectionSafetyToOutIgnoringZ(
      GenericPolyconeSection const &section, Vector3D<Precision> const &localPoint, Precision &safety)
  {
    safety             = kInfLength;
    bool found         = false;
    bool haveRadius    = false;
    Precision r        = 0.;
    const Precision r2 = localPoint.Perp2();
    for (auto const *cone : section.fCoaxialCones->fConeStructVector) {
      if (vecCore::math::Abs(localPoint.z()) > cone->fDz + kTolerance) continue;

      const Precision outerRadius = vecCore::math::Abs(cone->_frmax1 - cone->_frmax2) < cone->fOuterTolerance
                                        ? cone->_frmax2
                                        : cone->fRmax2 - cone->fDz * cone->fTanRMax + localPoint.z() * cone->fTanRMax;
      const Precision outerLimit  = outerRadius + kConeTolerance * cone->fSecRMax;
      if (r2 > outerLimit * outerLimit) continue;

      if (cone->_frmin1 > Precision(0.) || cone->_frmin2 > Precision(0.)) {
        const Precision innerRadius = vecCore::math::Abs(cone->_frmin1 - cone->_frmin2) < cone->fInnerTolerance
                                          ? cone->_frmin2
                                          : cone->fRmin2 - cone->fDz * cone->fTanRMin + localPoint.z() * cone->fTanRMin;
        const Precision innerLimit  = Max(Precision(0.), innerRadius - kConeTolerance * cone->fSecRMin);
        if (r2 < innerLimit * innerLimit) continue;
      }

      // The squared radial band rejects most non-owning cone pieces. Only the
      // remaining candidates pay the sqrt needed by the exact side safeties.
      if (!haveRadius) {
        r          = vecCore::math::Sqrt(r2);
        haveRadius = true;
      }

      Precision coneSafety = -StrictOuterSurfaceDistance(*cone, localPoint, r);
      if (coneSafety < -kConeTolerance) continue;

      if (cone->_frmin1 > Precision(0.) || cone->_frmin2 > Precision(0.)) {
        const Precision innerSafety = StrictInnerSurfaceDistance(*cone, localPoint, r);
        if (innerSafety < -kConeTolerance) continue;
        coneSafety = Min(coneSafety, innerSafety);
      }

      if (!cone->IsFullPhi()) {
        const Precision phiSafety = cone->fPhiWedge.SafetyToOut(localPoint);
        if (phiSafety < -kTolerance) continue;
        coneSafety = Min(coneSafety, phiSafety);
      }

      safety = Min(safety, Max(Precision(0.), coneSafety));
      found  = true;
    }
    return found;
  }

  /// @brief Test whether a shared z-plane point lies on a radial shell edge.
  /// @details Generic polycones can have radial shell edges at section
  /// transitions. Unlike a fully shared z plane, these edges remain true surface
  /// points and must not be collapsed to inside when both adjacent sections
  /// classify the point as surface.
  /// @param polycone Generic polycone storage.
  /// @param indexLow Lower adjacent section index.
  /// @param indexHigh Upper adjacent section index.
  /// @param localPoint Polycone-local query point.
  /// @return True if the point belongs to a shared radial ring.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static bool IsSharedSectionRing(UnplacedStruct_t const &polycone,
                                                                               int indexLow, int indexHigh,
                                                                               Vector3D<Real_v> const &localPoint)
  {
    GenericPolyconeSection const &sectionLow  = polycone.GetSection(indexLow);
    GenericPolyconeSection const &sectionHigh = polycone.GetSection(indexHigh);
    const auto localLowPoint                  = localPoint - Vector3D<Precision>(0, 0, sectionLow.fShift);
    const auto localHighPoint                 = localPoint - Vector3D<Precision>(0, 0, sectionHigh.fShift);

    return CoaxialConesImplementation::template IsOnRing<Real_v, false>(*sectionLow.fCoaxialCones, localLowPoint) ||
           CoaxialConesImplementation::template IsOnRing<Real_v, true>(*sectionHigh.fCoaxialCones, localHighPoint);
  }

  /// @brief Repair `SafetyToIn` answers for near-surface convention queries.
  /// @details This is intentionally not inlined: for normal outside queries the
  /// current-section safety is not close to zero, so keeping the full
  /// classification and normal-reconstruction block out of the hot caller
  /// avoids register pressure and code-size overhead. The helper handles the
  /// current sign convention for wrong-side inside starts and exact surface
  /// starts, including exposed section z boundaries.
  /// @param polycone Generic polycone storage.
  /// @param index Index of `sec` in `polycone`.
  /// @param sec Section used for the current safety estimate.
  /// @param point Polycone-local query point.
  /// @param safety Input/output safety value.
  /// @return True if the helper changed `safety` to a convention value.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE VECCORE_FORCE_NOINLINE static bool RepairSafetyToInNearSurface(
      UnplacedStruct_t const &polycone, int index, GenericPolyconeSection const &sec, Vector3D<Real_v> const &point,
      Real_v &safety)
  {
    const auto localPoint = point - Vector3D<Precision>(0, 0, sec.fShift);
    Vector3D<Precision> localPointPrecision(localPoint.x(), localPoint.y(), localPoint.z());
    Vector3D<Precision> normal;
    bool completelyInside(false), completelyOutside(false);
    GenericKernelForContainsAndInside<Real_v, true>(polycone, point, completelyInside, completelyOutside);
    if (completelyInside) {
      // SafetyToIn is an outside-state query; strictly inside starts are the
      // wrong side under the current VecGeom sign convention.
      safety = Real_v(-1.);
      return true;
    }
    if (!completelyInside && !completelyOutside) {
      safety = Real_v(0.);
      return true;
    }

    bool surfacePoint = TryGetStrictSectionSurfaceNormal(sec, localPointPrecision, normal);
    if (!surfacePoint) {
      surfacePoint = sec.fCoaxialCones->Normal(localPointPrecision, normal);
    }
    const bool onSectionZBoundary =
        vecCore::math::Abs(localPointPrecision.z() - sec.fCoaxialCones->fDz) < kTolerance ||
        vecCore::math::Abs(localPointPrecision.z() + sec.fCoaxialCones->fDz) < kTolerance;
    if (surfacePoint &&
        (!onSectionZBoundary ||
         IsExposedSectionZBoundary(polycone, index, sec, Vector3D<Precision>(point.x(), point.y(), point.z()),
                                   localPointPrecision))) {
      safety = Real_v(0.);
      return true;
    }

    return false;
  }

  /// @brief Classify a point against one generic-polycone section.
  /// @details The generic polycone is decomposed into z sections, each backed
  /// by a coaxial-cones helper shifted to its own local origin. This helper
  /// applies that shift and forwards to the section-level classifier.
  /// @param unplaced Generic polycone storage.
  /// @param isect Section index, or a negative value for an outside result.
  /// @param polyconePoint Polycone-local query point.
  /// @param secFullyInside Output true when the section owns the point inside.
  /// @param secFullyOutside Output true when the section classifies outside.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForASection(
      UnplacedStruct_t const &unplaced, int isect, Vector3D<Real_v> const &polyconePoint, bool &secFullyInside,
      bool &secFullyOutside)
  {

    using namespace ConeTypes;

    if (isect < 0) {
      secFullyInside  = false;
      secFullyOutside = true;
      return;
    }

    GenericPolyconeSection const &sec = unplaced.GetSection(isect);
    Vector3D<Precision> secLocalp     = polyconePoint - Vector3D<Precision>(0, 0, sec.fShift);
#ifdef POLYCONEDEBUG
    std::cerr << " isect=" << isect << "/" << unplaced.GetNSections() << " secLocalP=" << secLocalp
              << ", secShift=" << sec.fShift << " sec.fSolid=" << sec.fSolid << std::endl;
    if (sec.fSolid) sec.fSolid->Print();
#endif

    CoaxialConesImplementation::template GenericKernelForContainsAndInside<Real_v, ForInside>(
        *sec.fCoaxialCones, secLocalp, secFullyInside, secFullyOutside);
  }

  /// @brief Check whether a point is inside or on the generic polycone.
  /// @param genericPolycone Generic polycone storage.
  /// @param point Polycone-local query point.
  /// @param inside Output true for inside or surface points.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &genericPolycone,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused(false), outside(false);
    GenericKernelForContainsAndInside<Real_v, false>(genericPolycone, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classify a point as inside, outside, or surface.
  /// @param genericPolycone Generic polycone storage.
  /// @param point Polycone-local query point.
  /// @param inside Output `EInside` classification.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &genericPolycone,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {
    bool completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, true>(genericPolycone, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = EInside::kOutside;
    if (completelyinside) inside = EInside::kInside;
  }

  /// @brief Shared point-classification implementation for `Contains` and `Inside`.
  /// @details The z coordinate is probed at `z +/- kTolerance` to handle points
  /// near section transition planes. `Contains` only needs to prove outside,
  /// while `Inside` also needs a positive inside decision; when two neighboring
  /// sections both report surface, shared z planes are treated as inside except
  /// for radial shell rings that remain exposed surfaces.
  /// @param unplaced Generic polycone storage.
  /// @param localPoint Polycone-local query point.
  /// @param completelyInside Output true when the point is conclusively inside.
  /// @param completelyOutside Output true when the point is conclusively outside.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &localPoint, bool &completelyInside,
      bool &completelyOutside)
  {
    int indexLow  = unplaced.GetSectionIndex(localPoint.z() - kTolerance);
    int indexHigh = unplaced.GetSectionIndex(localPoint.z() + kTolerance);
    if (indexLow < 0 && indexHigh < 0) {
      completelyOutside = true;
      return;
    }
    if (indexLow < 0 && indexHigh == 0) {
      // Check location in section 0 and return
      GenericKernelForASection<Real_v, ForInside>(unplaced, 0, localPoint, completelyInside, completelyOutside);
      return;
    }
    if (indexHigh < 0 && indexLow == (unplaced.GetNSections() - 1)) {
      // Check location in section N-1 and return
      GenericKernelForASection<Real_v, ForInside>(unplaced, (unplaced.GetNSections() - 1), localPoint, completelyInside,
                                                  completelyOutside);

      return;
    }
    if (indexLow >= 0 && indexHigh >= 0) {
      if (indexLow == indexHigh) {
        // Check location in section indexLow and return
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexLow, localPoint, completelyInside,
                                                    completelyOutside);

        return;
      } else {

        bool secInLow = false, secOutLow = false;
        bool secInHigh = false, secOutHigh = false;

        GenericKernelForASection<Real_v, ForInside>(unplaced, indexLow, localPoint, secInLow, secOutLow);
        GenericKernelForASection<Real_v, ForInside>(unplaced, indexHigh, localPoint, secInHigh, secOutHigh);
        bool surfLow  = !secInLow && !secOutLow;
        bool surfHigh = !secInHigh && !secOutHigh;

        if (surfLow && surfHigh) {
          // Match PolyconeImplementation for shared z planes, except keep
          // generic-polycone radial shell edges as true surface boundaries.
          completelyInside = !IsSharedSectionRing<Real_v>(unplaced, indexLow, indexHigh, localPoint);
          return;
        } else {
          // else if point is on surface of only one of the two sections then point is actually on surface , the default
          // case,
          // so no need to check

          // What needs to check is if it is outside both ie. Outside indexLow section and Outside indexHigh section
          // then it is certainly outside
          if (secOutLow && secOutHigh) {
            completelyOutside = true;
            return;
          }
        }
      }
    }
  }

  /// @brief Compute distance from an outside point to the generic polycone.
  /// @details The ray is tested against section coaxial-cone helpers in the
  /// direction of increasing or decreasing z. Surface-start convention repair
  /// is intentionally guarded: positive hits are only converted to zero when
  /// the start is plausibly on a real section surface and the ray moves inward
  /// with a meaningful normal component. This avoids turning grazing rays into
  /// simultaneous zero `DistanceToIn`/`DistanceToOut` answers.
  /// @param polycone Generic polycone storage.
  /// @param point Polycone-local ray origin.
  /// @param direction Polycone-local ray direction.
  /// @param stepMax Maximum accepted ray distance.
  /// @param distance Output distance, or `kInfLength` when no entry is found.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &polycone,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const &stepMax, Real_v &distance)
  {
    // using namespace PolyconeTypes;
    Vector3D<Real_v> p = point;
    Vector3D<Real_v> v = direction;

#ifdef POLYCONEDEBUG
    std::cerr << "Polycone::DistToIn() (spot 1): point=" << point << ", dir=" << direction << ", localPoint=" << p
              << ", localDir=" << v << "\n";
#endif

    // TODO: add bounding box check maybe??

    distance      = kInfLength;
    int increment = (v.z() > 0) ? 1 : -1;
    if (vecCore::math::Abs(v.z()) < kTolerance) increment = 0;
    int index = polycone.GetSectionIndex(p.z());
    if (index == -1) index = 0;
    if (index == -2) index = polycone.GetNSections() - 1;

    if (index >= 0 && index < polycone.GetNSections()) {
      GenericPolyconeSection const &sec = polycone.GetSection(index);
      const auto localPoint             = p - Vector3D<Precision>(0, 0, sec.fShift);
      const bool onSectionZBoundary =
          vecCore::math::Abs(localPoint.z() - Real_v(sec.fCoaxialCones->fDz)) < Real_v(kTolerance) ||
          vecCore::math::Abs(localPoint.z() + Real_v(sec.fCoaxialCones->fDz)) < Real_v(kTolerance);
      if (onSectionZBoundary) {
        const Vector3D<Precision> localPointPrecision(localPoint.x(), localPoint.y(), localPoint.z());
        if (IsNonExposedSectionZBoundary(polycone, index, sec, Vector3D<Precision>(p.x(), p.y(), p.z()),
                                         localPointPrecision)) {
          // A start on an internal section hand-off is inside the full generic
          // polycone; DistanceToIn must report the wrong-side convention.
          distance = Real_v(-1.);
          return;
        }
      }
    }

    do {
      // now we have to find a section
      GenericPolyconeSection const &sec = polycone.GetSection(index);
      auto localPoint                   = p - Vector3D<Precision>(0, 0, sec.fShift);

#ifdef POLYCONEDEBUG
      std::cerr << "Polycone::DistToIn() (spot 2):" << " index=" << index << " NSec=" << polycone.GetNSections()
                << " &sec=" << &sec << " - secPars:" << " secOffset=" << sec.fShift << " Dz=" << sec.fSolid->GetDz()
                << " Rmin1=" << sec.fSolid->GetRmin1() << " Rmin2=" << sec.fSolid->GetRmin2()
                << " Rmax1=" << sec.fSolid->GetRmax1() << " Rmax2=" << sec.fSolid->GetRmax2()
                << " -- calling Cone::DistToIn()...\n";
#endif

      CoaxialConesImplementation::template DistanceToIn<Real_v>(*sec.fCoaxialCones, localPoint, v, stepMax, distance);

      if (distance > Real_v(kTolerance) && distance < Real_v(kInfLength)) {
        const Real_v localHitZ = p.z() + distance * v.z() - Real_v(sec.fShift);
        const bool onSectionZBoundary =
            vecCore::math::Abs(localHitZ - Real_v(sec.fCoaxialCones->fDz)) < Real_v(kTolerance) ||
            vecCore::math::Abs(localHitZ + Real_v(sec.fCoaxialCones->fDz)) < Real_v(kTolerance);
        if (onSectionZBoundary) {
          const Vector3D<Real_v> hit = p + distance * v;
          const Vector3D<Precision> localHitPrecision(hit.x(), hit.y(), localHitZ);
          // Positive section hits can land on internal z hand-offs. Skip those;
          // exposed z hits are snapped to the exact section plane so Normal()
          // sees the same boundary the distance query selected.
          if (IsNonExposedSectionZBoundary(polycone, index, sec, Vector3D<Precision>(hit.x(), hit.y(), hit.z()),
                                           localHitPrecision)) {
            distance = kInfLength;
          } else if (vecCore::math::Abs(v.z()) > Real_v(0.)) {
            const Real_v planeZ = Real_v(sec.fShift) +
                                  (localHitPrecision.z() > Precision(0.) ? Real_v(sec.fCoaxialCones->fDz)
                                                                         : -Real_v(sec.fCoaxialCones->fDz));
            distance = (planeZ - p.z()) / v.z();
          }
        }
      }

      bool correctedZero = false;
      if (distance >= Real_v(0.) && distance <= Real_v(kTolerance)) {
        const bool onSectionZBoundary =
            vecCore::math::Abs(localPoint.z() - Real_v(sec.fCoaxialCones->fDz)) < Real_v(kTolerance) ||
            vecCore::math::Abs(localPoint.z() + Real_v(sec.fCoaxialCones->fDz)) < Real_v(kTolerance);
        if (onSectionZBoundary) {
          Vector3D<Precision> localPointPrecision(localPoint.x(), localPoint.y(), localPoint.z());
          if (IsNonExposedSectionZBoundary(polycone, index, sec, Vector3D<Precision>(p.x(), p.y(), p.z()),
                                           localPointPrecision)) {
            // The section cone sees its own endcap, but the generic polycone
            // owns this shared z plane as interior material.
            distance      = Real_v(-1.);
            correctedZero = true;
          }
        }
      }

      if (!correctedZero && distance > Real_v(0.) && distance < Real_v(kInfLength)) {
        if (vecCore::math::Abs(localPoint.z()) < Real_v(sec.fCoaxialCones->fDz + kTolerance)) {
          Vector3D<Precision> localPointPrecision(localPoint.x(), localPoint.y(), localPoint.z());
          if (IsNearSectionSurface(sec, localPointPrecision)) {
            Vector3D<Precision> normal;
            const bool onSectionZBoundary =
                vecCore::math::Abs(localPointPrecision.z() - sec.fCoaxialCones->fDz) < kTolerance ||
                vecCore::math::Abs(localPointPrecision.z() + sec.fCoaxialCones->fDz) < kTolerance;
            const bool strictSurfaceEntry = TryGetStrictSectionSurfaceNormal(sec, localPointPrecision, normal);
            bool surfaceEntry             = strictSurfaceEntry;
            if (!surfaceEntry && !onSectionZBoundary) {
              surfaceEntry = sec.fCoaxialCones->Normal(localPointPrecision, normal);
            }
            const Real_v normalApproach = -v.Dot(normal);
            if (surfaceEntry &&
                // Shallow inward rays from an exact section surface can make
                // the cone kernel report the next crossing instead of the zero
                // entry. Generic coaxial sections can have degenerate
                // endpoints where the strict per-cone ownership test misses
                // the boundary; in that case a valid section normal is still
                // enough to identify the surface. Require a meaningful inward
                // normal displacement to avoid collapsing true grazing rays to
                // zero for both DistanceToIn/Out. Section end planes often
                // include reduced-contour slivers at vertices. Do not use a
                // fallback normal there to turn a tiny positive hit into a
                // zero-distance entry.
                (!onSectionZBoundary || strictSurfaceEntry) &&
                (!onSectionZBoundary ||
                 IsExposedSectionZBoundary(polycone, index, sec, Vector3D<Precision>(p.x(), p.y(), p.z()),
                                           localPointPrecision)) &&
                normalApproach > Real_v(0.) && distance * normalApproach > Real_v(kConeTolerance)) {
              distance = Real_v(0.);
            }
          }
        }
      }

#ifdef POLYCONEDEBUG
      std::cerr << "Polycone::DistToIn() (spot 3):" << " distToIn() = " << distance << "\n";
#endif

      if (distance < kInfLength || !increment) break;
      index += increment;
    } while (index >= 0 && index < polycone.GetNSections());
    return;
  }

  /// @brief Compute distance from an inside point to leave the generic polycone.
  /// @details For multi-section shapes the method walks section by section in
  /// the ray z direction, accumulating local exits and re-entering the next
  /// section at shared z transitions. Boundary starts at the outer z limits are
  /// handled explicitly so the section walk always makes progress.
  /// @param polycone Generic polycone storage.
  /// @param point Polycone-local ray origin.
  /// @param dir Polycone-local ray direction.
  /// @param stepMax Maximum accepted ray distance.
  /// @param distance Output distance, `-1` for wrong-side starts, or accumulated exit distance.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &polycone,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &dir,
                                                                         Real_v const &stepMax, Real_v &distance)
  {
    distance            = kInfLength;
    Vector3D<Real_v> pn = point;
    Precision dist      = 0.;
    int increment       = (dir.z() > 0) ? 1 : -1;
    if (vecCore::math::Abs(dir.z()) < kTolerance) increment = 0;

    // specialization for N==1??? It should be a cone in the first place
    if (polycone.GetNSections() == 1) {
      const GenericPolyconeSection &section = polycone.GetSection(0);

      CoaxialConesImplementation::template DistanceToOut<Real_v>(
          *section.fCoaxialCones, point - Vector3D<Precision>(0, 0, section.fShift), dir, stepMax, distance);

      return;
    }

    int indexLow  = polycone.GetSectionIndex(point.z() - kTolerance);
    int indexHigh = polycone.GetSectionIndex(point.z() + kTolerance);
    int index     = 0;
    if (indexLow < 0 && indexHigh < 0) {
      distance = -1;
      return;
    }

    if (indexLow < 0 && indexHigh >= 0) {
      // At the lower z boundary, a downward ray exits immediately; otherwise
      // use the first real section so the section-walk loop can make progress.
      if (dir.z() < -kTolerance) {
        distance = 0.;
        return;
      }
      indexLow = indexHigh;
    }

    if (indexLow >= 0 && indexHigh < 0) {
      // At the upper z boundary, an upward ray exits immediately; otherwise
      // use the last real section. Without this, the loop body is skipped while
      // indexLow remains valid forever.
      if (dir.z() > kTolerance) {
        distance = 0.;
        return;
      }
      indexHigh = indexLow;
    }

    Inside_t inside;
    Precision totalDistance = 0.;
    int count               = 0;
    do {

      if (indexLow >= 0 && indexHigh >= 0) {
        count++;
        index                                 = indexLow;
        const GenericPolyconeSection &section = polycone.GetSection(index);
        pn.z() -= section.fShift;
        CoaxialConesImplementation::template Inside<Real_v>(*section.fCoaxialCones, pn, inside);
        if (inside == EInside::kOutside) {
          if (count == 1) {
            distance = -1;
            return;
          } else {
            distance = totalDistance;
            return;
          }
        } else {
          CoaxialConesImplementation::template DistanceToOut<Real_v>(*section.fCoaxialCones, pn, dir, stepMax, dist);
          if (totalDistance < kTolerance && dist < kTolerance) {
            Vector3D<Precision> localPointPrecision(pn.x(), pn.y(), pn.z());
            Vector3D<Precision> normal;
            bool surfaceEntry = TryGetStrictSectionSurfaceNormal(section, localPointPrecision, normal);
            if (!surfaceEntry) {
              surfaceEntry = section.fCoaxialCones->Normal(localPointPrecision, normal);
            }
            if (surfaceEntry && dir.Dot(normal) < Real_v(0.)) {
              // Starting exactly on an exposed section boundary and heading
              // inward must not make DistanceToOut report the current surface.
              // Probe inside the same section and add back the local push.
              const Vector3D<Real_v> pushedPoint = pn + dir * Real_v(kConeTolerance);
              Inside_t pushedInside              = EInside::kOutside;
              CoaxialConesImplementation::template Inside<Real_v>(*section.fCoaxialCones, pushedPoint, pushedInside);
              if (pushedInside != EInside::kOutside) {
                Precision pushedDistance = kInfLength;
                CoaxialConesImplementation::template DistanceToOut<Real_v>(*section.fCoaxialCones, pushedPoint, dir,
                                                                           stepMax, pushedDistance);
                if (pushedDistance >= 0. && pushedDistance < kInfLength) {
                  dist = pushedDistance + kConeTolerance;
                }
              }
            }
          }
          if (dist < 0.) break;
          totalDistance += dist;
          pn += dir * dist;
          pn.z() += section.fShift;
        }
        indexLow += increment;
        indexHigh += increment;
      }
    } while (increment != 0 && indexLow > -1 && indexLow < polycone.GetNSections()); // end of do-while
    distance = totalDistance;
    return;
  }

  /// @brief Compute safety from an outside point to the generic polycone.
  /// @details The current section gives the initial safety bound, then
  /// neighboring sections are scanned only while their z-plane distance can
  /// still improve that bound. Near-zero current-section safeties are passed to
  /// a cold convention-repair helper so exact surface and wrong-side starts are
  /// treated consistently without bloating the common outside path.
  /// @param polycone Generic polycone storage.
  /// @param point Polycone-local query point.
  /// @param safety Output safety, `0` on surface, or `-1` for wrong-side inside starts.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &polycone,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {

    Vector3D<Real_v> p = point;
    int index          = polycone.GetSectionIndex(p.z());

    bool needZ = false;
    if (index < 0) {
      needZ = true;
      if (index == -1) index = 0;
      if (index == -2) index = polycone.GetNSections() - 1;
    }
    Precision minSafety               = 0; //= SafetyFromOutsideSection(index, p);
    GenericPolyconeSection const &sec = polycone.GetSection(index);
    // safety to current segment
    if (needZ) {
      CoaxialConesImplementation::template SafetyToIn<Real_v>(*sec.fCoaxialCones,
                                                              p - Vector3D<Precision>(0, 0, sec.fShift), safety);
    } else

      CoaxialConesImplementation::template SafetyToIn<Real_v>(*sec.fCoaxialCones,
                                                              p - Vector3D<Precision>(0, 0, sec.fShift), safety);

    if (!needZ && safety < Real_v(kConeTolerance) &&
        RepairSafetyToInNearSurface<Real_v>(polycone, index, sec, point, safety)) {
      return;
    }

    if (safety < kTolerance) return;
    minSafety       = safety;
    Precision zbase = polycone.fZs[index + 1];
    // going right
    for (int i = index + 1; i < polycone.GetNSections(); ++i) {
      Precision dz = polycone.fZs[i] - zbase;
      if (dz >= minSafety) break;

      GenericPolyconeSection const &sect = polycone.GetSection(i);
      CoaxialConesImplementation::template SafetyToIn<Real_v>(*sect.fCoaxialCones,
                                                              p - Vector3D<Precision>(0, 0, sect.fShift), safety);
      if (safety < minSafety) minSafety = safety;
    }

    // going left if this is possible
    if (index > 0) {
      zbase = polycone.fZs[index - 1];
      for (int i = index - 1; i >= 0; --i) {
        Precision dz = zbase - polycone.fZs[i];
        if (dz >= minSafety) break;
        GenericPolyconeSection const &sect = polycone.GetSection(i);

        CoaxialConesImplementation::template SafetyToIn<Real_v>(*sect.fCoaxialCones,
                                                                p - Vector3D<Precision>(0, 0, sect.fShift), safety);

        if (safety < minSafety) minSafety = safety;
      }
    }
    safety = minSafety;

    return;
  }

  /// @brief Compute safety from an inside point to the generic polycone boundary.
  /// @details The method first enforces the current `SafetyToOut` convention by
  /// classifying the point. It then computes the current-section exit safety and
  /// checks neighboring sections only while their z distance can reduce the
  /// current minimum.
  /// @param polycone Generic polycone storage.
  /// @param point Polycone-local query point.
  /// @param safety Output safety, `0` on surface, or `-1` for wrong-side outside starts.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &polycone,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {
    bool compIn(false), compOut(false);
    GenericKernelForContainsAndInside<Real_v, true>(polycone, point, compIn, compOut);
    if (compOut) {
      safety = -1;
      return;
    }

    if (!compIn && !compOut) {
      safety = 0.;
      return;
    }

    int index = polycone.GetSectionIndex(point.z());
    if (index < 0) {
      safety = -1;
      return;
    }

    GenericPolyconeSection const &sec = polycone.GetSection(index);

    Vector3D<Real_v> p = point - Vector3D<Precision>(0, 0, sec.fShift);
    CoaxialConesImplementation::template SafetyToOut<Real_v>(*sec.fCoaxialCones, p, safety);

    Precision minSafety = safety;
    if (minSafety >= Precision(0.) && minSafety < kConeTolerance) {
      const Vector3D<Precision> globalPoint(point.x(), point.y(), point.z());
      const Vector3D<Precision> localPoint(p.x(), p.y(), p.z());
      if (IsNonExposedSectionZBoundary(polycone, index, sec, globalPoint, localPoint)) {
        // Ignore a section-local z endcap safety when that plane is internal
        // to the full generic polycone.
        Precision lateralSafety = kInfLength;
        if (ComputeSectionSafetyToOutIgnoringZ(sec, localPoint, lateralSafety)) {
          safety = minSafety = lateralSafety;
        }
      }
    }
    if (minSafety == kInfLength) {
      safety = 0.;
      return;
    }
    if (minSafety < kTolerance) {
      safety = 0.;
      return;
    }

    Precision zbase = polycone.fZs[index + 1];
    for (int i = index + 1; i < polycone.GetNSections(); ++i) {
      Precision dz = polycone.fZs[i] - zbase;
      if (dz >= minSafety) break;
      GenericPolyconeSection const &sect = polycone.GetSection(i);
      p                                  = point - Vector3D<Precision>(0, 0, sect.fShift);

      CoaxialConesImplementation::template SafetyToIn<Real_v>(*sect.fCoaxialCones, p, safety);

      if (safety >= Real_v(0.) && safety < Real_v(kConeTolerance)) {
        const Vector3D<Precision> globalPoint(point.x(), point.y(), point.z());
        const Vector3D<Precision> localPoint(p.x(), p.y(), p.z());
        if (IsNonExposedSectionZBoundary(polycone, i, sect, globalPoint, localPoint)) {
          Precision lateralSafety = kInfLength;
          if (ComputeSectionSafetyToOutIgnoringZ(sect, localPoint, lateralSafety)) safety = lateralSafety;
        }
      }
      if (safety >= Real_v(0.) && safety < minSafety) minSafety = safety;
    }

    if (index > 0) {
      zbase = polycone.fZs[index - 1];
      for (int i = index - 1; i >= 0; --i) {
        Precision dz = zbase - polycone.fZs[i];
        if (dz >= minSafety) break;
        GenericPolyconeSection const &sect = polycone.GetSection(i);
        p                                  = point - Vector3D<Precision>(0, 0, sect.fShift);

        CoaxialConesImplementation::template SafetyToIn<Real_v>(*sect.fCoaxialCones, p, safety);

        if (safety >= Real_v(0.) && safety < Real_v(kConeTolerance)) {
          const Vector3D<Precision> globalPoint(point.x(), point.y(), point.z());
          const Vector3D<Precision> localPoint(p.x(), p.y(), p.z());
          if (IsNonExposedSectionZBoundary(polycone, i, sect, globalPoint, localPoint)) {
            Precision lateralSafety = kInfLength;
            if (ComputeSectionSafetyToOutIgnoringZ(sect, localPoint, lateralSafety)) safety = lateralSafety;
          }
        }
        if (safety >= Real_v(0.) && safety < minSafety) minSafety = safety;
      }
    }

    safety = minSafety;
    return;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_GENERICPOLYCONEIMPLEMENTATION_H_
