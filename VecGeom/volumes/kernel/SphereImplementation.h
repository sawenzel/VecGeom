
/// @file SphereImplementation.h
/// @brief Navigation kernels for spherical shells with optional phi and theta cuts.
/// @author Raman Sehgal (raman.sehgal@cern.ch)

#ifndef VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_SPHEREIMPLEMENTATION_H_

#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/SphereStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>
#include "VecGeom/volumes/kernel/OrbImplementation.h"
#include "VecGeom/volumes/SphereUtilities.h"

#include <cstdio>

namespace vecgeom {

VECGEOM_DEVICE_FORWARD_DECLARE(struct SphereImplementation;);
VECGEOM_DEVICE_DECLARE_CONV(struct, SphereImplementation);

inline namespace VECGEOM_IMPL_NAMESPACE {

class PlacedSphere;
template <typename T>
struct SphereStruct;
class UnplacedSphere;

/// @brief Implementation for full, hollow, and angular-cut spheres.
///
/// @details A sphere is modeled as a radial interval clipped by optional phi
/// and theta boundaries. Full variants skip the corresponding angular checks.
struct SphereImplementation {

  using PlacedShape_t    = PlacedSphere;
  using UnplacedStruct_t = SphereStruct<Precision>;
  using UnplacedVolume_t = UnplacedSphere;

  /// @brief Return whether a point is inside or on the sphere.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local point to test.
  /// @param inside Set to true for inside or surface points.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &sphere,
                                                                    Vector3D<Real_v> const &point, bool &inside)
  {
    bool unused = false, outside = false;
    GenericKernelForContainsAndInside<Real_v, false>(sphere, point, unused, outside);
    inside = !outside;
  }

  /// @brief Classify a point as inside, outside, or on the surface.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local point to classify.
  /// @param inside Set to the corresponding `EInside` value.
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &sphere,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    bool completelyinside = false, completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, true>(sphere, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    if (completelyoutside) inside = Inside_t(EInside::kOutside);
    if (completelyinside) inside = Inside_t(EInside::kInside);
  }

  /// @brief Evaluate complete-inside and complete-outside predicates.
  /// @details Combines the radial interval with active phi and theta clips.
  /// Points in tolerance regions can leave both outputs false.
  /// @tparam ForInside Enables complete-inside updates used by `Inside`.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param localPoint Local point to classify.
  /// @param completelyinside Set when all active predicates are safely inside.
  /// @param completelyoutside Set when any active predicate is safely outside.
  template <typename Real_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &sphere, Vector3D<Real_v> const &localPoint, bool &completelyinside,
      bool &completelyoutside)
  {
    Real_v rad2 = localPoint.Mag2();

    // Check radial surfaces
    // Radial check for GenericKernel Start
    if (sphere.fRmin)
      completelyinside =
          rad2 <= MakeMinusTolerantSquare<true>(sphere.fRmax) && rad2 >= MakePlusTolerantSquare<true>(sphere.fRmin);
    else
      completelyinside = rad2 <= MakeMinusTolerantSquare<true>(sphere.fRmax);

    if (sphere.fRmin)
      completelyoutside =
          rad2 >= MakePlusTolerantSquare<true>(sphere.fRmax) || rad2 <= MakeMinusTolerantSquare<true>(sphere.fRmin);
    else
      completelyoutside = rad2 >= MakePlusTolerantSquare<true>(sphere.fRmax);

    // Phi boundaries  : Do not check if it has no phi boundary!
    if (!sphere.fFullPhiSphere) {

      bool completelyoutsidephi = false;
      bool completelyinsidephi  = false;
      sphere.fPhiWedge.GenericKernelForContainsAndInside<Real_v, ForInside>(localPoint, completelyinsidephi,
                                                                            completelyoutsidephi);
      completelyoutside |= completelyoutsidephi;

      if (ForInside) completelyinside &= completelyinsidephi;
    }
    // Phi Check for GenericKernel Over

    // Theta bondaries
    if (!sphere.fFullThetaSphere) {

      bool completelyoutsidetheta = false;
      bool completelyinsidetheta  = false;
      sphere.fThetaCone.GenericKernelForContainsAndInside<Real_v, ForInside>(localPoint, completelyinsidetheta,
                                                                             completelyoutsidetheta);
      completelyoutside |= completelyoutsidetheta;

      if (ForInside) completelyinside &= completelyinsidetheta;
    }
    return;
  }

  /// @brief Compute the first valid entry distance.
  /// @details Radial shell roots, phi-plane roots, and theta-cone roots are
  /// filtered against the active cuts. Theta roots must cross into material;
  /// grazing roots are ignored except for the explicit apex-entry case.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local starting point.
  /// @param direction Normalized local direction.
  /// @param distance Set to the entry distance, `-1`, `0`, or `kInfLength`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &sphere,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /* stepMax */, Real_v &distance)
  {
    distance  = kInfLength;
    bool done = false;

    bool fullPhiSphere   = sphere.fFullPhiSphere;
    bool fullThetaSphere = sphere.fFullThetaSphere;

    // General Precalcs
    Real_v rad2    = point.Mag2();
    Real_v pDotV3d = point.Dot(direction);

    Real_v c = rad2 - sphere.fRmax * sphere.fRmax;

    if (SphereUtilities::IsCompletelyInside<Real_v>(sphere, point)) {
      distance = Real_v(-1.0);
      return;
    }

    if (SphereUtilities::IsPointOnSurfaceAndMovingOut<Real_v, false>(sphere, point, direction)) {
      distance = Real_v(0.0);
      return;
    }

    Real_v sd1(kInfLength);
    Real_v sd2(kInfLength);
    Real_v d2 = (pDotV3d * pDotV3d - c);
    if (d2 < Real_v(0.) || ((c > Real_v(0.)) && (pDotV3d > Real_v(0.)))) return;

    sd1 = -pDotV3d - Sqrt(d2);

    Real_v outerDist(kInfLength);
    Real_v innerDist(kInfLength);

    if (sphere.fFullSphere) {
      if (sd1 > Real_v(kTolerance)) outerDist = sd1;
    } else {
      if (sd1 > Real_v(kTolerance) && sd1 < Real_v(kInfLength)) {
        Vector3D<Real_v> tmpPt = point + sd1 * direction;
        if (sphere.fPhiWedge.Inside<Real_v, Inside_t>(tmpPt) != EInside::kOutside &&
            sphere.fThetaCone.Inside<Real_v, Inside_t>(tmpPt) != EInside::kOutside)
          outerDist = sd1;
      }
    }

    if (sphere.fRmin) {
      c  = rad2 - sphere.fRmin * sphere.fRmin;
      d2 = pDotV3d * pDotV3d - c;
      if (d2 >= Real_v(0.)) sd2 = -pDotV3d + Sqrt(d2);

      if (sphere.fFullSphere) {
        if (sd2 > Real_v(kTolerance)) innerDist = sd2;
      } else {
        if (sd2 > Real_v(kTolerance) && sd2 < Real_v(kInfLength)) {
          Vector3D<Real_v> tmpPt = point + sd2 * direction;
          if (sphere.fPhiWedge.Inside<Real_v, Inside_t>(tmpPt) != EInside::kOutside &&
              sphere.fThetaCone.Inside<Real_v, Inside_t>(tmpPt) != EInside::kOutside)
            innerDist = sd2;
        }
      }
    }

    distance = Min(outerDist, innerDist);

    if (!fullPhiSphere) {
      GetMinDistFromPhi<Real_v, true>(sphere, point, direction, done, distance);
    }

    Real_v distThetaMin(kInfLength);

    if (!fullThetaSphere) {
      bool intsect1 = false;
      bool intsect2 = false;
      Real_v distTheta1(kInfLength), distCone1(kInfLength);
      Real_v distTheta2(kInfLength), distCone2(kInfLength);
      Real_v rmin2 = sphere.fRmin * sphere.fRmin;
      Real_v rmax2 = sphere.fRmax * sphere.fRmax;
      Vector3D<Real_v> coneIntSecPt1;
      Vector3D<Real_v> coneIntSecPt2;

      sphere.fThetaCone.DistanceToIn<Real_v>(point, direction, distTheta1, distTheta2, intsect1,
                                             intsect2); //,cone1IntSecPt, cone2IntSecPt);
      bool cone1MovesIn    = false;
      bool cone1EntersApex = false;
      bool candidateCone1  = intsect1 && (distTheta1 > Real_v(kTolerance));
      Real_v pDotV2d       = point.x() * direction.x() + point.y() * direction.y();
      Real_v dirRho2       = direction.Perp2();
      Real_v pointZDir     = point.z() * direction.z();
      Real_v dirZ2         = direction.z() * direction.z();

      if (candidateCone1) {
        distCone1 = rad2 + distTheta1 * (Real_v(2.) * pDotV3d + distTheta1);
        candidateCone1 &= distCone1 >= rmin2 && distCone1 <= rmax2;
        if (candidateCone1) {
          Real_v motion1 = -direction.z();
          if (Abs(sphere.fSTheta - kHalfPi) > kTolerance) {
            Real_v tanTheta2 = Real_v(sphere.fThetaCone.GetTanSTheta2());
            Real_v a         = dirRho2 - dirZ2 * tanTheta2;
            Real_v b         = pDotV2d - pointZDir * tanTheta2;
            motion1          = b + distTheta1 * a;
          }
          cone1MovesIn    = SphereUtilities::IsThetaConeMotion<Real_v, true, false>(sphere, motion1);
          cone1EntersApex = (sphere.fRmin == 0.) && (distCone1 <= Real_v(kTolerance * kTolerance)) &&
                            sphere.fThetaCone.Contains<Real_v>(direction);
          if (!fullPhiSphere) cone1EntersApex &= sphere.fPhiWedge.Contains<Real_v>(direction);
        }
      }

      bool cone2MovesIn    = false;
      bool cone2EntersApex = false;
      bool candidateCone2  = intsect2 && (distTheta2 > Real_v(kTolerance));
      if (candidateCone2) {
        distCone2 = rad2 + distTheta2 * (Real_v(2.) * pDotV3d + distTheta2);
        candidateCone2 &= distCone2 >= rmin2 && distCone2 <= rmax2;
        if (candidateCone2) {
          Real_v motion2 = -direction.z();
          if (Abs(sphere.eTheta - kHalfPi) > kTolerance) {
            Real_v tanTheta2 = Real_v(sphere.fThetaCone.GetTanETheta2());
            Real_v a         = dirRho2 - dirZ2 * tanTheta2;
            Real_v b         = pDotV2d - pointZDir * tanTheta2;
            motion2          = b + distTheta2 * a;
          }
          cone2MovesIn    = SphereUtilities::IsThetaConeMotion<Real_v, false, false>(sphere, motion2);
          cone2EntersApex = (sphere.fRmin == 0.) && (distCone2 <= Real_v(kTolerance * kTolerance)) &&
                            sphere.fThetaCone.Contains<Real_v>(direction);
          if (!fullPhiSphere) cone2EntersApex &= sphere.fPhiWedge.Contains<Real_v>(direction);
        }
      }

      bool isValidCone1 = candidateCone1 && (cone1MovesIn || cone1EntersApex);
      bool isValidCone2 = candidateCone2 && (cone2MovesIn || cone2EntersApex);

      if (!fullPhiSphere && isValidCone1 && !cone1EntersApex) {
        coneIntSecPt1 = point + distTheta1 * direction;
        isValidCone1  = sphere.fPhiWedge.Contains<Real_v>(coneIntSecPt1);
      }
      if (!fullPhiSphere && isValidCone2 && !cone2EntersApex) {
        coneIntSecPt2 = point + distTheta2 * direction;
        isValidCone2  = sphere.fPhiWedge.Contains<Real_v>(coneIntSecPt2);
      }
      if (isValidCone2 && !isValidCone1) distThetaMin = distTheta2;
      if (isValidCone1 && !isValidCone2) distThetaMin = distTheta1;
      if (isValidCone1 && isValidCone2) distThetaMin = Min(distTheta1, distTheta2);

      bool isApexEntry = (sphere.fRmin == 0.) && (rad2 <= Real_v(kTolerance * kTolerance)) &&
                         sphere.fThetaCone.Contains<Real_v>(direction);
      if (!fullPhiSphere) isApexEntry &= sphere.fPhiWedge.Contains<Real_v>(direction);
      if (isApexEntry) distThetaMin = Real_v(0.);
    }

    distance = Min(distThetaMin, distance);
  }

  /// @brief Compute the first valid exit distance.
  /// @details Considers radial shell roots, theta-cone roots, and phi-plane
  /// roots, returning the nearest accepted boundary crossing.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local starting point.
  /// @param direction Normalized local direction.
  /// @param distance Set to the exit distance, `-1`, `0`, or `kInfLength`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &sphere,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {

    distance  = kInfLength;
    bool done = false;

    Real_v snxt(kInfLength);

    Real_v d2(0.);

    Real_v pDotV3d = point.Dot(direction);

    Real_v rad2 = point.Mag2();
    Real_v c    = rad2 - sphere.fRmax * sphere.fRmax;

    Real_v sd1(kInfLength);
    Real_v sd2(kInfLength);

    if (SphereUtilities::IsCompletelyOutside<Real_v>(sphere, point)) {
      distance = Real_v(-1.0);
      return;
    }

    if (SphereUtilities::IsPointOnSurfaceAndMovingOut<Real_v, true>(sphere, point, direction)) {
      distance = Real_v(0.0);
      return;
    }

    d2 = (pDotV3d * pDotV3d - c);
    if (d2 >= Real_v(0.)) sd1 = -pDotV3d + Sqrt(d2);

    if (sphere.fRmin) {
      c  = rad2 - sphere.fRmin * sphere.fRmin;
      d2 = (pDotV3d * pDotV3d - c);
      if (d2 >= Real_v(0.) && pDotV3d < -kToleranceDist<Real_v>) sd2 = -pDotV3d - Sqrt(d2);
    }

    snxt = Min(sd1, sd2);

    bool condSemi = ((sphere.fSTheta == 0. && sphere.eTheta == kPi / 2.) && direction.z() >= Real_v(0.)) ||
                    ((sphere.fSTheta == kPi / 2. && sphere.eTheta == kPi) && direction.z() <= Real_v(0.));
    if (condSemi) {
      distance = snxt;
      return;
    }

    Real_v distThetaMin(kInfLength);
    Real_v distPhiMin(kInfLength);

    if (!sphere.fFullThetaSphere) {
      bool intsect1 = false;
      bool intsect2 = false;
      Real_v distTheta1(kInfLength);
      Real_v distTheta2(kInfLength);
      sphere.fThetaCone.DistanceToOut<Real_v>(point, direction, distTheta1, distTheta2, intsect1, intsect2);
      if (intsect2 && !intsect1) distThetaMin = distTheta2;
      if (!intsect2 && intsect1) distThetaMin = distTheta1;
      if (intsect2 && intsect1) distThetaMin = Min(distTheta1, distTheta2);
    }

    distance = Min(distThetaMin, snxt);

    if (!sphere.fFullPhiSphere) {
      if (sphere.fDPhi <= kPi) {
        Real_v distPhi1;
        Real_v distPhi2;
        sphere.fPhiWedge.DistanceToOut<Real_v>(point, direction, distPhi1, distPhi2);
        distPhiMin = Min(distPhi1, distPhi2);
        distance   = Min(distPhiMin, distance);
      } else {
        GetMinDistFromPhi<Real_v, false>(sphere, point, direction, done, distance);
      }
    }
  }

  /// @brief Test phi-plane crossings and update the best distance.
  /// @details Candidate phi intersections are accepted only when the hit point
  /// is on the selected phi surface and inside the radial/theta extents.
  /// @tparam DistToIn Selects entry or exit phi-plane queries.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param localPoint Local starting point.
  /// @param localDir Normalized local direction.
  /// @param done Suppresses updates when an earlier phase completed the query.
  /// @param distance Current best distance, updated if a closer phi hit exists.
  template <typename Real_v, bool DistToIn>
  VECCORE_ATT_HOST_DEVICE static void GetMinDistFromPhi(UnplacedStruct_t const &sphere,
                                                        Vector3D<Real_v> const &localPoint,
                                                        Vector3D<Real_v> const &localDir, bool done, Real_v &distance)
  {
    Real_v distPhi1(kInfLength);
    Real_v distPhi2(kInfLength);
    Real_v dist(kInfLength);

    if (DistToIn)
      sphere.fPhiWedge.DistanceToIn<Real_v>(localPoint, localDir, distPhi1, distPhi2);
    else
      sphere.fPhiWedge.DistanceToOut<Real_v>(localPoint, localDir, distPhi1, distPhi2);

    bool containsCond1 = false, containsCond2 = false;
    bool tempCond = false;
    // Min Face
    dist = Min(distPhi1, distPhi2);
    if (dist < kInfLength) {
      Vector3D<Real_v> tmpPt = localPoint + dist * localDir;
      Real_v rad2            = tmpPt.Mag2();

      tempCond = ((dist == distPhi1) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, true>(tmpPt)) ||
                 ((dist == distPhi2) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, false>(tmpPt));

      containsCond1 = tempCond && (rad2 > sphere.fRmin * sphere.fRmin) && (rad2 < sphere.fRmax * sphere.fRmax) &&
                      sphere.fThetaCone.Contains<Real_v>(tmpPt);
    }

    if (!done && containsCond1) distance = Min(dist, distance);

    // Max Face
    dist = Max(distPhi1, distPhi2);
    if (dist < kInfLength) {
      Vector3D<Real_v> tmpPt = localPoint + dist * localDir;

      Real_v rad2 = tmpPt.Mag2();
      tempCond    = ((dist == distPhi1) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, true>(tmpPt)) ||
                    ((dist == distPhi2) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, false>(tmpPt));

      containsCond2 = tempCond && (rad2 > sphere.fRmin * sphere.fRmin) && (rad2 < sphere.fRmax * sphere.fRmax) &&
                      sphere.fThetaCone.Contains<Real_v>(tmpPt);
    }
    if (!done && !containsCond1 && containsCond2) distance = Min(dist, distance);
  }

  /// @brief Compute signed safety for entering the sphere.
  /// @details Full-theta spheres use radial safety, optionally combined with
  /// phi safety, to decide outside, surface, and inside states directly.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local point to test.
  /// @param safety Set to positive distance, `0`, or `-1`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &sphere,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    // General Precalcs
    Real_v rad = point.Mag();

    Real_v safeRMin(0.);
    Real_v safeRMax(0.);

    if (sphere.fFullThetaSphere) {
      // With no theta cuts, signed radial/phi safety fully determines inside,
      // surface, and outside state, so avoid a separate classification pass.
      if (sphere.fRmin) {
        safeRMin = sphere.fRmin - rad;
        safeRMax = rad - sphere.fRmax;
        safety   = safeRMin > safeRMax ? safeRMin : safeRMax;
      } else {
        safety = rad - sphere.fRmax;
      }

      if (!sphere.fFullPhiSphere) safety = Max(sphere.fPhiWedge.SafetyToIn<Real_v>(point), safety);
      if (safety > Real_v(kTolerance)) return;
      safety = safety < Real_v(-kTolerance) ? Real_v(-1.) : Real_v(0.);
      return;
    }

    bool completelyinside  = false;
    bool completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, true>(sphere, point, completelyinside, completelyoutside);

    if (completelyinside) {
      safety = Real_v(-1.0);
      return;
    }

    if (!completelyoutside) {
      safety = Real_v(0.0);
      return;
    }

    if (sphere.fRmin) {
      safeRMin = sphere.fRmin - rad;
      safeRMax = rad - sphere.fRmax;
      safety   = safeRMin > safeRMax ? safeRMin : safeRMax;
    } else {
      safety = rad - sphere.fRmax;
    }
    // Distance to r shells over

    // Distance to phi extent
    if (!sphere.fFullPhiSphere) {
      Real_v safetyPhi = sphere.fPhiWedge.SafetyToIn<Real_v>(point);
      safety           = Max(safetyPhi, safety);
    }

    // Distance to Theta extent
    if (!sphere.fFullThetaSphere) {
      Real_v safetyTheta = sphere.fThetaCone.SafetyToIn<Real_v>(point);
      safety             = Max(safetyTheta, safety);
    }
  }

  /// @brief Compute signed safety for leaving the sphere.
  /// @details Uses the nearest active radial, phi, or theta boundary for points
  /// classified safely inside.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local point to test.
  /// @param safety Set to positive distance, `0`, or `-1`.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &sphere,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {

    Real_v rad = point.Mag();

    bool completelyinside  = false;
    bool completelyoutside = false;
    GenericKernelForContainsAndInside<Real_v, true>(sphere, point, completelyinside, completelyoutside);
    if (completelyoutside) {
      safety = Real_v(-1.0);
      return;
    }

    if (!completelyinside) {
      safety = Real_v(0.0);
      return;
    }

    // Distance to r shells
    if (sphere.fRmin) {
      Real_v safeRMin = (rad - sphere.fRmin);
      Real_v safeRMax = (sphere.fRmax - rad);
      safety          = safeRMin < safeRMax ? safeRMin : safeRMax;
    } else {
      safety = sphere.fRmax - rad;
    }

    // Distance to phi extent
    if (!sphere.fFullPhiSphere) {
      Real_v safetyPhi = sphere.fPhiWedge.SafetyToOut<Real_v>(point);
      safety           = Min(safetyPhi, safety);
    }

    // Distance to Theta extent
    if (!sphere.fFullThetaSphere) {
      Real_v safeTheta = sphere.fThetaCone.SafetyToOut<Real_v>(point);
      safety           = Min(safeTheta, safety);
    }
  }

  /// @brief Approximate the normal from the closest sphere boundary.
  /// @details Used by `Normal` when the point is not on a recognized surface.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local point to evaluate.
  /// @return Normal of the closest radial, phi, or theta boundary.
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> ApproxSurfaceNormalKernel(UnplacedStruct_t const &sphere,
                                                                            Vector3D<Real_v> const &point)
  {
    using vecCore::math::Min;
    Vector3D<Real_v> norm(0., 0., 0.);
    Real_v radius     = point.Mag();
    Real_v distRMax   = Abs(radius - sphere.fRmax);
    Real_v distRMin   = InfinityLength<Real_v>();
    Real_v distPhi1   = InfinityLength<Real_v>();
    Real_v distPhi2   = InfinityLength<Real_v>();
    Real_v distTheta1 = InfinityLength<Real_v>();
    Real_v distTheta2 = InfinityLength<Real_v>();
    Real_v distMin    = distRMax;

    if (sphere.fRmin > 0.) {
      distRMin = Abs(sphere.fRmin - radius);
      distMin  = Min(distRMin, distRMax);
    }

    if (!sphere.fFullPhiSphere) {
      distPhi1 = Abs(point.x() * sphere.fPhiWedge.GetNormal1().x() + point.y() * sphere.fPhiWedge.GetNormal1().y());
      distPhi2 = Abs(point.x() * sphere.fPhiWedge.GetNormal2().x() + point.y() * sphere.fPhiWedge.GetNormal2().y());
      distMin  = Min(distMin, distPhi1, distPhi2);
    }

    if (!sphere.fFullThetaSphere) {
      Real_v rho = point.Perp();
      distTheta1 = sphere.fThetaCone.DistanceToLine<Real_v>(sphere.fThetaCone.GetSlope1(), rho, point.z());
      distTheta2 = sphere.fThetaCone.DistanceToLine<Real_v>(sphere.fThetaCone.GetSlope2(), rho, point.z());
      distMin    = Min(distMin, distTheta1, distTheta2);
    }

    if (distMin == distRMax) norm = point.Unit();
    if (distMin == distRMin) norm = -point.Unit();

    Vector3D<Real_v> normal1 = sphere.fPhiWedge.GetNormal1();
    Vector3D<Real_v> normal2 = sphere.fPhiWedge.GetNormal2();
    if (distMin == distPhi1) norm = -normal1;
    if (distMin == distPhi2) norm = -normal2;

    if (distMin == distTheta1) norm += sphere.fThetaCone.GetNormal1<Real_v>(point);
    if (distMin == distTheta2) norm += sphere.fThetaCone.GetNormal2<Real_v>(point);

    return norm;
  }

  /// @brief Return the outward normal and whether a surface was identified.
  /// @details Surface normals are accumulated from all matching active
  /// boundaries; off-surface points use the closest-boundary fallback.
  /// @param sphere Sphere parameters and cached angular state.
  /// @param point Local surface point.
  /// @param valid Set when at least one surface was identified.
  /// @return Outward normal direction.
  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> Normal(UnplacedStruct_t const &sphere,
                                                                              Vector3D<Real_v> const &point,
                                                                              typename vecCore::Mask_v<Real_v> &valid)
  {
    Vector3D<Real_v> normal(0., 0., 0.);
    normal.Set(1e-30);

    /* Assumption : This function assumes that the point is on the surface.
     *
     * Algorithm :
     * Detect all those surfaces on which the point is at, and count the
     * numOfSurfaces. if(numOfSurfaces == 1) then normal corresponds to the
     * normal for that particular case.
     *
     * if(numOfSurfaces > 1 ), then add the normals corresponds to different
     * cases, and finally normalize it and return.
     *
     * We need following function
     * IsPointOnInnerRadius()
     * IsPointOnOuterRadius()
     * IsPointOnStartPhi()
     * IsPointOnEndPhi()
     * IsPointOnStartTheta()
     * IsPointOnEndTheta()
     *
     * set valid=true if numOfSurface > 0
     *
     * if above mentioned assumption not followed , ie.
     * In case the given point is outside, then find the closest boundary,
     * the required normal will be the normal to that boundary.
     * This logic is implemented in "ApproxSurfaceNormalKernel" function
     */

    bool isPointOutside = false;

    bool isPointInside = false;
    GenericKernelForContainsAndInside<Real_v, true>(sphere, point, isPointInside, isPointOutside);
    if (isPointInside || isPointOutside) normal = ApproxSurfaceNormalKernel<Real_v>(sphere, point);

    valid = false;

    int noSurfaces            = 0;
    bool isPointOnOuterRadius = SphereUtilities::IsPointOnOuterRadius<Real_v>(sphere, point);

    if (isPointOnOuterRadius) ++noSurfaces;
    if (!isPointOutside && isPointOnOuterRadius) normal += point.Unit();

    if (sphere.fRmin) {
      bool isPointOnInnerRadius = SphereUtilities::IsPointOnInnerRadius<Real_v>(sphere, point);
      if (isPointOnInnerRadius) ++noSurfaces;
      if (!isPointOutside && isPointOnInnerRadius) normal -= point.Unit();
    }

    if (!sphere.fFullPhiSphere) {
      bool isPointOnStartPhi = SphereUtilities::IsPointOnStartPhi<Real_v>(sphere, point);
      bool isPointOnEndPhi   = SphereUtilities::IsPointOnEndPhi<Real_v>(sphere, point);
      if (isPointOnStartPhi) ++noSurfaces;
      if (isPointOnEndPhi) ++noSurfaces;
      if (!isPointOutside && isPointOnStartPhi) normal -= sphere.fPhiWedge.GetNormal1();
      if (!isPointOutside && isPointOnEndPhi) normal -= sphere.fPhiWedge.GetNormal2();
    }

    if (!sphere.fFullThetaSphere) {
      bool isPointOnStartTheta = SphereUtilities::IsPointOnStartTheta<Real_v>(sphere, point);
      bool isPointOnEndTheta   = SphereUtilities::IsPointOnEndTheta<Real_v>(sphere, point);

      if (isPointOnStartTheta) ++noSurfaces;
      if (!isPointOutside && isPointOnStartTheta) normal += sphere.fThetaCone.GetNormal1<Real_v>(point);

      if (isPointOnEndTheta) ++noSurfaces;
      if (!isPointOutside && isPointOnEndTheta) normal += sphere.fThetaCone.GetNormal2<Real_v>(point);

      Vector3D<Real_v> tempNormal(0., 0., -1.);
      if (!isPointOutside && isPointOnStartTheta && isPointOnEndTheta && (sphere.eTheta <= kPi / 2.))
        normal = tempNormal;
      Vector3D<Real_v> tempNormal2(0., 0., 1.);
      if (!isPointOutside && isPointOnStartTheta && isPointOnEndTheta && (sphere.fSTheta >= kPi / 2.))
        normal = tempNormal2;
    }

    normal.Normalize();

    valid = (noSurfaces > 0);

    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_sphereIMPLEMENTATION_H_
