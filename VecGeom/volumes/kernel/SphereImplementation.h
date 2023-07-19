
/// @file SphereImplementation.h
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

struct SphereImplementation {

  using PlacedShape_t    = PlacedSphere;
  using UnplacedStruct_t = SphereStruct<Precision>;
  using UnplacedVolume_t = UnplacedSphere;

  template <typename Real_v, typename Bool_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Contains(UnplacedStruct_t const &sphere,
                                                                    Vector3D<Real_v> const &point, Bool_v &inside)
  {
    Bool_v unused, outside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, false>(sphere, point, unused, outside);
    inside = !outside;
  }

  // BIG QUESTION: DO WE WANT TO GIVE ALL 3 TEMPLATE PARAMETERS
  // -- OR -- DO WE WANT TO DEDUCE Bool_v, Index_t from Real_v???
  template <typename Real_v, typename Inside_t>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void Inside(UnplacedStruct_t const &sphere,
                                                                  Vector3D<Real_v> const &point, Inside_t &inside)
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    using InsideBool_v = vecCore::Mask_v<Inside_t>;
    Bool_v completelyinside, completelyoutside;
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(sphere, point, completelyinside, completelyoutside);
    inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
    vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  }

  template <typename Real_v, typename Bool_v, bool ForInside>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void GenericKernelForContainsAndInside(
      UnplacedStruct_t const &sphere, Vector3D<Real_v> const &localPoint, Bool_v &completelyinside,
      Bool_v &completelyoutside)
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

      Bool_v completelyoutsidephi(false);
      Bool_v completelyinsidephi(false);
      sphere.fPhiWedge.GenericKernelForContainsAndInside<Real_v, ForInside>(localPoint, completelyinsidephi,
                                                                            completelyoutsidephi);
      completelyoutside |= completelyoutsidephi;

      if (ForInside) completelyinside &= completelyinsidephi;
    }
    // Phi Check for GenericKernel Over

    // Theta bondaries
    if (!sphere.fFullThetaSphere) {

      Bool_v completelyoutsidetheta(false);
      Bool_v completelyinsidetheta(false);
      sphere.fThetaCone.GenericKernelForContainsAndInside<Real_v, ForInside>(localPoint, completelyinsidetheta,
                                                                             completelyoutsidetheta);
      completelyoutside |= completelyoutsidetheta;

      if (ForInside) completelyinside &= completelyinsidetheta;
    }
    return;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToIn(UnplacedStruct_t const &sphere,
                                                                        Vector3D<Real_v> const &point,
                                                                        Vector3D<Real_v> const &direction,
                                                                        Real_v const & /* stepMax */, Real_v &distance)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    distance     = kInfLength;

    Bool_v done(false);

    bool fullPhiSphere   = sphere.fFullPhiSphere;
    bool fullThetaSphere = sphere.fFullThetaSphere;

    Vector3D<Real_v> tmpPt;
    // General Precalcs
    Real_v rad2    = point.Mag2();
    Real_v pDotV3d = point.Dot(direction);

    Real_v c = rad2 - sphere.fRmax * sphere.fRmax;

    Bool_v cond = SphereUtilities::IsCompletelyInside<Real_v>(sphere, point);
    vecCore__MaskedAssignFunc(distance, cond, Real_v(-1.0));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    cond = SphereUtilities::IsPointOnSurfaceAndMovingOut<Real_v, false>(sphere, point, direction);
    vecCore__MaskedAssignFunc(distance, !done && cond, Real_v(0.0));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    Real_v sd1(kInfLength);
    Real_v sd2(kInfLength);
    Real_v d2 = (pDotV3d * pDotV3d - c);
    cond      = (d2 < Real_v(0.) || ((c > Real_v(0.)) && (pDotV3d > Real_v(0.))));
    done |= cond;
    if (vecCore::MaskFull(done)) return; // Returning in case of no intersection with outer shell

    // Note: Abs(d2) was introduced to avoid Sqrt(negative) in other lanes than the ones satisfying d2>=0.
    vecCore__MaskedAssignFunc(sd1, d2 >= Real_v(0.), (-pDotV3d - Sqrt(Abs(d2))));

    Real_v outerDist(kInfLength);
    Real_v innerDist(kInfLength);

    if (sphere.fFullSphere) {
      vecCore::MaskedAssign(outerDist, !done && (sd1 >= Real_v(0.)), sd1);
    } else {
      tmpPt = point + sd1 * direction;
      vecCore::MaskedAssign(outerDist,
                            !done && sphere.fPhiWedge.Contains<Real_v>(tmpPt) &&
                                sphere.fThetaCone.Contains<Real_v>(tmpPt) && (sd1 >= Real_v(0.)),
                            sd1);
    }

    if (sphere.fRmin) {
      c  = rad2 - sphere.fRmin * sphere.fRmin;
      d2 = pDotV3d * pDotV3d - c;
      // Note: Abs(d2) was introduced to avoid Sqrt(negative) in other lanes than the ones satisfying d2>=0.
      vecCore__MaskedAssignFunc(sd2, d2 >= Real_v(0.), (-pDotV3d + Sqrt(Abs(d2))));

      if (sphere.fFullSphere) {
        vecCore::MaskedAssign(innerDist, !done && (sd2 >= Real_v(0.)), sd2);
      } else {
        //   std::cerr<<" ---- Called by InnerRad ---- " << std::endl;
        tmpPt = point + sd2 * direction;
        vecCore::MaskedAssign(innerDist,
                              !done && (sd2 >= Real_v(0.)) && sphere.fPhiWedge.Contains<Real_v>(tmpPt) &&
                                  sphere.fThetaCone.Contains<Real_v>(tmpPt),
                              sd2);
      }
    }

    distance = Min(outerDist, innerDist);

    if (!fullPhiSphere) {
      GetMinDistFromPhi<Real_v, true>(sphere, point, direction, done, distance);
    }

    Real_v distThetaMin(kInfLength);

    if (!fullThetaSphere) {
      Bool_v intsect1(false);
      Bool_v intsect2(false);
      Real_v distTheta1(kInfLength);
      Real_v distTheta2(kInfLength);

      sphere.fThetaCone.DistanceToIn<Real_v>(point, direction, distTheta1, distTheta2, intsect1,
                                             intsect2); //,cone1IntSecPt, cone2IntSecPt);
      Vector3D<Real_v> coneIntSecPt1 = point + distTheta1 * direction;
      Real_v distCone1               = coneIntSecPt1.Mag2();

      Vector3D<Real_v> coneIntSecPt2 = point + distTheta2 * direction;
      Real_v distCone2               = coneIntSecPt2.Mag2();

      Bool_v isValidCone1 =
          (distCone1 >= sphere.fRmin * sphere.fRmin && distCone1 <= sphere.fRmax * sphere.fRmax) && intsect1;
      Bool_v isValidCone2 =
          (distCone2 >= sphere.fRmin * sphere.fRmin && distCone2 <= sphere.fRmax * sphere.fRmax) && intsect2;

      if (!fullPhiSphere) {
        isValidCone1 &= sphere.fPhiWedge.Contains<Real_v>(coneIntSecPt1);
        isValidCone2 &= sphere.fPhiWedge.Contains<Real_v>(coneIntSecPt2);
      }
      vecCore::MaskedAssign(distThetaMin, (!done && isValidCone2 && !isValidCone1), distTheta2);
      vecCore::MaskedAssign(distThetaMin, (!done && isValidCone1 && !isValidCone2), distTheta1);
      vecCore__MaskedAssignFunc(distThetaMin, (!done && isValidCone1 && isValidCone2), Min(distTheta1, distTheta2));
    }

    distance = Min(distThetaMin, distance);
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void DistanceToOut(UnplacedStruct_t const &sphere,
                                                                         Vector3D<Real_v> const &point,
                                                                         Vector3D<Real_v> const &direction,
                                                                         Real_v const & /* stepMax */, Real_v &distance)
  {

    using Bool_v = typename vecCore::Mask_v<Real_v>;

    distance = kInfLength;
    Bool_v done(false);

    Real_v snxt(kInfLength);

    // Intersection point
    Vector3D<Real_v> intSecPt;
    Real_v d2(0.);

    Real_v pDotV3d = point.Dot(direction);

    Real_v rad2 = point.Mag2();
    Real_v c    = rad2 - sphere.fRmax * sphere.fRmax;

    Real_v sd1(kInfLength);
    Real_v sd2(kInfLength);

    Bool_v cond = SphereUtilities::IsCompletelyOutside<Real_v>(sphere, point);
    vecCore__MaskedAssignFunc(distance, cond, Real_v(-1.0));

    done |= cond;
    if (vecCore::MaskFull(done)) return;

    cond = SphereUtilities::IsPointOnSurfaceAndMovingOut<Real_v, true>(sphere, point, direction);
    vecCore__MaskedAssignFunc(distance, !done && cond, Real_v(0.0));
    done |= cond;
    if (vecCore::MaskFull(done)) return;

    // Note: Abs(d2) was introduced to avoid Sqrt(negative) in other lanes than the ones satisfying d2>=0.
    d2 = (pDotV3d * pDotV3d - c);
    vecCore__MaskedAssignFunc(sd1, (!done && (d2 >= Real_v(0.))), (-pDotV3d + Sqrt(Abs(d2))));

    if (sphere.fRmin) {
      c  = rad2 - sphere.fRmin * sphere.fRmin;
      d2 = (pDotV3d * pDotV3d - c);
      vecCore__MaskedAssignFunc(sd2, (!done && (d2 >= Real_v(0.)) && (pDotV3d < Real_v(0.))),
                                (-pDotV3d - Sqrt(Abs(d2))));
    }

    snxt = Min(sd1, sd2);

    Bool_v condSemi = (Bool_v(sphere.fSTheta == 0. && sphere.eTheta == kPi / 2.) && direction.z() >= Real_v(0.)) ||
                      (Bool_v(sphere.fSTheta == kPi / 2. && sphere.eTheta == kPi) && direction.z() <= Real_v(0.));
    vecCore::MaskedAssign(distance, !done && condSemi, snxt);
    done |= condSemi;
    if (vecCore::MaskFull(done)) return;

    Real_v distThetaMin(kInfLength);
    Real_v distPhiMin(kInfLength);

    if (!sphere.fFullThetaSphere) {
      Bool_v intsect1(false);
      Bool_v intsect2(false);
      Real_v distTheta1(kInfLength);
      Real_v distTheta2(kInfLength);
      sphere.fThetaCone.DistanceToOut<Real_v>(point, direction, distTheta1, distTheta2, intsect1, intsect2);
      vecCore::MaskedAssign(distThetaMin, (intsect2 && !intsect1), distTheta2);
      vecCore::MaskedAssign(distThetaMin, (!intsect2 && intsect1), distTheta1);
      vecCore__MaskedAssignFunc(distThetaMin, (intsect2 && intsect1), Min(distTheta1, distTheta2));
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

  template <typename Real_v, bool DistToIn>
  VECCORE_ATT_HOST_DEVICE static void GetMinDistFromPhi(UnplacedStruct_t const &sphere,
                                                        Vector3D<Real_v> const &localPoint,
                                                        Vector3D<Real_v> const &localDir,
                                                        typename vecCore::Mask_v<Real_v> &done, Real_v &distance)
  {
    using Bool_v = typename vecCore::Mask_v<Real_v>;
    Real_v distPhi1(kInfLength);
    Real_v distPhi2(kInfLength);
    Real_v dist(kInfLength);

    if (DistToIn)
      sphere.fPhiWedge.DistanceToIn<Real_v>(localPoint, localDir, distPhi1, distPhi2);
    else
      sphere.fPhiWedge.DistanceToOut<Real_v>(localPoint, localDir, distPhi1, distPhi2);

    Bool_v containsCond1(false), containsCond2(false);
    // Min Face
    dist                   = Min(distPhi1, distPhi2);
    Vector3D<Real_v> tmpPt = localPoint + dist * localDir;
    Real_v rad2            = tmpPt.Mag2();

    Bool_v tempCond(false);
    tempCond = ((dist == distPhi1) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, true>(tmpPt)) ||
               ((dist == distPhi2) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, false>(tmpPt));

    containsCond1 = tempCond && (rad2 > sphere.fRmin * sphere.fRmin) && (rad2 < sphere.fRmax * sphere.fRmax) &&
                    sphere.fThetaCone.Contains<Real_v>(tmpPt);

    vecCore__MaskedAssignFunc(distance, !done && containsCond1, Min(dist, distance));

    // Max Face
    dist  = Max(distPhi1, distPhi2);
    tmpPt = localPoint + dist * localDir;

    rad2     = tmpPt.Mag2();
    tempCond = Bool_v(false);
    tempCond = ((dist == distPhi1) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, true>(tmpPt)) ||
               ((dist == distPhi2) && sphere.fPhiWedge.IsOnSurfaceGeneric<Real_v, false>(tmpPt));

    containsCond2 = tempCond && (rad2 > sphere.fRmin * sphere.fRmin) && (rad2 < sphere.fRmax * sphere.fRmax) &&
                    sphere.fThetaCone.Contains<Real_v>(tmpPt);
    vecCore__MaskedAssignFunc(distance, ((!done) && (!containsCond1) && containsCond2), Min(dist, distance));
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToIn(UnplacedStruct_t const &sphere,
                                                                      Vector3D<Real_v> const &point, Real_v &safety)
  {
    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);

    // General Precalcs
    Real_v rad = point.Mag();

    Real_v safeRMin(0.);
    Real_v safeRMax(0.);

    Bool_v completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(sphere, point, completelyinside, completelyoutside);

    vecCore__MaskedAssignFunc(safety, completelyinside, Real_v(-1.0));
    done |= completelyinside;
    if (vecCore::MaskFull(done)) return;

    Bool_v isOnSurface = !completelyinside && !completelyoutside;
    vecCore__MaskedAssignFunc(safety, !done && isOnSurface, Real_v(0.0));
    done |= isOnSurface;
    if (vecCore::MaskFull(done)) return;

    if (sphere.fRmin) {
      safeRMin = sphere.fRmin - rad;
      safeRMax = rad - sphere.fRmax;
      safety   = vecCore::Blend(!done && (safeRMin > safeRMax), safeRMin, safeRMax);
    } else {
      vecCore__MaskedAssignFunc(safety, !done, (rad - sphere.fRmax));
    }
    // Distance to r shells over

    // Distance to phi extent
    if (!sphere.fFullPhiSphere) {
      Real_v safetyPhi = sphere.fPhiWedge.SafetyToIn<Real_v>(point);
      vecCore__MaskedAssignFunc(safety, !done, Max(safetyPhi, safety));
    }

    // Distance to Theta extent
    if (!sphere.fFullThetaSphere) {
      Real_v safetyTheta = sphere.fThetaCone.SafetyToIn<Real_v>(point);
      vecCore__MaskedAssignFunc(safety, !done, Max(safetyTheta, safety));
    }
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static void SafetyToOut(UnplacedStruct_t const &sphere,
                                                                       Vector3D<Real_v> const &point, Real_v &safety)
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    Real_v rad   = point.Mag();

    Bool_v done(false);

    Bool_v completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(sphere, point, completelyinside, completelyoutside);
    vecCore__MaskedAssignFunc(safety, completelyoutside, Real_v(-1.0));
    done |= completelyoutside;
    if (vecCore::MaskFull(done)) return;

    Bool_v isOnSurface = !completelyinside && !completelyoutside;
    vecCore__MaskedAssignFunc(safety, !done && isOnSurface, Real_v(0.0));
    done |= isOnSurface;
    if (vecCore::MaskFull(done)) return;

    // Distance to r shells
    if (sphere.fRmin) {
      Real_v safeRMin = (rad - sphere.fRmin);
      Real_v safeRMax = (sphere.fRmax - rad);
      safety          = vecCore::Blend(!done && (safeRMin < safeRMax), safeRMin, safeRMax);
    } else {
      vecCore__MaskedAssignFunc(safety, !done, (sphere.fRmax - rad));
    }

    // Distance to phi extent
    if (!sphere.fFullPhiSphere) {
      Real_v safetyPhi = sphere.fPhiWedge.SafetyToOut<Real_v>(point);
      vecCore__MaskedAssignFunc(safety, !done, Min(safetyPhi, safety));
    }

    // Distance to Theta extent
    Real_v safeTheta(0.);
    if (!sphere.fFullThetaSphere) {
      safeTheta = sphere.fThetaCone.SafetyToOut<Real_v>(point);
      vecCore__MaskedAssignFunc(safety, !done, Min(safeTheta, safety));
    }
  }

  /* This function should be called from NormalKernel, only for the
   * cases when the point is not on the surface and one want to calculate
   * the SurfaceNormal.
   *
   * Algo : Find the boundary which is closest to the point,
   * and return the normal to that boundary.
   *
   */
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

    vecCore__MaskedAssignFunc(norm, distMin == distRMax, point.Unit());
    vecCore__MaskedAssignFunc(norm, distMin == distRMin, -point.Unit());

    Vector3D<Real_v> normal1 = sphere.fPhiWedge.GetNormal1();
    Vector3D<Real_v> normal2 = sphere.fPhiWedge.GetNormal2();
    vecCore__MaskedAssignFunc(norm, distMin == distPhi1, -normal1);
    vecCore__MaskedAssignFunc(norm, distMin == distPhi2, -normal2);

    vecCore__MaskedAssignFunc(norm, distMin == distTheta1, norm + sphere.fThetaCone.GetNormal1<Real_v>(point));
    vecCore__MaskedAssignFunc(norm, distMin == distTheta2, norm + sphere.fThetaCone.GetNormal2<Real_v>(point));

    return norm;
  }

  template <typename Real_v>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE static Vector3D<Real_v> Normal(UnplacedStruct_t const &sphere,
                                                                              Vector3D<Real_v> const &point,
                                                                              typename vecCore::Mask_v<Real_v> &valid)
  {
    Vector3D<Real_v> normal(0., 0., 0.);
    normal.Set(1e-30);

    using Bool_v = vecCore::Mask_v<Real_v>;

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

    Bool_v isPointOutside(false);

    // May be required Later
    /*
    if (!ForDistanceToOut) {
      Bool_v unused(false);
      GenericKernelForContainsAndInside<Real_v, true>(sphere, point, unused, isPointOutside);
      vecCore__MaskedAssignFunc(unused || isPointOutside, ApproxSurfaceNormalKernel<Real_v>(sphere, point), &normal);
    }
    */

    Bool_v isPointInside(false);
    GenericKernelForContainsAndInside<Real_v, Bool_v, true>(sphere, point, isPointInside, isPointOutside);
    vecCore__MaskedAssignFunc(normal, isPointInside || isPointOutside,
                              ApproxSurfaceNormalKernel<Real_v>(sphere, point));

    valid = Bool_v(false);

    Real_v noSurfaces(0.);
    Bool_v isPointOnOuterRadius = SphereUtilities::IsPointOnOuterRadius<Real_v>(sphere, point);

    vecCore__MaskedAssignFunc(noSurfaces, isPointOnOuterRadius, noSurfaces + 1);
    vecCore__MaskedAssignFunc(normal, !isPointOutside && isPointOnOuterRadius, normal + (point.Unit()));

    if (sphere.fRmin) {
      Bool_v isPointOnInnerRadius = SphereUtilities::IsPointOnInnerRadius<Real_v>(sphere, point);
      vecCore__MaskedAssignFunc(noSurfaces, isPointOnInnerRadius, noSurfaces + 1);
      vecCore__MaskedAssignFunc(normal, !isPointOutside && isPointOnInnerRadius, normal - point.Unit());
    }

    if (!sphere.fFullPhiSphere) {
      Bool_v isPointOnStartPhi = SphereUtilities::IsPointOnStartPhi<Real_v>(sphere, point);
      Bool_v isPointOnEndPhi   = SphereUtilities::IsPointOnEndPhi<Real_v>(sphere, point);
      vecCore__MaskedAssignFunc(noSurfaces, isPointOnStartPhi, noSurfaces + 1);
      vecCore__MaskedAssignFunc(noSurfaces, isPointOnEndPhi, noSurfaces + 1);
      vecCore__MaskedAssignFunc(normal, !isPointOutside && isPointOnStartPhi, normal - sphere.fPhiWedge.GetNormal1());
      vecCore__MaskedAssignFunc(normal, !isPointOutside && isPointOnEndPhi, normal - sphere.fPhiWedge.GetNormal2());
    }

    if (!sphere.fFullThetaSphere) {
      Bool_v isPointOnStartTheta = SphereUtilities::IsPointOnStartTheta<Real_v>(sphere, point);
      Bool_v isPointOnEndTheta   = SphereUtilities::IsPointOnEndTheta<Real_v>(sphere, point);

      vecCore__MaskedAssignFunc(noSurfaces, isPointOnStartTheta, noSurfaces + 1);
      vecCore__MaskedAssignFunc(normal, !isPointOutside && isPointOnStartTheta,
                                normal + sphere.fThetaCone.GetNormal1<Real_v>(point));

      vecCore__MaskedAssignFunc(noSurfaces, isPointOnEndTheta, noSurfaces + 1);
      vecCore__MaskedAssignFunc(normal, !isPointOutside && isPointOnEndTheta,
                                normal + sphere.fThetaCone.GetNormal2<Real_v>(point));

      Vector3D<Real_v> tempNormal(0., 0., -1.);
      vecCore__MaskedAssignFunc(
          normal, !isPointOutside && isPointOnStartTheta && isPointOnEndTheta && (sphere.eTheta <= kPi / 2.),
          tempNormal);
      Vector3D<Real_v> tempNormal2(0., 0., 1.);
      vecCore__MaskedAssignFunc(
          normal, !isPointOutside && isPointOnStartTheta && isPointOnEndTheta && (sphere.fSTheta >= kPi / 2.),
          tempNormal2);
    }

    normal.Normalize();

    valid = (noSurfaces > Real_v(0.));

    return normal;
  }
};
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_KERNEL_sphereIMPLEMENTATION_H_
