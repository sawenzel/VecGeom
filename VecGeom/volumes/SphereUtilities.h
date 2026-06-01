#ifndef VECGEOM_VOLUMES_SPHEREUTILITIES_H_
#define VECGEOM_VOLUMES_SPHEREUTILITIES_H_

#include "VecGeom/base/Global.h"

#ifndef VECCORE_CUDA
#include "VecGeom/base/RNG.h"
#endif

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/Wedge_Evolution.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/SphereStruct.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <cstdio>

namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

class UnplacedSphere;
template <typename T>
struct SphereStruct;

template <typename T>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE T sqr(T x)
{
  return x * x;
}

#ifndef VECCORE_CUDA
// Generate radius in annular ring according to uniform area
template <typename T>
inline T GetRadiusInRing(T rmin, T rmax)
{
  if (rmin == rmax) return rmin;

  T rng(RNG::Instance().uniform(0.0, 1.0));

  if (rmin <= T(0.0)) return rmax * Sqrt(rng);

  T rmin2 = rmin * rmin;
  T rmax2 = rmax * rmax;

  return Sqrt(rng * (rmax2 - rmin2) + rmin2);
}
#endif

namespace SphereUtilities {
using UnplacedStruct_t = SphereStruct<Precision>;

template <class Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointOnInnerRadius(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{
  auto mag2 = point.Mag2();
  return mag2 <= MakePlusTolerantSquare<true>(unplaced.fRmin) && mag2 >= MakeMinusTolerantSquare<true>(unplaced.fRmin);
}

template <class Real_v>
VECCORE_ATT_HOST_DEVICE bool IsPointOnOuterRadius(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{
  auto mag2 = point.Mag2();
  return mag2 <= MakePlusTolerantSquare<true>(unplaced.fRmax) && mag2 >= MakeMinusTolerantSquare<true>(unplaced.fRmax);
}

template <class Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsPointOnStartPhi(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point)
{

  return unplaced.fPhiWedge.IsOnSurfaceGeneric<Real_v>(unplaced.fPhiWedge.GetAlong1(), unplaced.fPhiWedge.GetNormal1(),
                                                       point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsPointOnEndPhi(UnplacedStruct_t const &unplaced,
                                                                  Vector3D<Real_v> const &point)
{

  return unplaced.fPhiWedge.IsOnSurfaceGeneric<Real_v>(unplaced.fPhiWedge.GetAlong2(), unplaced.fPhiWedge.GetNormal2(),
                                                       point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsPointOnStartTheta(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &point)
{

  return unplaced.fThetaCone.IsOnSurfaceGeneric<Real_v, true>(point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsPointOnEndTheta(UnplacedStruct_t const &unplaced,
                                                                    Vector3D<Real_v> const &point)
{

  return unplaced.fThetaCone.IsOnSurfaceGeneric<Real_v, false>(point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsCompletelyOutside(UnplacedStruct_t const &unplaced,
                                                                      Vector3D<Real_v> const &localPoint)
{

  Real_v rad              = localPoint.Mag();
  bool outsideRadiusRange = (rad > (unplaced.fRmax + kTolerance)) || (rad < (unplaced.fRmin - kTolerance));
  bool outsidePhiRange = false, insidePhiRange = false;
  unplaced.fPhiWedge.GenericKernelForContainsAndInside<Real_v, true>(localPoint, insidePhiRange, outsidePhiRange);
  bool outsideThetaRange = unplaced.fThetaCone.IsCompletelyOutside<Real_v>(localPoint);
  bool completelyoutside = outsideRadiusRange || outsidePhiRange || outsideThetaRange;
  return completelyoutside;
}

template <class Real_v>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsCompletelyInside(UnplacedStruct_t const &unplaced,
                                                                     Vector3D<Real_v> const &localPoint)
{
  Real_v rad             = localPoint.Mag();
  bool insideRadiusRange = (rad < (unplaced.fRmax - kTolerance)) && (rad > (unplaced.fRmin + kTolerance));
  bool outsidePhiRange = false, insidePhiRange = false;
  unplaced.fPhiWedge.GenericKernelForContainsAndInside<Real_v, true>(localPoint, insidePhiRange, outsidePhiRange);
  bool insideThetaRange = unplaced.fThetaCone.IsCompletelyInside<Real_v>(localPoint);
  bool completelyinside = insideRadiusRange && insidePhiRange && insideThetaRange;
  return completelyinside;
}

template <class Real_v, bool ForInnerRadius, bool MovingOut>
VECCORE_ATT_HOST_DEVICE bool IsPointOnRadialSurfaceAndMovingOut(UnplacedStruct_t const &unplaced,
                                                                Vector3D<Real_v> const &point,
                                                                Vector3D<Real_v> const &dir)
{
  // Rays from rmax+tolerance or rmin-tolerance can be moving out even if not going fully "backward"
  if (MovingOut) {
    if (ForInnerRadius) {
      return IsPointOnInnerRadius<Real_v>(unplaced, point) && (dir.Dot(-point) > Real_v(kSqrtTolerance));
    } else {
      return IsPointOnOuterRadius<Real_v>(unplaced, point) && (dir.Dot(point) > Real_v(kSqrtTolerance));
    }
  } else {
    if (ForInnerRadius) {
      return IsPointOnInnerRadius<Real_v>(unplaced, point) && (dir.Dot(-point) < Real_v(-kSqrtTolerance));
    } else
      return IsPointOnOuterRadius<Real_v>(unplaced, point) && (dir.Dot(point) < Real_v(-kSqrtTolerance));
  }
}

template <class Real_v, bool ForStartTheta>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Real_v ThetaConeImplicitMotion(UnplacedStruct_t const &unplaced,
                                                                            Vector3D<Real_v> const &point,
                                                                            Vector3D<Real_v> const &dir)
{
  auto theta = ForStartTheta ? unplaced.fSTheta : unplaced.eTheta;
  if (Abs(theta - kHalfPi) <= kTolerance) return -dir.z();

  auto tanTheta2 =
      ForStartTheta ? Real_v(unplaced.fThetaCone.GetTanSTheta2()) : Real_v(unplaced.fThetaCone.GetTanETheta2());
  return point.x() * dir.x() + point.y() * dir.y() - tanTheta2 * point.z() * dir.z();
}

template <class Real_v, bool ForStartTheta, bool MovingOut>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsThetaConeMotion(UnplacedStruct_t const &unplaced,
                                                                    Real_v const &motion)
{
  auto theta = ForStartTheta ? unplaced.fSTheta : unplaced.eTheta;
  if (ForStartTheta) {
    if (MovingOut) return theta <= kHalfPi ? motion < Real_v(-kSqrtTolerance) : motion > Real_v(kSqrtTolerance);
    return theta <= kHalfPi ? motion > Real_v(kSqrtTolerance) : motion < Real_v(-kSqrtTolerance);
  }

  if (MovingOut) return theta <= kHalfPi ? motion > Real_v(kSqrtTolerance) : motion < Real_v(-kSqrtTolerance);
  return theta <= kHalfPi ? motion < Real_v(-kSqrtTolerance) : motion > Real_v(kSqrtTolerance);
}

template <class Real_v, bool ForStartTheta, bool MovingOut>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsPointOnThetaSurfaceAndMovingOut(UnplacedStruct_t const &unplaced,
                                                                                    Vector3D<Real_v> const &point,
                                                                                    Vector3D<Real_v> const &dir)
{
  if (!unplaced.fThetaCone.IsOnSurfaceGeneric<Real_v, ForStartTheta>(point)) return false;

  auto motion = ThetaConeImplicitMotion<Real_v, ForStartTheta>(unplaced, point, dir);
  return IsThetaConeMotion<Real_v, ForStartTheta, MovingOut>(unplaced, motion);
}

template <class Real_v, bool MovingOut>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool IsPointOnSurfaceAndMovingOut(UnplacedStruct_t const &unplaced,
                                                                               Vector3D<Real_v> const &point,
                                                                               Vector3D<Real_v> const &dir)
{

  bool tempOuterRad = IsPointOnRadialSurfaceAndMovingOut<Real_v, false, MovingOut>(unplaced, point, dir);
  bool tempInnerRad = false, tempStartPhi = false, tempEndPhi = false, tempStartTheta = false, tempEndTheta = false;
  if (unplaced.fRmin) tempInnerRad = IsPointOnRadialSurfaceAndMovingOut<Real_v, true, MovingOut>(unplaced, point, dir);
  if (unplaced.fDPhi < (kTwoPi - kHalfTolerance)) {
    tempStartPhi = unplaced.fPhiWedge.IsPointOnSurfaceAndMovingOut<Real_v, true, MovingOut>(point, dir);
    tempEndPhi   = unplaced.fPhiWedge.IsPointOnSurfaceAndMovingOut<Real_v, false, MovingOut>(point, dir);
  }
  if (unplaced.fDTheta < (kPi - kHalfTolerance)) {
    tempStartTheta = IsPointOnThetaSurfaceAndMovingOut<Real_v, true, MovingOut>(unplaced, point, dir);
    tempEndTheta   = IsPointOnThetaSurfaceAndMovingOut<Real_v, false, MovingOut>(unplaced, point, dir);
  }

  bool isPointOnSurfaceAndMovingOut =
      ((tempOuterRad || tempInnerRad) && unplaced.fPhiWedge.Contains<Real_v>(point) &&
       unplaced.fThetaCone.Contains<Real_v>(point)) ||
      ((tempStartPhi || tempEndPhi) && (point.Mag2() >= unplaced.fRmin * unplaced.fRmin) &&
       (point.Mag2() <= unplaced.fRmax * unplaced.fRmax) && unplaced.fThetaCone.Contains<Real_v>(point)) ||
      ((tempStartTheta || tempEndTheta) && (point.Mag2() >= unplaced.fRmin * unplaced.fRmin) &&
       (point.Mag2() <= unplaced.fRmax * unplaced.fRmax) && unplaced.fPhiWedge.Contains<Real_v>(point));

  return isPointOnSurfaceAndMovingOut;
}

} // namespace SphereUtilities
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif // VECGEOM_VOLUMES_SPHEREUTILITIES_H_
