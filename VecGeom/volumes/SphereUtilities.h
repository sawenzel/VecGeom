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
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
T sqr(T x)
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
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnInnerRadius(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{
  auto mag2 = point.Mag2();
  return mag2 <= MakePlusTolerantSquare<true>(unplaced.fRmin) &&
         mag2 >= MakeMinusTolerantSquare<true>(unplaced.fRmin);
}

template <class Real_v>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnOuterRadius(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{
  auto mag2 = point.Mag2();
  return mag2 <= MakePlusTolerantSquare<true>(unplaced.fRmax) &&
         mag2 >= MakeMinusTolerantSquare<true>(unplaced.fRmax);
}

template <class Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnStartPhi(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{

  return unplaced.fPhiWedge.IsOnSurfaceGeneric<Real_v>(unplaced.fPhiWedge.GetAlong1(), unplaced.fPhiWedge.GetNormal1(),
                                                       point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnEndPhi(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{

  return unplaced.fPhiWedge.IsOnSurfaceGeneric<Real_v>(unplaced.fPhiWedge.GetAlong2(), unplaced.fPhiWedge.GetNormal2(),
                                                       point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnStartTheta(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{

  return unplaced.fThetaCone.IsOnSurfaceGeneric<Real_v, true>(point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnEndTheta(UnplacedStruct_t const &unplaced, Vector3D<Real_v> const &point)
{

  return unplaced.fThetaCone.IsOnSurfaceGeneric<Real_v, false>(point);
}

template <class Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsCompletelyOutside(UnplacedStruct_t const &unplaced,
                                                     Vector3D<Real_v> const &localPoint)
{

  using Bool_v              = vecCore::Mask_v<Real_v>;
  Real_v rad                = localPoint.Mag();
  Bool_v outsideRadiusRange = (rad > (unplaced.fRmax + kTolerance)) || (rad < (unplaced.fRmin - kTolerance));
  Bool_v outsidePhiRange(false), insidePhiRange(false);
  unplaced.fPhiWedge.GenericKernelForContainsAndInside<Real_v, true>(localPoint, insidePhiRange, outsidePhiRange);
  Bool_v outsideThetaRange = unplaced.fThetaCone.IsCompletelyOutside<Real_v>(localPoint);
  Bool_v completelyoutside = outsideRadiusRange || outsidePhiRange || outsideThetaRange;
  return completelyoutside;
}

template <class Real_v>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsCompletelyInside(UnplacedStruct_t const &unplaced,
                                                    Vector3D<Real_v> const &localPoint)
{
  using Bool_v             = vecCore::Mask_v<Real_v>;
  Real_v rad               = localPoint.Mag();
  Bool_v insideRadiusRange = (rad < (unplaced.fRmax - kTolerance)) && (rad > (unplaced.fRmin + kTolerance));
  Bool_v outsidePhiRange(false), insidePhiRange(false);
  unplaced.fPhiWedge.GenericKernelForContainsAndInside<Real_v, true>(localPoint, insidePhiRange, outsidePhiRange);
  Bool_v insideThetaRange = unplaced.fThetaCone.IsCompletelyInside<Real_v>(localPoint);
  Bool_v completelyinside = insideRadiusRange && insidePhiRange && insideThetaRange;
  return completelyinside;
}

template <class Real_v, bool ForInnerRadius, bool MovingOut>
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnRadialSurfaceAndMovingOut(UnplacedStruct_t const &unplaced,
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

template <class Real_v, bool MovingOut>
VECGEOM_FORCE_INLINE
VECCORE_ATT_HOST_DEVICE
typename vecCore::Mask_v<Real_v> IsPointOnSurfaceAndMovingOut(UnplacedStruct_t const &unplaced,
                                                              Vector3D<Real_v> const &point,
                                                              Vector3D<Real_v> const &dir)
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  Bool_v tempOuterRad = IsPointOnRadialSurfaceAndMovingOut<Real_v, false, MovingOut>(unplaced, point, dir);
  Bool_v tempInnerRad(false), tempStartPhi(false), tempEndPhi(false), tempStartTheta(false), tempEndTheta(false);
  if (unplaced.fRmin) tempInnerRad = IsPointOnRadialSurfaceAndMovingOut<Real_v, true, MovingOut>(unplaced, point, dir);
  if (unplaced.fDPhi < (kTwoPi - kHalfTolerance)) {
    tempStartPhi = unplaced.fPhiWedge.IsPointOnSurfaceAndMovingOut<Real_v, true, MovingOut>(point, dir);
    tempEndPhi   = unplaced.fPhiWedge.IsPointOnSurfaceAndMovingOut<Real_v, false, MovingOut>(point, dir);
  }
  if (unplaced.fDTheta < (kPi - kHalfTolerance)) {
    tempStartTheta = unplaced.fThetaCone.IsPointOnSurfaceAndMovingOut<Real_v, true, MovingOut>(point, dir);
    tempEndTheta   = unplaced.fThetaCone.IsPointOnSurfaceAndMovingOut<Real_v, false, MovingOut>(point, dir);
  }

  Bool_v isPointOnSurfaceAndMovingOut =
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
