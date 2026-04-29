/*
 * Wedge.h
 *
 *  Created on: 09.10.2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_WEDGE_EVOLUTION_H_
#define VECGEOM_VOLUMES_WEDGE_EVOLUTION_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>

namespace vecgeom {
namespace evolution {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a wedge which is represented by an angle. It
 * can be used to divide 3D spaces or to clip wedges from solids.
 * The wedge has an "inner" and "outer" side. For an angle = 180 degree, the wedge is essentially
 * an ordinary halfspace. Usually the wedge is used to cut out "phi" sections along z-direction.
 *
 * Idea: should have Unplaced and PlacedWegdes, should have specializations
 * for "PhiWegde" and which are used in symmetric
 * shapes such as tubes or spheres.
 *
 * Note: This class is meant as an auxiliary class so it is a bit outside the ordinary volume
 * hierarchy.
 *
 *       / +++++++++++
 *      / ++++++++++++
 *     / +++++++++++++
 *    / +++++ INSIDE +
 *   / +++++++++++++++
 *  / fDPhi +++++++++
 * x------------------ ( this is at angle fSPhi )
 *
 *     OUTSIDE
 *
 */
class Wedge {

private:
  Precision fSPhi = 0.;              // starting angle
  Precision fDPhi = 0.;              // delta angle representing/defining the wedge
  Vector3D<Precision> fAlongVector1; // vector along the first plane
  Vector3D<Precision> fAlongVector2; // vector aling the second plane

  Vector3D<Precision> fNormalVector1; // normal vector for first plane
  // convention is that it points inwards

  Vector3D<Precision> fNormalVector2; // normal vector for second plane
                                      // convention is that it points inwards

public:
  VECCORE_ATT_HOST_DEVICE
  Wedge(Precision angle, Precision zeroangle = 0) { Init(angle, zeroangle); }

  VECCORE_ATT_HOST_DEVICE
  Wedge() {}

  VECCORE_ATT_HOST_DEVICE
  ~Wedge() {}

  VECCORE_ATT_HOST_DEVICE
  void Init(Precision const &dphi, Precision const &sphi)
  {
    Set(dphi, sphi);
    UpdateNormals();
  }

  VECCORE_ATT_HOST_DEVICE
  void SetStartPhi(Precision const &arg) { fSPhi = arg; }

  VECCORE_ATT_HOST_DEVICE
  void SetDeltaPhi(Precision const &arg) { fDPhi = arg; }

  VECCORE_ATT_HOST_DEVICE
  void Set(Precision const &dphi, Precision const &sphi)
  {
    SetStartPhi(sphi);
    SetDeltaPhi(dphi);
  }

  VECCORE_ATT_HOST_DEVICE
  void UpdateNormals()
  {
    fAlongVector1.x() = std::cos(fSPhi);
    fAlongVector1.y() = std::sin(fSPhi);
    fAlongVector2.x() = std::cos(fSPhi + fDPhi);
    fAlongVector2.y() = std::sin(fSPhi + fDPhi);

    fNormalVector1.x() = -std::sin(fSPhi);
    fNormalVector1.y() = std::cos(fSPhi); // not the + sign
    fNormalVector2.x() = std::sin(fSPhi + fDPhi);
    fNormalVector2.y() = -std::cos(fSPhi + fDPhi); // note the - sign
  }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetAlong1() const { return fAlongVector1; }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetAlong2() const { return fAlongVector2; }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetNormal1() const { return fNormalVector1; }

  VECCORE_ATT_HOST_DEVICE
  Vector3D<Precision> GetNormal2() const { return fNormalVector2; }

  /* Function Name : GetNormal<ForStartPhi>()
   *
   * The function is the templatized version GetNormal1() and GetNormal2() function and will
   * return the normal depending upon the boolean template parameter "ForStartPhi"
   * which if passed as true, will return normal to the StartingPhi of Wedge,
   * if passed as false, will return normal to the EndingPhi of Wedge
   *
   * from user point of view the same work can be done by calling GetNormal1() and GetNormal2()
   * functions, but this implementation will be used by "IsPointOnSurfaceAndMovingOut()" function
   */
  template <bool ForStartPhi>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Precision> GetNormal() const;

  // very important:
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Contains(Vector3D<Real_v> const &point) const;

  VECCORE_ATT_HOST_DEVICE bool Contains(Vector3D<Precision> const &point) const;

  // GL note: for tubes, use of TubeImpl::PointInCyclicalSector outperformed next two methods in vector mode
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> ContainsWithBoundary(Vector3D<Real_v> const &point) const;

  VECCORE_ATT_HOST_DEVICE bool ContainsWithBoundary(Vector3D<Precision> const &point) const;

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> ContainsWithoutBoundary(Vector3D<Real_v> const &point) const;

  VECCORE_ATT_HOST_DEVICE bool ContainsWithoutBoundary(Vector3D<Precision> const &point) const;

  template <typename Real_v, typename Inside_t>
  VECCORE_ATT_HOST_DEVICE Inside_t Inside(Vector3D<Real_v> const &point) const;

  template <typename Inside_t>
  VECCORE_ATT_HOST_DEVICE Inside_t Inside(Vector3D<Precision> const &point) const;

  // static function determining if input points are on a plane surface which is part of a wedge
  // ( given by along and normal )
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE static vecCore::Mask_v<Real_v> IsOnSurfaceGeneric(Vector3D<Precision> const &alongVector,
                                                                            Vector3D<Precision> const &normalVector,
                                                                            Vector3D<Real_v> const &point);

  VECCORE_ATT_HOST_DEVICE static bool IsOnSurfaceGeneric(Vector3D<Precision> const &alongVector,
                                                         Vector3D<Precision> const &normalVector,
                                                         Vector3D<Precision> const &point);

  /* Function Name :  IsOnSurfaceGeneric<Real_v, ForStartPhi>()
   *
   * This version of IsOnSurfaceGeneric is having one more template parameter of type boolean,
   * which if passed as true, will check whether the point is on StartingPhi Surface of Wedge,
   * and if passed as false, will check whether the point is on EndingPhi Surface of Wedge
   *
   * this implementation will be used by "IsPointOnSurfaceAndMovingOut()" function.
   */
  template <typename Real_v, bool ForStartPhi>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsOnSurfaceGeneric(
      Vector3D<Real_v> const &point) const;

  /* Function Name : IsPointOnSurfaceAndMovingOut<Real_v, ForStartPhi, MovingOut>
   *
   * This function is written to check if the point is on surface and is moving inside or outside.
   * This will basically be used by "DistanceToInKernel()" and "DistanceToOutKernel()" of the shapes,
   * which uses wedge.
   *
   * It contains two extra template boolean parameters "ForStartPhi" and "MovingOut",
   * So call like "IsPointOnSurfaceAndMovingOut<Real_v,true,true>" will check whether the points is on
   * the StartingPhi Surface of wedge and moving outside.
   *
   * So overall can be called in following four combinations
   * 1) "IsPointOnSurfaceAndMovingOut<Real_v,true,true>" : Point on StartingPhi surface of wedge and moving OUT
   * 2) "IsPointOnSurfaceAndMovingOut<Real_v,true,false>" : Point on StartingPhi surface of wedge and moving IN
   * 3) "IsPointOnSurfaceAndMovingOut<Real_v,false,true>" : Point on EndingPhi surface of wedge and moving OUT
   * 2) "IsPointOnSurfaceAndMovingOut<Real_v,false,false>" : Point on EndingPhi surface of wedge and moving IN
   *
   * Very useful for DistanceToIn and DistanceToOut.
   */
  template <typename Real_v, bool ForStartPhi, bool MovingOut>
  VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsPointOnSurfaceAndMovingOut(
      Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const;

  VECCORE_ATT_HOST_DEVICE
  bool IsOnSurface1(Vector3D<Precision> const &point) const
  {
    return Wedge::IsOnSurfaceGeneric(fAlongVector1, fNormalVector1, point);
  }

  VECCORE_ATT_HOST_DEVICE
  bool IsOnSurface2(Vector3D<Precision> const &point) const
  {
    return Wedge::IsOnSurfaceGeneric(fAlongVector2, fNormalVector2, point);
  }

  /**
   * estimate of the smallest distance to the Wedge boundary when
   * the point is located outside the Wedge
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v SafetyToIn(Vector3D<Real_v> const &point) const;

  VECCORE_ATT_HOST_DEVICE Precision SafetyToIn(Vector3D<Precision> const &point) const;

  /**
   * estimate of the smallest distance to the Wedge boundary when
   * the point is located inside the Wedge ( within the defining phi angle )
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v SafetyToOut(Vector3D<Real_v> const &point) const;

  VECCORE_ATT_HOST_DEVICE Precision SafetyToOut(Vector3D<Precision> const &point) const;

  /**
   * estimate of the distance to the Wedge boundary with given direction
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE void DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                            Real_v &distWedge1, Real_v &distWedge2) const;

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE void DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                             Real_v &distWedge1, Real_v &distWedge2) const;

  // this could be useful to be public such that other shapes can directly
  // use completelyinside + completelyoutside

  template <typename Real_v, bool ForInside>
  VECCORE_ATT_HOST_DEVICE void GenericKernelForContainsAndInside(
      Vector3D<Real_v> const &localPoint, typename vecCore::Mask_v<Real_v> &completelyinside,
      typename vecCore::Mask_v<Real_v> &completelyoutside) const;

  template <bool ForInside>
  VECCORE_ATT_HOST_DEVICE void GenericKernelForContainsAndInside(Vector3D<Precision> const &localPoint,
                                                                 bool &completelyinside, bool &completelyoutside) const;

}; // end of class Wedge

template <bool ForStartPhi>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Vector3D<Precision> Wedge::GetNormal() const
{
  if (ForStartPhi)
    return fNormalVector1;
  else
    return fNormalVector2;
}

template <typename Real_v, bool ForStartPhi, bool MovingOut>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Wedge::IsPointOnSurfaceAndMovingOut(
    Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
{

  if (MovingOut)
    return IsOnSurfaceGeneric<Real_v, ForStartPhi>(point) &&
           (dir.Dot(-GetNormal<ForStartPhi>()) > Real_v(0.005 * kHalfTolerance));
  else
    return IsOnSurfaceGeneric<Real_v, ForStartPhi>(point) &&
           (dir.Dot(-GetNormal<ForStartPhi>()) < Real_v(0.005 * kHalfTolerance));
}

template <typename Real_v, bool ForStartPhi>
VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Wedge::IsOnSurfaceGeneric(
    Vector3D<Real_v> const &point) const
{

  if (ForStartPhi)
    return IsOnSurfaceGeneric<Real_v>(fAlongVector1, fNormalVector1, point);
  else
    return IsOnSurfaceGeneric<Real_v>(fAlongVector2, fNormalVector2, point);
}

template <typename Real_v, typename Inside_t>
VECCORE_ATT_HOST_DEVICE Inside_t Wedge::Inside(Vector3D<Real_v> const &point) const
{
  using Bool_v       = vecCore::Mask_v<Real_v>;
  using InsideBool_v = vecCore::Mask_v<Inside_t>;
  Bool_v completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Real_v, true>(point, completelyinside, completelyoutside);
  Inside_t inside(EInside::kSurface);
  vecCore::MaskedAssign(inside, (InsideBool_v)completelyoutside, Inside_t(EInside::kOutside));
  vecCore::MaskedAssign(inside, (InsideBool_v)completelyinside, Inside_t(EInside::kInside));
  return inside;
}

template <typename Inside_t>
VECCORE_ATT_HOST_DEVICE Inside_t Wedge::Inside(Vector3D<Precision> const &point) const
{
  bool completelyinside  = false;
  bool completelyoutside = false;
  GenericKernelForContainsAndInside<true>(point, completelyinside, completelyoutside);
  Inside_t inside(EInside::kSurface);
  if (completelyoutside) inside = EInside::kOutside;
  if (completelyinside) inside = EInside::kInside;
  return inside;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Wedge::ContainsWithBoundary(
    Vector3D<Real_v> const &point) const
{
  typedef typename vecCore::Mask_v<Real_v> Bool_v;
  Bool_v completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Real_v, true>(point, completelyinside, completelyoutside);
  return !completelyoutside;
}

VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool Wedge::ContainsWithBoundary(Vector3D<Precision> const &point) const
{
  bool completelyinside  = false;
  bool completelyoutside = false;
  // Boundary-inclusive contains must use the same on-surface handling as the
  // vector overload.
  GenericKernelForContainsAndInside<true>(point, completelyinside, completelyoutside);
  return !completelyoutside;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Wedge::ContainsWithoutBoundary(
    Vector3D<Real_v> const &point) const
{
  typedef typename vecCore::Mask_v<Real_v> Bool_v;
  Bool_v completelyinside, completelyoutside;
  GenericKernelForContainsAndInside<Real_v, true>(point, completelyinside, completelyoutside);
  return completelyinside;
}

VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool Wedge::ContainsWithoutBoundary(
    Vector3D<Precision> const &point) const
{
  bool completelyinside  = false;
  bool completelyoutside = false;
  GenericKernelForContainsAndInside<true>(point, completelyinside, completelyoutside);
  return completelyinside;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Wedge::Contains(Vector3D<Real_v> const &point) const
{
  typedef typename vecCore::Mask_v<Real_v> Bool_v;
  Bool_v unused(false);
  Bool_v outside(false);
  GenericKernelForContainsAndInside<Real_v, false>(point, unused, outside);
  return !outside;
}

VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool Wedge::Contains(Vector3D<Precision> const &point) const
{
  bool unused  = false;
  bool outside = false;
  GenericKernelForContainsAndInside<false>(point, unused, outside);
  return !outside;
}

// Implementation follows
template <typename Real_v, bool ForInside>
VECCORE_ATT_HOST_DEVICE void Wedge::GenericKernelForContainsAndInside(
    Vector3D<Real_v> const &localPoint, typename vecCore::Mask_v<Real_v> &completelyinside,
    typename vecCore::Mask_v<Real_v> &completelyoutside) const
{

  // this part of the code assumes some symmetry knowledge and is currently only
  // correct for a PhiWedge assumed to be aligned along the z-axis.
  Real_v x(localPoint.x());
  Real_v y(localPoint.y());
  Real_v startx(fAlongVector1.x());
  Real_v starty(fAlongVector1.y());
  Real_v endx(fAlongVector2.x());
  Real_v endy(fAlongVector2.y());

  Real_v startCheck = (-x * starty + y * startx);
  Real_v endCheck   = (-endx * y + endy * x);

  completelyoutside = startCheck < Real_v(0.);
  if (fDPhi < kPi)
    completelyoutside |= endCheck < Real_v(0.);
  else
    completelyoutside &= endCheck < Real_v(0.);
  if (ForInside) {
    // TODO: see if the compiler optimizes across these function calls since
    // a couple of multiplications inside IsOnSurfaceGeneric are already done previously
    typename vecCore::Mask_v<Real_v> onSurface =
        Wedge::IsOnSurfaceGeneric<Real_v>(fAlongVector1, fNormalVector1, localPoint) ||
        Wedge::IsOnSurfaceGeneric<Real_v>(fAlongVector2, fNormalVector2, localPoint);
    completelyoutside &= !onSurface;
    completelyinside = !onSurface && !completelyoutside;
  }
}

template <bool ForInside>
VECCORE_ATT_HOST_DEVICE void Wedge::GenericKernelForContainsAndInside(Vector3D<Precision> const &localPoint,
                                                                      bool &completelyinside,
                                                                      bool &completelyoutside) const
{
  Precision x(localPoint.x());
  Precision y(localPoint.y());
  Precision startx(fAlongVector1.x());
  Precision starty(fAlongVector1.y());
  Precision endx(fAlongVector2.x());
  Precision endy(fAlongVector2.y());

  Precision startCheck = (-x * starty + y * startx);
  Precision endCheck   = (-endx * y + endy * x);

  completelyoutside = startCheck < Precision(0.);
  if (fDPhi < kPi)
    completelyoutside |= endCheck < Precision(0.);
  else
    completelyoutside &= endCheck < Precision(0.);

  if (!ForInside) return;

  const bool nearStartSurface = Abs(startCheck) < Precision(kTolerance);
  const bool nearEndSurface   = Abs(endCheck) < Precision(kTolerance);
  if (!nearStartSurface && !nearEndSurface) {
    completelyinside = !completelyoutside;
    return;
  }

  bool onSurface = false;
  if (nearStartSurface) {
    const Precision startAlong = x * startx + y * starty;
    onSurface                  = startAlong >= Precision(0.);
  }
  if (!onSurface && nearEndSurface) {
    const Precision endAlong = x * endx + y * endy;
    onSurface                = endAlong >= Precision(0.);
  }

  completelyoutside &= !onSurface;
  completelyinside = !onSurface && !completelyoutside;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Wedge::IsOnSurfaceGeneric(
    Vector3D<Precision> const &alongVector, Vector3D<Precision> const &normalVector, Vector3D<Real_v> const &point)
{
  // on right side of half plane ??
  typedef typename vecCore::Mask_v<Real_v> Bool_v;
  Bool_v condition1 = alongVector.x() * point.x() + alongVector.y() * point.y() >= Real_v(0.);
  if (vecCore::MaskEmpty(condition1)) return Bool_v(false);
  // within the right distance to the plane ??
  Bool_v condition2 = Abs(normalVector.x() * point.x() + normalVector.y() * point.y()) < kTolerance;
  return condition1 && condition2;
}

VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE bool Wedge::IsOnSurfaceGeneric(
    Vector3D<Precision> const &alongVector, Vector3D<Precision> const &normalVector, Vector3D<Precision> const &point)
{
  bool condition1 = alongVector.x() * point.x() + alongVector.y() * point.y() >= Precision(0.);
  if (!condition1) return false;
  bool condition2 = Abs(normalVector.x() * point.x() + normalVector.y() * point.y()) < kTolerance;
  return condition2;
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v Wedge::SafetyToOut(Vector3D<Real_v> const &point) const
{

  // algorithm: calculate projections to both planes
  // return minimum / maximum depending on fAngle < PI or not

  // assuming that we have z wedge and the planes pass through the origin
  Real_v dist1 = point.x() * fNormalVector1.x() + point.y() * fNormalVector1.y();
  Real_v dist2 = point.x() * fNormalVector2.x() + point.y() * fNormalVector2.y();

  if (fDPhi < kPi) {
    return Min(dist1, dist2);
  } else {
    return Max(dist1, dist2);
  }
}

VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Precision Wedge::SafetyToOut(Vector3D<Precision> const &point) const
{
  Precision dist1 = point.x() * fNormalVector1.x() + point.y() * fNormalVector1.y();
  Precision dist2 = point.x() * fNormalVector2.x() + point.y() * fNormalVector2.y();

  if (fDPhi < kPi) {
    return Min(dist1, dist2);
  } else {
    return Max(dist1, dist2);
  }
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE Real_v Wedge::SafetyToIn(Vector3D<Real_v> const &point) const
{

  // algorithm: calculate projections to both planes
  // return maximum / minimum depending on fAngle < PI or not
  // assuming that we have z wedge and the planes pass through the origin

  // actually we

  Real_v dist1 = point.x() * fNormalVector1.x() + point.y() * fNormalVector1.y();
  Real_v dist2 = point.x() * fNormalVector2.x() + point.y() * fNormalVector2.y();

  if (fDPhi < kPi) {
    return Max(-1 * dist1, -1 * dist2);
  } else {
    return Min(-1 * dist1, -1 * dist2);
  }
}

VECGEOM_FORCE_INLINE VECCORE_ATT_HOST_DEVICE Precision Wedge::SafetyToIn(Vector3D<Precision> const &point) const
{
  Precision dist1 = point.x() * fNormalVector1.x() + point.y() * fNormalVector1.y();
  Precision dist2 = point.x() * fNormalVector2.x() + point.y() * fNormalVector2.y();

  if (fDPhi < kPi) {
    return Max(-dist1, -dist2);
  } else {
    return Min(-dist1, -dist2);
  }
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void Wedge::DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                                 Real_v &distWedge1, Real_v &distWedge2) const
{
  using Bool_v = vecCore::Mask_v<Real_v>;
  // algorithm::first calculate projections of direction to both planes,
  // then calculate real distance along given direction,
  // distance can be negative

  distWedge1 = kInfLength;
  distWedge2 = kInfLength;

  Real_v comp1 = dir.x() * fNormalVector1.x() + dir.y() * fNormalVector1.y();
  Real_v comp2 = dir.x() * fNormalVector2.x() + dir.y() * fNormalVector2.y();

  Bool_v cmp1 = comp1 > Real_v(0.);
  if (!vecCore::MaskEmpty(cmp1)) {
    Real_v tmp = -(point.x() * fNormalVector1.x() + point.y() * fNormalVector1.y()) / comp1;
    vecCore::MaskedAssign(tmp, tmp > Real_v(-kTolerance) && tmp < Real_v(0.), Real_v(0.));
    vecCore::MaskedAssign(distWedge1, cmp1 && tmp >= Real_v(0.), tmp);
  }
  Bool_v cmp2 = comp2 > Real_v(0.);
  if (!vecCore::MaskEmpty(cmp2)) {
    Real_v tmp = -(point.x() * fNormalVector2.x() + point.y() * fNormalVector2.y()) / comp2;
    vecCore::MaskedAssign(tmp, tmp > Real_v(-kTolerance) && tmp < Real_v(0.), Real_v(0.));
    vecCore::MaskedAssign(distWedge2, cmp2 && tmp >= Real_v(0.), tmp);
  }
}

template <typename Real_v>
VECCORE_ATT_HOST_DEVICE void Wedge::DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                                  Real_v &distWedge1, Real_v &distWedge2) const
{

  using Bool_v = vecCore::Mask_v<Real_v>;

  // algorithm::first calculate projections of direction to both planes,
  // then calculate real distance along given direction,
  // distance can be negative

  Real_v comp1 = dir.x() * fNormalVector1.x() + dir.y() * fNormalVector1.y();
  Real_v comp2 = dir.x() * fNormalVector2.x() + dir.y() * fNormalVector2.y();

  // std::cerr << "c1 " << comp1 << "\n";
  // std::cerr << "c2 " << comp2 << "\n";
  distWedge1 = kInfLength;
  distWedge2 = kInfLength;

  Bool_v cmp1 = comp1 < Real_v(0.);
  if (!vecCore::MaskEmpty(cmp1)) {
    Real_v tmp = -(point.x() * fNormalVector1.x() + point.y() * fNormalVector1.y()) / comp1;
    vecCore::MaskedAssign(tmp, tmp > Real_v(-kTolerance) && tmp < Real_v(0.), Real_v(0.));
    vecCore::MaskedAssign(distWedge1, cmp1 && tmp > Real_v(0.), tmp);
  }

  Bool_v cmp2 = comp2 < Real_v(0.);
  if (!vecCore::MaskEmpty(cmp2)) {
    Real_v tmp = -(point.x() * fNormalVector2.x() + point.y() * fNormalVector2.y()) / comp2;
    vecCore::MaskedAssign(tmp, tmp > Real_v(-kTolerance) && tmp < Real_v(0.), Real_v(0.));
    vecCore::MaskedAssign(distWedge2, cmp2 && tmp > Real_v(0.), tmp);
  }
}
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace evolution
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_WEDGE_H_ */
