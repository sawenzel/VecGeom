/*
 * ThetaCone.h
 *
 *      Author: Raman Sehgal
 */

#ifndef VECGEOM_VOLUMES_THETACONE_H_
#define VECGEOM_VOLUMES_THETACONE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <VecCore/VecCore>
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a ThetaCone (basically a double cone) which is represented by an angle theta ( 0 < theta < Pi).
 *It
 *
 * The ThetaCone has an "startTheta" and "endTheta" angle. For an angle = 90 degree, the ThetaCone is essentially
 * XY plane with circular boundary. Usually the ThetaCone is used to cut out "theta" sections along z-direction.
 *
 *
 * Note: This class is meant as an auxiliary class so it is a bit outside the ordinary volume
 * hierarchy.
 *
 *      \ ++++ /
 *       \    /
 *        \  /
 *         \/
 *         /\
 *        /  \
 *       /    \
 *      / ++++ \
 *
 *DistanceToIn and DistanceToOut provides distances with the First and Second ThetaCone in "distThetaCone1" and
 *"distThetaCone2" reference variables.
 *Reference bool variable "intsect1" and "intsect2" is used to detect the real intersection cone, i.e. whether the point
 *really intersects with a ThetaCone or not.
 */
class ThetaCone {

private:
  Precision fSTheta; // starting angle
  Precision fDTheta; // delta angle
  Precision kAngTolerance;
  Precision halfAngTolerance;
  Precision fETheta; // ending angle
  Precision tanSTheta;
  Precision tanETheta;
  Precision tanBisector;
  Precision slope1, slope2;
  Precision tanSTheta2;
  Precision tanETheta2;

public:
  VECCORE_ATT_HOST_DEVICE
  ThetaCone() {}

  VECCORE_ATT_HOST_DEVICE
  ThetaCone(Precision sTheta, Precision dTheta) : fSTheta(sTheta), fDTheta(dTheta), kAngTolerance(kTolerance)
  {

    // initialize angles
    fETheta               = fSTheta + fDTheta;
    halfAngTolerance      = (0.5 * kAngTolerance);
    Precision tempfSTheta = fSTheta;
    Precision tempfETheta = fETheta;

    if (fSTheta > kPi / 2) tempfSTheta = kPi - fSTheta;
    if (fETheta > kPi / 2) tempfETheta = kPi - fETheta;

    tanSTheta   = tan(tempfSTheta);
    tanSTheta2  = tanSTheta * tanSTheta;
    tanETheta   = tan(tempfETheta);
    tanETheta2  = tanETheta * tanETheta;
    tanBisector = tan(tempfSTheta + (fDTheta / 2));
    if (fSTheta > kPi / 2 && fETheta > kPi / 2) tanBisector = tan(tempfSTheta - (fDTheta / 2));
    slope1 = tan(kPi / 2 - fSTheta);
    slope2 = tan(kPi / 2 - fETheta);
  }

  VECCORE_ATT_HOST_DEVICE
  ~ThetaCone() {}

  VECCORE_ATT_HOST_DEVICE
  Precision GetSlope1() const { return slope1; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetSlope2() const { return slope2; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTanSTheta2() const { return tanSTheta2; }

  VECCORE_ATT_HOST_DEVICE
  Precision GetTanETheta2() const { return tanETheta2; }

  /* Function to calculate normal at a point to the Cone formed at
   * by StartTheta.
   *
   * @inputs : Vector3D : Point at which normal needs to be calculated
   *
   * @output : Vector3D : calculated normal at the input point.
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Vector3D<Real_v> GetNormal1(Vector3D<Real_v> const &point) const
  {

    Vector3D<Real_v> normal(2. * point.x(), 2. * point.y(), -2. * tanSTheta2 * point.z());

    if (fSTheta <= kPi / 2.)
      return -normal;
    else
      return normal;
  }

  /* Function to calculate normal at a point to the Cone formed at
   *  by EndTheta.
   *
   * @inputs : Vector3D : Point at which normal needs to be calculated
   *
   * @output : Vector3D : calculated normal at the input point.
   */

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Vector3D<Real_v> GetNormal2(Vector3D<Real_v> const &point) const
  {

    Vector3D<Real_v> normal(2 * point.x(), 2 * point.y(), -2 * tanETheta2 * point.z());

    if (fETheta <= kPi / 2.)
      return normal;
    else
      return -normal;
  }

  /* Function Name : GetNormal<Real_v, ForStartTheta>()
   *
   * The function is the templatized version GetNormal1() and GetNormal2() function with one more template
   * parameter and will return the normal depending upon the boolean template parameter "ForStartTheta"
   * which if passed as true, will return normal to the StartingTheta of ThetaCone,
   * if passed as false, will return normal to the EndingTheta of ThetaCone
   *
   * from user point of view the same work can be done by calling GetNormal1() and GetNormal2()
   * functions, but this implementation will be used by "IsPointOnSurfaceAndMovingOut()" function
   */
  template <typename Real_v, bool ForStartTheta>
  VECCORE_ATT_HOST_DEVICE Vector3D<Real_v> GetNormal(Vector3D<Real_v> const &point) const
  {

    if (ForStartTheta) {

      Vector3D<Real_v> normal(point.x(), point.y(), -tanSTheta2 * point.z());

      if (fSTheta <= kHalfPi)
        return -normal;
      else
        return normal;

    } else {

      Vector3D<Real_v> normal(point.x(), point.y(), -tanETheta2 * point.z());

      if (fETheta <= kHalfPi)
        return normal;
      else
        return -normal;
    }
  }

  /* Function Name :  IsOnSurfaceGeneric<Real_v, ForStartTheta>()
   *
   * This version of IsOnSurfaceGeneric is having one more template parameter of type boolean,
   * which if passed as true, will check whether the point is on StartingTheta Surface of ThetaCone,
   * and if passed as false, will check whether the point is on EndingTheta Surface of ThetaCone
   *
   * this implementation will be used by "IsPointOnSurfaceAndMovingOut()" function.
   */
  template <typename Real_v, bool ForStartTheta>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsOnSurfaceGeneric(Vector3D<Real_v> const &point) const
  {
    Real_v rhs(0.);
    if (ForStartTheta) {
      rhs = Abs(tanSTheta * point.z());
    } else {
      rhs = Abs(tanETheta * point.z());
    }
    Real_v rho2 = point.Perp2();
    return rho2 >= MakeMinusTolerantSquare<true>(rhs) && rho2 <= MakePlusTolerantSquare<true>(rhs);
  }

  /* Function Name : IsPointOnSurfaceAndMovingOut<Real_v, ForStartTheta, MovingOut>
   *
   * This function is written to check if the point is on surface and is moving inside or outside.
   * This will basically be used by "DistanceToInKernel()" and "DistanceToOutKernel()" of the shapes,
   * which uses ThetaCone.
   *
   * It contains two extra template boolean parameters "ForStartTheta" and "MovingOut",
   * So call like "IsPointOnSurfaceAndMovingOut<Real_v,true,true>" will check whether the points is on
   * the StartingTheta Surface of Theta and moving outside.
   *
   * So overall can be called in following four combinations
   * 1) "IsPointOnSurfaceAndMovingOut<Real_v,true,true>" : Point on StartingTheta surface of ThetaCone and moving OUT
   * 2) "IsPointOnSurfaceAndMovingOut<Real_v,true,false>" : Point on StartingTheta surface of ThetaCone and moving IN
   * 3) "IsPointOnSurfaceAndMovingOut<Real_v,false,true>" : Point on EndingTheta surface of ThetaCone and moving OUT
   * 4) "IsPointOnSurfaceAndMovingOut<Real_v,false,false>" : Point on EndingTheta surface of ThetaCone and moving IN
   *
   * Very useful for DistanceToIn and DistanceToOut.
   */
  template <typename Real_v, bool ForStartTheta, bool MovingOut>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsPointOnSurfaceAndMovingOut(
      Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir) const
  {

    if (MovingOut) {
      return IsOnSurfaceGeneric<Real_v, ForStartTheta>(point) &&
             (dir.Dot(GetNormal<Real_v, ForStartTheta>(point)) > Real_v(0.));
    } else {
      return IsOnSurfaceGeneric<Real_v, ForStartTheta>(point) &&
             (dir.Dot(GetNormal<Real_v, ForStartTheta>(point)) < Real_v(0.));
    }
  }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> Contains(Vector3D<Real_v> const &point) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v unused(false);
    Bool_v outside(false);
    GenericKernelForContainsAndInside<Real_v, false>(point, unused, outside);
    return !outside;
  }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> ContainsWithBoundary(
      Vector3D<Real_v> const & /*point*/) const
  {
  }

  template <typename Real_v, typename Inside_t>
  VECCORE_ATT_HOST_DEVICE Inside_t Inside(Vector3D<Real_v> const &point) const
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

  /**
   * estimate of the smallest distance to the ThetaCone boundary when
   * the point is located outside the ThetaCone
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v SafetyToIn(Vector3D<Real_v> const &point) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v safeTheta(0.);
    Real_v pointRad = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v sfTh1    = DistanceToLine<Real_v>(slope1, pointRad, point.z());
    Real_v sfTh2    = DistanceToLine<Real_v>(slope2, pointRad, point.z());

    safeTheta   = Min(sfTh1, sfTh2);
    Bool_v done = Contains<Real_v>(point);
    vecCore__MaskedAssignFunc(safeTheta, done, Real_v(0.));
    if (vecCore::MaskFull(done)) return safeTheta;

    // Case 1 : Both cones are in Positive Z direction
    if (fSTheta < kPi / 2 + halfAngTolerance) {
      if (fETheta < kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          vecCore::MaskedAssign(safeTheta, (!done && point.z() < Real_v(0.)), sfTh2);
        }
      }

      // Case 2 : First Cone is in Positive Z direction and Second is in Negative Z direction
      if (fETheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          vecCore::MaskedAssign(safeTheta, (!done && point.z() > Real_v(0.)), sfTh1);
          vecCore::MaskedAssign(safeTheta, (!done && point.z() < Real_v(0.)), sfTh2);
        }
      }
    }

    // Case 3 : Both cones are in Negative Z direction
    if (fETheta > kPi / 2 + halfAngTolerance) {
      if (fSTheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          vecCore::MaskedAssign(safeTheta, (!done && point.z() > Real_v(0.)), sfTh1);
        }
      }
    }

    return safeTheta;
  }

  /**
   * estimate of the smallest distance to the ThetaCone boundary when
   * the point is located inside the ThetaCone ( within the defining phi angle )
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v SafetyToOut(Vector3D<Real_v> const &point) const
  {

    Real_v pointRad    = Sqrt(point.x() * point.x() + point.y() * point.y());
    Real_v bisectorRad = Abs(point.z() * tanBisector);

    vecCore::Mask<Real_v> condition(false);
    Real_v sfTh1 = DistanceToLine<Real_v>(slope1, pointRad, point.z());
    Real_v sfTh2 = DistanceToLine<Real_v>(slope2, pointRad, point.z());

    // Case 1 : Both cones are in Positive Z direction
    if (fSTheta < kPi / 2 + halfAngTolerance) {
      if (fETheta < kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          condition = (pointRad < bisectorRad) && (fSTheta != Real_v(0.));
        }
      }

      // Case 2 : First Cone is in Positive Z direction and Second is in Negative Z direction
      if (fETheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          condition = sfTh1 < sfTh2;
        }
      }
    }

    // Case 3 : Both cones are in Negative Z direction
    if (fETheta > kPi / 2 + halfAngTolerance) {
      if (fSTheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          condition = !((pointRad < bisectorRad) && (fETheta != Real_v(kPi)));
        }
      }
    }

    return vecCore::Blend(condition, sfTh1, sfTh2);
  }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE Real_v DistanceToLine(Precision const &slope, Real_v const &x, Real_v const &y) const
  {

    Real_v dist = (y - slope * x) / Sqrt(Real_v(1.) + slope * slope);
    return Abs(dist);
  }

  /**
   * estimate of the distance to the ThetaCone boundary with given direction
   */
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE void DistanceToIn(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                            Real_v &distThetaCone1, Real_v &distThetaCone2,
                                            typename vecCore::Mask_v<Real_v> &intsect1,
                                            typename vecCore::Mask_v<Real_v> &intsect2) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;
    Bool_v done(false);
    Bool_v fal(false);

    distThetaCone1 = Real_v(kInfLength);
    distThetaCone2 = Real_v(kInfLength);
    intsect1       = fal;
    intsect2       = fal;
    if (fSTheta >= fETheta) return;

    Real_v firstRoot(kInfLength), secondRoot(kInfLength);

    Real_v pDotV2d = point.x() * dir.x() + point.y() * dir.y();
    Real_v rho2    = point.x() * point.x() + point.y() * point.y();
    Real_v dirRho2 = dir.Perp2();

    if (fSTheta > 0) {
      Real_v b = pDotV2d - point.z() * dir.z() * tanSTheta2;
      // Real_v a = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;
      Real_v a  = dirRho2 - dir.z() * dir.z() * tanSTheta2;
      Real_v c  = rho2 - point.z() * point.z() * tanSTheta2;
      Real_v d2 = b * b - a * c;
      // Catch intersection with the tip of the cone even if rounded off
      vecCore__MaskedAssignFunc(d2, ((Abs(d2) < kTolerance)), Real_v(0.));
      Real_v aInv = Real_v(1.) / NonZero(a);

      if (fSTheta < kHalfPi - halfAngTolerance) {
        vecCore__MaskedAssignFunc(firstRoot, (d2 > Real_v(-kTolerance)), (-b + Sqrt(Abs(d2))) * aInv);
      } else if (fSTheta > kHalfPi + halfAngTolerance) {
        vecCore__MaskedAssignFunc(firstRoot, (d2 > Real_v(-kTolerance)), (-b - Sqrt(Abs(d2))) * aInv);
      }
      done |= (Abs(firstRoot) < Real_v(3.) * kTolerance);
      vecCore__MaskedAssignFunc(firstRoot, ((Abs(firstRoot) < Real_v(3.) * kTolerance)), Real_v(0.));
      vecCore__MaskedAssignFunc(firstRoot, (!done && (firstRoot < Real_v(0.))), InfinityLength<Real_v>());
    }

    if (fETheta < kPi) {
      Real_v b = pDotV2d - point.z() * dir.z() * tanETheta2;
      // Real_v a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2;
      Real_v a  = dirRho2 - dir.z() * dir.z() * tanETheta2;
      Real_v c  = rho2 - point.z() * point.z() * tanETheta2;
      Real_v d2 = b * b - a * c;
      // Catch intersection with the tip of the cone even if rounded off
      vecCore__MaskedAssignFunc(d2, ((Abs(d2) < kTolerance)), Real_v(0.));
      Real_v aInv = Real_v(1.) / NonZero(a);

      if (fETheta < kHalfPi - halfAngTolerance) {
        vecCore__MaskedAssignFunc(secondRoot, (d2 > Real_v(-kTolerance)), (-b - Sqrt(Abs(d2))) * aInv);
      } else if (fETheta > kHalfPi + halfAngTolerance) {
        vecCore__MaskedAssignFunc(secondRoot, (d2 > Real_v(-kTolerance)), (-b + Sqrt(Abs(d2))) * aInv);
      }
      vecCore__MaskedAssignFunc(secondRoot, (!done && (Abs(secondRoot) < Real_v(3.) * kTolerance)), Real_v(0.));
      done |= (Abs(secondRoot) < Real_v(3.) * kTolerance);
      vecCore__MaskedAssignFunc(secondRoot, !done && (secondRoot < Real_v(0.)), InfinityLength<Real_v>());
    }

    distThetaCone1 = firstRoot;
    distThetaCone2 = secondRoot;
    intsect1       = distThetaCone1 != kInfLength;
    intsect2       = distThetaCone2 != kInfLength;

    Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
    Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

    if (fSTheta < kHalfPi - halfAngTolerance)
      intsect1 &= zOfIntSecPtCone1 > Real_v(-kTolerance);
    else if (fSTheta > kHalfPi + halfAngTolerance)
      intsect1 &= zOfIntSecPtCone1 < Real_v(kTolerance);
    else {
      vecCore__MaskedAssignFunc(distThetaCone1, (dir.z() < Real_v(0.)), -point.z() / dir.z());
      Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
      intsect1                = Abs(zOfIntSecPtCone1) < kTolerance;
    }

    if (fETheta < kHalfPi - halfAngTolerance)
      intsect2 &= zOfIntSecPtCone2 > Real_v(-kTolerance);
    else if (fETheta > kHalfPi + halfAngTolerance)
      intsect2 &= zOfIntSecPtCone2 < Real_v(kTolerance);
    else {
      vecCore__MaskedAssignFunc(distThetaCone2, (dir.z() > Real_v(0.)), -point.z() / dir.z());
      Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
      intsect2                = Abs(zOfIntSecPtCone2) < kTolerance;
    }
  }

  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE void DistanceToOut(Vector3D<Real_v> const &point, Vector3D<Real_v> const &dir,
                                             Real_v &distThetaCone1, Real_v &distThetaCone2,
                                             typename vecCore::Mask_v<Real_v> &intsect1,
                                             typename vecCore::Mask_v<Real_v> &intsect2) const
  {

    using Bool_v = vecCore::Mask_v<Real_v>;

    Real_v inf(kInfLength);

    // Real_v firstRoot(kInfLength), secondRoot(kInfLength);
    Real_v firstRoot(kInfLength), secondRoot(kInfLength);

    Real_v pDotV2d = point.x() * dir.x() + point.y() * dir.y();
    Real_v rho2    = point.x() * point.x() + point.y() * point.y();

    Real_v b  = pDotV2d - point.z() * dir.z() * tanSTheta2;
    Real_v a  = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;
    Real_v c  = rho2 - point.z() * point.z() * tanSTheta2;
    Real_v d2 = b * b - a * c;
    vecCore__MaskedAssignFunc(d2, d2 < Real_v(0.) && Abs(d2) < kHalfTolerance, Real_v(0.));
    vecCore__MaskedAssignFunc(firstRoot, (d2 >= Real_v(0.)) && b >= Real_v(0.) && a != Real_v(0.),
                              ((-b - Sqrt(Abs(d2))) / NonZero(a)));
    vecCore__MaskedAssignFunc(firstRoot, (d2 >= Real_v(0.)) && b < Real_v(0.), ((c) / NonZero(-b + Sqrt(Abs(d2)))));
    vecCore__MaskedAssignFunc(firstRoot, firstRoot < Real_v(0.), InfinityLength<Real_v>());

    Real_v b2 = point.x() * dir.x() + point.y() * dir.y() - point.z() * dir.z() * tanETheta2;
    Real_v a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2;

    Real_v c2  = point.x() * point.x() + point.y() * point.y() - point.z() * point.z() * tanETheta2;
    Real_v d22 = (b2 * b2) - (a2 * c2);
    vecCore__MaskedAssignFunc(d22, d22 < Real_v(0.) && Abs(d22) < kHalfTolerance, Real_v(0.));

    vecCore__MaskedAssignFunc(secondRoot, (d22 >= Real_v(0.)) && b2 >= Real_v(0.),
                              ((c2) / NonZero(-b2 - Sqrt(Abs(d22)))));
    vecCore__MaskedAssignFunc(secondRoot, (d22 >= Real_v(0.)) && b2 < Real_v(0.) && a2 != Real_v(0.),
                              ((-b2 + Sqrt(Abs(d22))) / NonZero(a2)));

    vecCore__MaskedAssignFunc(secondRoot, secondRoot < Real_v(0.) && Abs(secondRoot) > kTolerance,
                              InfinityLength<Real_v>());
    vecCore__MaskedAssignFunc(secondRoot, Abs(secondRoot) < kTolerance, Real_v(0.));

    if (fSTheta < kPi / 2 + halfAngTolerance) {
      if (fETheta < kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          distThetaCone1          = firstRoot;
          distThetaCone2          = secondRoot;
          Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
          Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

          intsect1 = ((d2 > 0) && (distThetaCone1 != kInfLength) && ((zOfIntSecPtCone1) > -kHalfTolerance));
          intsect2 = ((d22 > 0) && (distThetaCone2 != kInfLength) && ((zOfIntSecPtCone2) > -kHalfTolerance));

          Real_v dirRho2 = dir.x() * dir.x() + dir.y() * dir.y();
          Real_v zs(kInfLength);
          if (fSTheta) zs = dirRho2 / tanSTheta;
          Real_v ze(kInfLength);
          if (fETheta) ze = dirRho2 / tanETheta;
          Bool_v cond = (point.x() == Real_v(0.) && point.y() == Real_v(0.) && point.z() == Real_v(0.) &&
                         dir.z() < zs && dir.z() < ze);
          vecCore__MaskedAssignFunc(distThetaCone1, cond, Real_v(0.));
          vecCore__MaskedAssignFunc(distThetaCone2, cond, Real_v(0.));
          intsect1 |= cond;
          intsect2 |= cond;
        }
      }

      // Second cone is the XY plane, compute distance to plane
      if (fETheta >= kPi / 2 - halfAngTolerance && fETheta <= kPi / 2 + halfAngTolerance) {
        distThetaCone1 = firstRoot;
        distThetaCone2 = inf;
        vecCore__MaskedAssignFunc(distThetaCone2, (dir.z() < Real_v(0.)), Real_v(-1.) * point.z() / dir.z());
        Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
        Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
        intsect2 =
            ((distThetaCone2 != kInfLength) && (Abs(zOfIntSecPtCone2) < kHalfTolerance) && !(dir.z() == Real_v(0.)));
        intsect1 = ((d2 >= 0) && (distThetaCone1 != kInfLength) && (Abs(zOfIntSecPtCone1) < kHalfTolerance) &&
                    !(dir.z() == Real_v(0.)));
      }

      if (fETheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          distThetaCone1 = firstRoot;
          vecCore__MaskedAssignFunc(secondRoot, (d22 >= Real_v(0.)) && b2 > Real_v(0.) && a2 != Real_v(0.),
                                    ((-b2 - Sqrt(Abs(d22))) / NonZero(a2)));
          vecCore__MaskedAssignFunc(secondRoot, (d22 >= Real_v(0.)) && b2 <= Real_v(0.),
                                    ((c2) / NonZero(-b2 + Sqrt(Abs(d22)))));
          vecCore__MaskedAssignFunc(secondRoot, secondRoot < Real_v(0.), InfinityLength<Real_v>());
          distThetaCone2          = secondRoot;
          Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
          Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
          intsect1 = ((d2 >= 0) && (distThetaCone1 != kInfLength) && ((zOfIntSecPtCone1) > -kHalfTolerance));
          intsect2 = ((d22 >= 0) && (distThetaCone2 != kInfLength) && ((zOfIntSecPtCone2) < kHalfTolerance));
        }
      }
    }

    if (fETheta > kPi / 2 + halfAngTolerance) {
      if (fSTheta < fETheta) {
        secondRoot = kInfLength;
        vecCore__MaskedAssignFunc(secondRoot, (d22 >= Real_v(0.)) && b2 > Real_v(0.) && a2 != Real_v(0.),
                                  ((-b2 - Sqrt(Abs(d22))) / NonZero(a2)));
        vecCore__MaskedAssignFunc(secondRoot, (d22 >= Real_v(0.)) && b2 <= Real_v(0.),
                                  ((c2) / NonZero(-b2 + Sqrt(Abs(d22)))));
        distThetaCone2 = secondRoot;

        if (fSTheta > kPi / 2 + halfAngTolerance) {
          vecCore__MaskedAssignFunc(firstRoot, (d2 >= Real_v(0.)) && b > Real_v(0.),
                                    ((c) / NonZero(-b - Sqrt(Abs(d2)))));
          vecCore__MaskedAssignFunc(firstRoot, (d2 >= Real_v(0.)) && b <= Real_v(0.) && a != Real_v(0.),
                                    ((-b + Sqrt(Abs(d2))) / NonZero(a)));
          vecCore__MaskedAssignFunc(firstRoot, firstRoot < Real_v(0.), InfinityLength<Real_v>());
          distThetaCone1          = firstRoot;
          Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
          intsect1 = ((d2 > 0) && (distThetaCone1 != kInfLength) && ((zOfIntSecPtCone1) < kHalfTolerance));
          Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
          intsect2 = ((d22 > 0) && (distThetaCone2 != kInfLength) && ((zOfIntSecPtCone2) < kHalfTolerance));

          Real_v dirRho2 = dir.x() * dir.x() + dir.y() * dir.y();
          Real_v zs(-kInfLength);
          if (tanSTheta) zs = -dirRho2 / tanSTheta;
          Real_v ze(-kInfLength);
          if (tanETheta) ze = -dirRho2 / tanETheta;
          Bool_v cond = (point.x() == Real_v(0.) && point.y() == Real_v(0.) && point.z() == Real_v(0.) &&
                         dir.z() > zs && dir.z() > ze);
          vecCore__MaskedAssignFunc(distThetaCone1, cond, Real_v(0.));
          vecCore__MaskedAssignFunc(distThetaCone2, cond, Real_v(0.));
          // intsect1 |= (cond && tr);
          // intsect2 |= (cond && tr);
          intsect1 |= cond;
          intsect2 |= cond;
        }
      }
    }

    // First cone is the XY plane, compute distance to plane
    if (fSTheta >= kPi / 2 - halfAngTolerance && fSTheta <= kPi / 2 + halfAngTolerance) {
      distThetaCone2 = secondRoot;
      distThetaCone1 = kInfLength;
      vecCore__MaskedAssignFunc(distThetaCone1, (dir.z() > Real_v(0.)), Real_v(-1.) * point.z() / NonZero(dir.z()));
      Real_v zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());

      Real_v zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

      intsect1 =
          ((distThetaCone1 != kInfLength) && (Abs(zOfIntSecPtCone1) < kHalfTolerance) && (dir.z() != Real_v(0.)));
      intsect2 = ((d22 >= 0) && (distThetaCone2 != kInfLength) && (Abs(zOfIntSecPtCone2) < kHalfTolerance) &&
                  (dir.z() != Real_v(0.)));
    }
  }

  // This could be useful in case somebody just want to check whether point is completely inside ThetaRange
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsCompletelyInside(Vector3D<Real_v> const &localPoint) const
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    Real_v rho         = Sqrt(localPoint.Mag2() - (localPoint.z() * localPoint.z()));
    Real_v cone1Radius = Abs(localPoint.z() * tanSTheta);
    Real_v cone2Radius = Abs(localPoint.z() * tanETheta);
    Bool_v isPointOnZAxis =
        localPoint.z() != Real_v(0.) && localPoint.x() == Real_v(0.) && localPoint.y() == Real_v(0.);

    Bool_v isPointOnXYPlane =
        localPoint.z() == Real_v(0.) && (localPoint.x() != Real_v(0.) || localPoint.y() != Real_v(0.));

    Real_v startTheta(fSTheta), endTheta(fETheta);

    Bool_v completelyinside = (isPointOnZAxis && ((startTheta == Real_v(0.) && endTheta == kPi) ||
                                                  (localPoint.z() > Real_v(0.) && startTheta == Real_v(0.)) ||
                                                  (localPoint.z() < Real_v(0.) && endTheta == kPi)));

    completelyinside |=
        (!completelyinside && (isPointOnXYPlane && (startTheta < Real_v(kHalfPi) && endTheta > Real_v(kHalfPi) &&
                                                    (Real_v(kHalfPi) - startTheta) > kAngTolerance &&
                                                    (endTheta - Real_v(kHalfPi)) > kTolerance)));

    if (fSTheta < kHalfPi + halfAngTolerance) {
      if (fETheta < kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Real_v tolAngMin = cone1Radius + Real_v(2 * kAngTolerance * 10.);
          Real_v tolAngMax = cone2Radius - Real_v(2 * kAngTolerance * 10.);

          completelyinside |=
              (!completelyinside &&
               (((rho <= tolAngMax) && (rho >= tolAngMin) && (localPoint.z() > Real_v(0.)) &&
                 Bool_v(fSTheta != Real_v(0.))) ||
                ((rho <= tolAngMax) && Bool_v(fSTheta == Real_v(0.)) && (localPoint.z() > Real_v(0.)))));
        }
      }

      if (fETheta > kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Real_v tolAngMin = cone1Radius + Real_v(2 * kAngTolerance * 10.);
          Real_v tolAngMax = cone2Radius + Real_v(2 * kAngTolerance * 10.);

          completelyinside |= (!completelyinside && (((rho >= tolAngMin) && (localPoint.z() > Real_v(0.))) ||
                                                     ((rho >= tolAngMax) && (localPoint.z() < Real_v(0.)))));
        }
      }

      if (fETheta >= kHalfPi - halfAngTolerance && fETheta <= kHalfPi + halfAngTolerance) {

        completelyinside &= !(Abs(localPoint.z()) < halfAngTolerance);
      }
    }

    if (fETheta > kHalfPi + halfAngTolerance) {
      if (fSTheta >= kHalfPi - halfAngTolerance && fSTheta <= kHalfPi + halfAngTolerance) {

        completelyinside &= !(Abs(localPoint.z()) < halfAngTolerance);
      }

      if (fSTheta > kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Real_v tolAngMin = cone1Radius - Real_v(2 * kAngTolerance * 10.);
          Real_v tolAngMax = cone2Radius + Real_v(2 * kAngTolerance * 10.);

          completelyinside |=
              (!completelyinside &&
               (((rho <= tolAngMin) && (rho >= tolAngMax) && (localPoint.z() < Real_v(0.)) && Bool_v(fETheta != kPi)) ||
                ((rho <= tolAngMin) && (localPoint.z() < Real_v(0.)) && Bool_v(fETheta == kPi))));
        }
      }
    }
    return completelyinside;
  }

  // This could be useful in case somebody just want to check whether point is completely outside ThetaRange
  template <typename Real_v>
  VECCORE_ATT_HOST_DEVICE typename vecCore::Mask_v<Real_v> IsCompletelyOutside(Vector3D<Real_v> const &localPoint) const
  {

    using Bool_v       = vecCore::Mask_v<Real_v>;
    Real_v diff        = localPoint.Perp2();
    Real_v rho         = Sqrt(diff);
    Real_v cone1Radius = Abs(localPoint.z() * tanSTheta);
    Real_v cone2Radius = Abs(localPoint.z() * tanETheta);
    Bool_v isPointOnZAxis =
        localPoint.z() != Real_v(0.) && localPoint.x() == Real_v(0.) && localPoint.y() == Real_v(0.);

    Bool_v isPointOnXYPlane =
        localPoint.z() == Real_v(0.) && (localPoint.x() != Real_v(0.) || localPoint.y() != Real_v(0.));

    // Real_v startTheta(fSTheta), endTheta(fETheta);

    Bool_v completelyoutside = (isPointOnZAxis && ((Bool_v(fSTheta != Real_v(0.)) && Bool_v(fETheta != kPi)) ||
                                                   (localPoint.z() > Real_v(0.) && Bool_v(fSTheta != Real_v(0.))) ||
                                                   (localPoint.z() < Real_v(0.) && Bool_v(fETheta != kPi))));

    completelyoutside |=
        (!completelyoutside &&
         (isPointOnXYPlane && Bool_v(((fSTheta < kHalfPi) && (fETheta < kHalfPi) &&
                                      ((kHalfPi - fSTheta) > kAngTolerance) && ((kHalfPi - fETheta) > kTolerance)) ||
                                     ((fSTheta > kHalfPi && fETheta > kHalfPi) &&
                                      ((fSTheta - kHalfPi) > kAngTolerance) && ((fETheta - kHalfPi) > kTolerance)))));

    if (fSTheta < kHalfPi + halfAngTolerance) {
      if (fETheta < kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {

          Real_v tolAngMin2 = cone1Radius - Real_v(2 * kAngTolerance * 10.);
          Real_v tolAngMax2 = cone2Radius + Real_v(2 * kAngTolerance * 10.);

          completelyoutside |=
              (!completelyoutside && ((rho < tolAngMin2) || (rho > tolAngMax2) || (localPoint.z() < Real_v(0.))));
        }
      }

      if (fETheta > kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Real_v tolAngMin2 = cone1Radius - Real_v(2 * kAngTolerance * 10.);
          Real_v tolAngMax2 = cone2Radius - Real_v(2 * kAngTolerance * 10.);
          completelyoutside |= (!completelyoutside && (((rho < tolAngMin2) && (localPoint.z() > Real_v(0.))) ||
                                                       ((rho < tolAngMax2) && (localPoint.z() < Real_v(0.)))));
        }
      }

      if (fETheta >= kHalfPi - halfAngTolerance && fETheta <= kHalfPi + halfAngTolerance) {

        // completelyinside &= !(Abs(localPoint.z()) < halfAngTolerance);
        completelyoutside &= !(Abs(localPoint.z()) < halfAngTolerance);
      }
    }

    if (fETheta > kHalfPi + halfAngTolerance) {
      if (fSTheta >= kHalfPi - halfAngTolerance && fSTheta <= kHalfPi + halfAngTolerance) {

        // completelyinside &= !(Abs(localPoint.z()) < halfAngTolerance);
        completelyoutside &= !(Abs(localPoint.z()) < halfAngTolerance);
      }

      if (fSTheta > kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {

          Real_v tolAngMin2 = cone1Radius + Real_v(2 * kAngTolerance * 10.);
          Real_v tolAngMax2 = cone2Radius - Real_v(2 * kAngTolerance * 10.);
          completelyoutside |=
              (!completelyoutside && ((rho < tolAngMax2) || (rho > tolAngMin2) || (localPoint.z() > Real_v(0.))));
        }
      }
    }

    return completelyoutside;
  }

  template <typename Real_v, bool ForInside>
  VECCORE_ATT_HOST_DEVICE void GenericKernelForContainsAndInside(
      Vector3D<Real_v> const &localPoint, typename vecCore::Mask_v<Real_v> &completelyinside,
      typename vecCore::Mask_v<Real_v> &completelyoutside) const
  {
    if (ForInside) completelyinside = IsCompletelyInside<Real_v>(localPoint);

    completelyoutside = IsCompletelyOutside<Real_v>(localPoint);
  }

}; // end of class ThetaCone
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_THETACONE_H_ */
