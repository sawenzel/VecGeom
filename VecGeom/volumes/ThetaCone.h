/*
 * ThetaCone.h
 *
 *      Author: Raman Sehgal
 */

#ifndef VECGEOM_VOLUMES_THETACONE_H_
#define VECGEOM_VOLUMES_THETACONE_H_

#include "VecGeom/base/Global.h"
#include "VecGeom/volumes/kernel/GenericKernels.h"
#include <iomanip>
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
  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE Vector3D<typename Backend::precision_v> GetNormal1(
      Vector3D<typename Backend::precision_v> const &point) const
  {

    Vector3D<typename Backend::precision_v> normal(2. * point.x(), 2. * point.y(), -2. * tanSTheta2 * point.z());

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

  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE Vector3D<typename Backend::precision_v> GetNormal2(
      Vector3D<typename Backend::precision_v> const &point) const
  {

    Vector3D<typename Backend::precision_v> normal(2 * point.x(), 2 * point.y(), -2 * tanETheta2 * point.z());

    if (fETheta <= kPi / 2.)
      return normal;
    else
      return -normal;
  }

  /* Function Name : GetNormal<Backend, ForStartTheta>()
   *
   * The function is the templatized version GetNormal1() and GetNormal2() function with one more template
   * parameter and will return the normal depending upon the boolean template parameter "ForStartTheta"
   * which if passed as true, will return normal to the StartingTheta of ThetaCone,
   * if passed as false, will return normal to the EndingTheta of ThetaCone
   *
   * from user point of view the same work can be done by calling GetNormal1() and GetNormal2()
   * functions, but this implementation will be used by "IsPointOnSurfaceAndMovingOut()" function
   */
  template <typename Backend, bool ForStartTheta>
  VECCORE_ATT_HOST_DEVICE Vector3D<typename Backend::precision_v> GetNormal(
      Vector3D<typename Backend::precision_v> const &point) const
  {

    if (ForStartTheta) {

      Vector3D<typename Backend::precision_v> normal(point.x(), point.y(), -tanSTheta2 * point.z());

      if (fSTheta <= kHalfPi)
        return -normal;
      else
        return normal;

    } else {

      Vector3D<typename Backend::precision_v> normal(point.x(), point.y(), -tanETheta2 * point.z());

      if (fETheta <= kHalfPi)
        return normal;
      else
        return -normal;
    }
  }

  /* Function Name :  IsOnSurfaceGeneric<Backend, ForStartTheta>()
   *
   * This version of IsOnSurfaceGeneric is having one more template parameter of type boolean,
   * which if passed as true, will check whether the point is on StartingTheta Surface of ThetaCone,
   * and if passed as false, will check whether the point is on EndingTheta Surface of ThetaCone
   *
   * this implementation will be used by "IsPointOnSurfaceAndMovingOut()" function.
   */
  template <typename Backend, bool ForStartTheta>
  VECCORE_ATT_HOST_DEVICE typename Backend::bool_v IsOnSurfaceGeneric(
      Vector3D<typename Backend::precision_v> const &point) const
  {

    typedef typename Backend::precision_v Float_t;
    Float_t rhs(0.);
    if (ForStartTheta) {
      rhs = Abs(tanSTheta * point.z());
    } else {
      rhs = Abs(tanETheta * point.z());
    }
    Float_t rho2 = point.Perp2();
    return rho2 >= MakeMinusTolerantSquare<true>(rhs) && rho2 <= MakePlusTolerantSquare<true>(rhs);
  }

  /* Function Name : IsPointOnSurfaceAndMovingOut<Backend, ForStartTheta, MovingOut>
   *
   * This function is written to check if the point is on surface and is moving inside or outside.
   * This will basically be used by "DistanceToInKernel()" and "DistanceToOutKernel()" of the shapes,
   * which uses ThetaCone.
   *
   * It contains two extra template boolean parameters "ForStartTheta" and "MovingOut",
   * So call like "IsPointOnSurfaceAndMovingOut<Backend,true,true>" will check whether the points is on
   * the StartingTheta Surface of Theta and moving outside.
   *
   * So overall can be called in following four combinations
   * 1) "IsPointOnSurfaceAndMovingOut<Backend,true,true>" : Point on StartingTheta surface of ThetaCone and moving OUT
   * 2) "IsPointOnSurfaceAndMovingOut<Backend,true,false>" : Point on StartingTheta surface of ThetaCone and moving IN
   * 3) "IsPointOnSurfaceAndMovingOut<Backend,false,true>" : Point on EndingTheta surface of ThetaCone and moving OUT
   * 4) "IsPointOnSurfaceAndMovingOut<Backend,false,false>" : Point on EndingTheta surface of ThetaCone and moving IN
   *
   * Very useful for DistanceToIn and DistanceToOut.
   */
  template <typename Backend, bool ForStartTheta, bool MovingOut>
  VECCORE_ATT_HOST_DEVICE typename Backend::bool_v IsPointOnSurfaceAndMovingOut(
      Vector3D<typename Backend::precision_v> const &point, Vector3D<typename Backend::precision_v> const &dir) const
  {

    if (MovingOut) {
      return IsOnSurfaceGeneric<Backend, ForStartTheta>(point) &&
             (dir.Dot(GetNormal<Backend, ForStartTheta>(point)) > 0.);
    } else {
      return IsOnSurfaceGeneric<Backend, ForStartTheta>(point) &&
             (dir.Dot(GetNormal<Backend, ForStartTheta>(point)) < 0.);
    }
  }

  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::bool_v Contains(Vector3D<typename Backend::precision_v> const &point) const
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t unused(false);
    Bool_t outside(false);
    GenericKernelForContainsAndInside<Backend, false>(point, unused, outside);
    return !outside;
  }

  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::bool_v ContainsWithBoundary(
      Vector3D<typename Backend::precision_v> const & /*point*/) const
  {
  }

  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::inside_v Inside(Vector3D<typename Backend::precision_v> const &point) const
  {

    typedef typename Backend::bool_v Bool_t;
    Bool_t completelyinside(false), completelyoutside(false);
    GenericKernelForContainsAndInside<Backend, true>(point, completelyinside, completelyoutside);
    typename Backend::inside_v inside = EInside::kSurface;
    vecCore::MaskedAssign(inside, completelyoutside, EInside::kOutside);
    vecCore::MaskedAssign(inside, completelyinside, EInside::kInside);
    return inside;
  }

  /**
   * estimate of the smallest distance to the ThetaCone boundary when
   * the point is located outside the ThetaCone
   */
  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::precision_v SafetyToIn(
      Vector3D<typename Backend::precision_v> const &point) const
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t safeTheta(0.);
    Float_t pointRad = Sqrt(point.x() * point.x() + point.y() * point.y());
    Float_t sfTh1    = DistanceToLine<Backend>(slope1, pointRad, point.z());
    Float_t sfTh2    = DistanceToLine<Backend>(slope2, pointRad, point.z());

    safeTheta   = Min(sfTh1, sfTh2);
    Bool_t done = Contains<Backend>(point);
    vecCore__MaskedAssignFunc(safeTheta, done, Float_t(0.0));
    if (vecCore::MaskFull(done)) return safeTheta;

    // Case 1 : Both cones are in Positive Z direction
    if (fSTheta < kPi / 2 + halfAngTolerance) {
      if (fETheta < kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          vecCore::MaskedAssign(safeTheta, (!done && point.z() < 0.0), sfTh2);
        }
      }

      // Case 2 : First Cone is in Positive Z direction and Second is in Negative Z direction
      if (fETheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          vecCore::MaskedAssign(safeTheta, (!done && point.z() > 0.0), sfTh1);
          vecCore::MaskedAssign(safeTheta, (!done && point.z() < 0.0), sfTh2);
        }
      }
    }

    // Case 3 : Both cones are in Negative Z direction
    if (fETheta > kPi / 2 + halfAngTolerance) {
      if (fSTheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          vecCore::MaskedAssign(safeTheta, (!done && point.z() > 0.0), sfTh1);
        }
      }
    }

    return safeTheta;
  }

  /**
   * estimate of the smallest distance to the ThetaCone boundary when
   * the point is located inside the ThetaCone ( within the defining phi angle )
   */
  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::precision_v SafetyToOut(
      Vector3D<typename Backend::precision_v> const &point) const
  {

    typedef typename Backend::precision_v Float_t;

    Float_t pointRad    = Sqrt(point.x() * point.x() + point.y() * point.y());
    Float_t bisectorRad = Abs(point.z() * tanBisector);

    vecCore::Mask<Float_t> condition(false);
    Float_t sfTh1 = DistanceToLine<Backend>(slope1, pointRad, point.z());
    Float_t sfTh2 = DistanceToLine<Backend>(slope2, pointRad, point.z());

    // Case 1 : Both cones are in Positive Z direction
    if (fSTheta < kPi / 2 + halfAngTolerance) {
      if (fETheta < kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          condition = (pointRad < bisectorRad) && (fSTheta != Float_t(0.0));
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
          condition = !((pointRad < bisectorRad) && (fETheta != Float_t(kPi)));
        }
      }
    }

    return vecCore::Blend(condition, sfTh1, sfTh2);
  }

  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::precision_v DistanceToLine(Precision const &slope,
                                                                       typename Backend::precision_v const &x,
                                                                       typename Backend::precision_v const &y) const
  {

    typedef typename Backend::precision_v Float_t;
    Float_t dist = (y - slope * x) / Sqrt(1. + slope * slope);
    return Abs(dist);
  }

  /**
   * estimate of the distance to the ThetaCone boundary with given direction
   */
  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE void DistanceToIn(Vector3D<typename Backend::precision_v> const &point,
                                            Vector3D<typename Backend::precision_v> const &dir,
                                            typename Backend::precision_v &distThetaCone1,
                                            typename Backend::precision_v &distThetaCone2,
                                            typename Backend::bool_v &intsect1,
                                            typename Backend::bool_v &intsect2) const
  {

    {
      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;

      Bool_t done(false);
      Bool_t fal(false);

      Float_t firstRoot(kInfLength), secondRoot(kInfLength);

      Float_t pDotV2d = point.x() * dir.x() + point.y() * dir.y();
      Float_t rho2    = point.x() * point.x() + point.y() * point.y();
      Float_t dirRho2 = dir.Perp2();

      Float_t b = pDotV2d - point.z() * dir.z() * tanSTheta2;
      // Float_t a = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;
      Float_t a    = dirRho2 - dir.z() * dir.z() * tanSTheta2;
      Float_t c    = rho2 - point.z() * point.z() * tanSTheta2;
      Float_t d2   = b * b - a * c;
      Float_t aInv = 1. / NonZero(a);

      vecCore__MaskedAssignFunc(firstRoot, (d2 > 0.0), (-b + Sqrt(Abs(d2))) * aInv);
      done |= (Abs(firstRoot) < 3.0 * kTolerance);
      vecCore__MaskedAssignFunc(firstRoot, ((Abs(firstRoot) < 3.0 * kTolerance)), Float_t(0.0));
      vecCore__MaskedAssignFunc(firstRoot, (!done && (firstRoot < 0.0)), InfinityLength<Float_t>());

      Float_t b2 = pDotV2d - point.z() * dir.z() * tanETheta2;
      // Float_t a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2;
      Float_t a2    = dirRho2 - dir.z() * dir.z() * tanETheta2;
      Float_t c2    = rho2 - point.z() * point.z() * tanETheta2;
      Float_t d22   = b2 * b2 - a2 * c2;
      Float_t a2Inv = 1. / NonZero(a2);

      vecCore__MaskedAssignFunc(secondRoot, (d22 > 0.0), (-b2 - Sqrt(Abs(d22))) * a2Inv);
      vecCore__MaskedAssignFunc(secondRoot, (!done && (Abs(secondRoot) < 3.0 * kTolerance)), Float_t(0.0));
      done |= (Abs(secondRoot) < 3.0 * kTolerance);
      vecCore__MaskedAssignFunc(secondRoot, !done && (secondRoot < 0.0), InfinityLength<Float_t>());

      if (fSTheta < kHalfPi + halfAngTolerance) {
        if (fETheta < kHalfPi + halfAngTolerance) {
          if (fSTheta < fETheta) {
            distThetaCone1           = firstRoot;
            distThetaCone2           = secondRoot;
            Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
            Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

            intsect1 = ((d2 > 0.) && (zOfIntSecPtCone1 > 0.));
            intsect2 = ((d22 > 0.) && (zOfIntSecPtCone2 > 0.));
          }
        }

        if (fETheta >= kHalfPi - halfAngTolerance && fETheta <= kHalfPi + halfAngTolerance) {
          vecCore__MaskedAssignFunc(distThetaCone2, (dir.z() > 0.0), -point.z() / dir.z());
          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
          intsect2                 = ((distThetaCone2 != kInfLength) && (Abs(zOfIntSecPtCone2) < halfAngTolerance));
        }

        if (fETheta > kHalfPi + halfAngTolerance) {
          if (fSTheta < fETheta) {
            distThetaCone1 = firstRoot;
            vecCore__MaskedAssignFunc(secondRoot, (d22 > 0.0), (-b2 + Sqrt(Abs(d22))) * a2Inv);

            done = fal;
            done |= (Abs(secondRoot) < 3.0 * kTolerance);
            vecCore__MaskedAssignFunc(secondRoot, ((Abs(secondRoot) < 3.0 * kTolerance)), Float_t(0.0));
            vecCore__MaskedAssignFunc(secondRoot, !done && (secondRoot < 0.0), InfinityLength<Float_t>());
            distThetaCone2 = secondRoot;

            Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
            Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

            intsect1 = ((d2 > 0) && (distThetaCone1 != kInfLength) && (zOfIntSecPtCone1 > 0.));
            intsect2 = ((d22 > 0) && (distThetaCone2 != kInfLength) && (zOfIntSecPtCone2 < 0.));
          }
        }
      }

      if (fSTheta >= kHalfPi - halfAngTolerance) {
        if (fETheta > kHalfPi + halfAngTolerance) {
          if (fSTheta < fETheta) {
            vecCore__MaskedAssignFunc(firstRoot, (d2 > 0.0), (-b - Sqrt(Abs(d2))) * aInv);
            done = fal;
            done |= (Abs(firstRoot) < 3.0 * kTolerance);
            vecCore__MaskedAssignFunc(firstRoot, ((Abs(firstRoot) < 3.0 * kTolerance)), Float_t(0.0));
            vecCore__MaskedAssignFunc(firstRoot, !done && (firstRoot < 0.0), InfinityLength<Float_t>());
            distThetaCone1 = firstRoot;

            vecCore__MaskedAssignFunc(secondRoot, (d22 > 0.0), (-b2 + Sqrt(Abs(d22))) * a2Inv);
            done = fal;
            done |= (Abs(secondRoot) < 3.0 * kTolerance);
            vecCore__MaskedAssignFunc(secondRoot, ((Abs(secondRoot) < 3.0 * kTolerance)), Float_t(0.0));
            vecCore__MaskedAssignFunc(secondRoot, !done && (secondRoot < 0.0), InfinityLength<Float_t>());
            distThetaCone2 = secondRoot;

            Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
            Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

            intsect1 = ((d2 > 0) && (distThetaCone1 != kInfLength) && (zOfIntSecPtCone1 < 0.));
            intsect2 = ((d22 > 0) && (distThetaCone2 != kInfLength) && (zOfIntSecPtCone2 < 0.));
          }
        }
      }

      if (fSTheta >= kHalfPi - halfAngTolerance && fSTheta <= kHalfPi + halfAngTolerance) {
        vecCore__MaskedAssignFunc(distThetaCone1, (dir.z() < 0.0), -point.z() / dir.z());
        Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
        intsect1                 = ((distThetaCone1 != kInfLength) && (Abs(zOfIntSecPtCone1) < halfAngTolerance));
      }
    }
  }

  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE void DistanceToOut(Vector3D<typename Backend::precision_v> const &point,
                                             Vector3D<typename Backend::precision_v> const &dir,
                                             typename Backend::precision_v &distThetaCone1,
                                             typename Backend::precision_v &distThetaCone2,
                                             typename Backend::bool_v &intsect1,
                                             typename Backend::bool_v &intsect2) const
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;

    Float_t inf(kInfLength);

    // Float_t firstRoot(kInfLength), secondRoot(kInfLength);
    Float_t firstRoot(kInfLength), secondRoot(kInfLength);

    Float_t pDotV2d = point.x() * dir.x() + point.y() * dir.y();
    Float_t rho2    = point.x() * point.x() + point.y() * point.y();

    Float_t b  = pDotV2d - point.z() * dir.z() * tanSTheta2;
    Float_t a  = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;
    Float_t c  = rho2 - point.z() * point.z() * tanSTheta2;
    Float_t d2 = b * b - a * c;
    vecCore__MaskedAssignFunc(d2, d2 < 0.0 && Abs(d2) < kHalfTolerance, Float_t(0.0));
    vecCore__MaskedAssignFunc(firstRoot, (d2 >= 0.0) && b >= 0.0 && a != 0.0, ((-b - Sqrt(Abs(d2))) / NonZero(a)));
    vecCore__MaskedAssignFunc(firstRoot, (d2 >= 0.0) && b < 0.0, ((c) / NonZero(-b + Sqrt(Abs(d2)))));
    vecCore__MaskedAssignFunc(firstRoot, firstRoot < 0.0, InfinityLength<Float_t>());

    Float_t b2 = point.x() * dir.x() + point.y() * dir.y() - point.z() * dir.z() * tanETheta2;
    Float_t a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2;

    Float_t c2  = point.x() * point.x() + point.y() * point.y() - point.z() * point.z() * tanETheta2;
    Float_t d22 = (b2 * b2) - (a2 * c2);
    vecCore__MaskedAssignFunc(d22, d22 < 0.0 && Abs(d22) < kHalfTolerance, Float_t(0.0));

    vecCore__MaskedAssignFunc(secondRoot, (d22 >= 0.0) && b2 >= 0.0, ((c2) / NonZero(-b2 - Sqrt(Abs(d22)))));
    vecCore__MaskedAssignFunc(secondRoot, (d22 >= 0.0) && b2 < 0.0 && a2 != 0.0,
                              ((-b2 + Sqrt(Abs(d22))) / NonZero(a2)));

    vecCore__MaskedAssignFunc(secondRoot, secondRoot < 0.0 && Abs(secondRoot) > kTolerance, InfinityLength<Float_t>());
    vecCore__MaskedAssignFunc(secondRoot, Abs(secondRoot) < kTolerance, Float_t(0.0));

    if (fSTheta < kPi / 2 + halfAngTolerance) {
      if (fETheta < kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          distThetaCone1           = firstRoot;
          distThetaCone2           = secondRoot;
          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

          intsect1 = ((d2 > 0) && (distThetaCone1 != kInfLength) && ((zOfIntSecPtCone1) > -kHalfTolerance));
          intsect2 = ((d22 > 0) && (distThetaCone2 != kInfLength) && ((zOfIntSecPtCone2) > -kHalfTolerance));

          Float_t dirRho2 = dir.x() * dir.x() + dir.y() * dir.y();
          Float_t zs(kInfLength);
          if (fSTheta) zs = dirRho2 / tanSTheta;
          Float_t ze(kInfLength);
          if (fETheta) ze = dirRho2 / tanETheta;
          Bool_t cond = (point.x() == 0. && point.y() == 0. && point.z() == 0. && dir.z() < zs && dir.z() < ze);
          vecCore__MaskedAssignFunc(distThetaCone1, cond, Float_t(0.0));
          vecCore__MaskedAssignFunc(distThetaCone2, cond, Float_t(0.0));
          intsect1 |= cond;
          intsect2 |= cond;
        }
      }

      if (fETheta >= kPi / 2 - halfAngTolerance && fETheta <= kPi / 2 + halfAngTolerance) {
        distThetaCone1 = firstRoot;
        distThetaCone2 = inf;
        vecCore__MaskedAssignFunc(distThetaCone2, (dir.z() < 0.0), -1. * point.z() / dir.z());
        Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
        Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
        intsect2 = ((d22 >= 0) && (distThetaCone2 != kInfLength) && (Abs(zOfIntSecPtCone2) < kHalfTolerance) &&
                    !(dir.z() == 0.));
        intsect1 = ((d2 >= 0) && (distThetaCone1 != kInfLength) && (Abs(zOfIntSecPtCone1) < kHalfTolerance) &&
                    !(dir.z() == 0.));
      }

      if (fETheta > kPi / 2 + halfAngTolerance) {
        if (fSTheta < fETheta) {
          distThetaCone1 = firstRoot;
          vecCore__MaskedAssignFunc(secondRoot, (d22 >= 0.0) && b2 > 0.0 && a2 != 0.0,
                                    ((-b2 - Sqrt(Abs(d22))) / NonZero(a2)));
          vecCore__MaskedAssignFunc(secondRoot, (d22 >= 0.0) && b2 <= 0.0, ((c2) / NonZero(-b2 + Sqrt(Abs(d22)))));
          vecCore__MaskedAssignFunc(secondRoot, secondRoot < 0.0, InfinityLength<Float_t>());
          distThetaCone2           = secondRoot;
          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
          intsect1 = ((d2 >= 0) && (distThetaCone1 != kInfLength) && ((zOfIntSecPtCone1) > -kHalfTolerance));
          intsect2 = ((d22 >= 0) && (distThetaCone2 != kInfLength) && ((zOfIntSecPtCone2) < kHalfTolerance));
        }
      }
    }

    if (fETheta > kPi / 2 + halfAngTolerance) {
      if (fSTheta < fETheta) {
        secondRoot = kInfLength;
        vecCore__MaskedAssignFunc(secondRoot, (d22 >= 0.0) && b2 > 0.0 && a2 != 0.0,
                                  ((-b2 - Sqrt(Abs(d22))) / NonZero(a2)));
        vecCore__MaskedAssignFunc(secondRoot, (d22 >= 0.0) && b2 <= 0.0, ((c2) / NonZero(-b2 + Sqrt(Abs(d22)))));
        distThetaCone2 = secondRoot;

        if (fSTheta > kPi / 2 + halfAngTolerance) {
          vecCore__MaskedAssignFunc(firstRoot, (d2 >= 0.0) && b > 0.0, ((c) / NonZero(-b - Sqrt(Abs(d2)))));
          vecCore__MaskedAssignFunc(firstRoot, (d2 >= 0.0) && b <= 0.0 && a != 0.0,
                                    ((-b + Sqrt(Abs(d2))) / NonZero(a)));
          vecCore__MaskedAssignFunc(firstRoot, firstRoot < 0.0, InfinityLength<Float_t>());
          distThetaCone1           = firstRoot;
          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
          intsect1 = ((d2 > 0) && (distThetaCone1 != kInfLength) && ((zOfIntSecPtCone1) < kHalfTolerance));
          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
          intsect2 = ((d22 > 0) && (distThetaCone2 != kInfLength) && ((zOfIntSecPtCone2) < kHalfTolerance));

          Float_t dirRho2 = dir.x() * dir.x() + dir.y() * dir.y();
          Float_t zs(-kInfLength);
          if (tanSTheta) zs = -dirRho2 / tanSTheta;
          Float_t ze(-kInfLength);
          if (tanETheta) ze = -dirRho2 / tanETheta;
          Bool_t cond = (point.x() == 0. && point.y() == 0. && point.z() == 0. && dir.z() > zs && dir.z() > ze);
          vecCore__MaskedAssignFunc(distThetaCone1, cond, Float_t(0.0));
          vecCore__MaskedAssignFunc(distThetaCone2, cond, Float_t(0.0));
          // intsect1 |= (cond && tr);
          // intsect2 |= (cond && tr);
          intsect1 |= cond;
          intsect2 |= cond;
        }
      }
    }

    if (fSTheta >= kPi / 2 - halfAngTolerance && fSTheta <= kPi / 2 + halfAngTolerance) {
      distThetaCone2 = secondRoot;
      distThetaCone1 = kInfLength;
      vecCore__MaskedAssignFunc(distThetaCone1, (dir.z() > 0.), -1. * point.z() / NonZero(dir.z()));
      Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());

      Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

      intsect1 =
          ((d2 >= 0) && (distThetaCone1 != kInfLength) && (Abs(zOfIntSecPtCone1) < kHalfTolerance) && (dir.z() != 0.));
      intsect2 =
          ((d22 >= 0) && (distThetaCone2 != kInfLength) && (Abs(zOfIntSecPtCone2) < kHalfTolerance) && (dir.z() != 0.));
    }
  }

  // This could be useful in case somebody just want to check whether point is completely inside ThetaRange
  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::bool_v IsCompletelyInside(
      Vector3D<typename Backend::precision_v> const &localPoint) const
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    // Float_t pi(kPi), zero(0.);
    Float_t rho           = Sqrt(localPoint.Mag2() - (localPoint.z() * localPoint.z()));
    Float_t cone1Radius   = Abs(localPoint.z() * tanSTheta);
    Float_t cone2Radius   = Abs(localPoint.z() * tanETheta);
    Bool_t isPointOnZAxis = localPoint.z() != 0. && localPoint.x() == 0. && localPoint.y() == 0.;

    Bool_t isPointOnXYPlane = localPoint.z() == 0. && (localPoint.x() != 0. || localPoint.y() != 0.);

    Float_t startTheta(fSTheta), endTheta(fETheta);

    Bool_t completelyinside =
        (isPointOnZAxis && ((startTheta == 0. && endTheta == kPi) || (localPoint.z() > 0. && startTheta == 0.) ||
                            (localPoint.z() < 0. && endTheta == kPi)));

    completelyinside |= (!completelyinside && (isPointOnXYPlane && (startTheta < kHalfPi && endTheta > kHalfPi &&
                                                                    (kHalfPi - startTheta) > kAngTolerance &&
                                                                    (endTheta - kHalfPi) > kTolerance)));

    if (fSTheta < kHalfPi + halfAngTolerance) {
      if (fETheta < kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Float_t tolAngMin = cone1Radius + 2 * kAngTolerance * 10.;
          Float_t tolAngMax = cone2Radius - 2 * kAngTolerance * 10.;

          completelyinside |=
              (!completelyinside &&
               (((rho <= tolAngMax) && (rho >= tolAngMin) && (localPoint.z() > 0.) && Bool_t(fSTheta != 0.)) ||
                ((rho <= tolAngMax) && Bool_t(fSTheta == 0.) && (localPoint.z() > 0.))));
        }
      }

      if (fETheta > kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Float_t tolAngMin = cone1Radius + 2 * kAngTolerance * 10.;
          Float_t tolAngMax = cone2Radius + 2 * kAngTolerance * 10.;

          completelyinside |= (!completelyinside && (((rho >= tolAngMin) && (localPoint.z() > 0.)) ||
                                                     ((rho >= tolAngMax) && (localPoint.z() < 0.))));
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
          Float_t tolAngMin = cone1Radius - 2 * kAngTolerance * 10.;
          Float_t tolAngMax = cone2Radius + 2 * kAngTolerance * 10.;

          completelyinside |=
              (!completelyinside &&
               (((rho <= tolAngMin) && (rho >= tolAngMax) && (localPoint.z() < 0.) && Bool_t(fETheta != kPi)) ||
                ((rho <= tolAngMin) && (localPoint.z() < 0.) && Bool_t(fETheta == kPi))));
        }
      }
    }
    return completelyinside;
  }

  // This could be useful in case somebody just want to check whether point is completely outside ThetaRange
  template <typename Backend>
  VECCORE_ATT_HOST_DEVICE typename Backend::bool_v IsCompletelyOutside(
      Vector3D<typename Backend::precision_v> const &localPoint) const
  {

    typedef typename Backend::precision_v Float_t;
    typedef typename Backend::bool_v Bool_t;
    // Float_t pi(kPi), zero(0.);
    Float_t diff          = localPoint.Perp2();
    Float_t rho           = Sqrt(diff);
    Float_t cone1Radius   = Abs(localPoint.z() * tanSTheta);
    Float_t cone2Radius   = Abs(localPoint.z() * tanETheta);
    Bool_t isPointOnZAxis = localPoint.z() != 0. && localPoint.x() == 0. && localPoint.y() == 0.;

    Bool_t isPointOnXYPlane = localPoint.z() == 0. && (localPoint.x() != 0. || localPoint.y() != 0.);

    // Float_t startTheta(fSTheta), endTheta(fETheta);

    Bool_t completelyoutside = (isPointOnZAxis && ((Bool_t(fSTheta != 0.) && Bool_t(fETheta != kPi)) ||
                                                   (localPoint.z() > 0. && Bool_t(fSTheta != 0.)) ||
                                                   (localPoint.z() < 0. && Bool_t(fETheta != kPi))));

    completelyoutside |=
        (!completelyoutside &&
         (isPointOnXYPlane && Bool_t(((fSTheta < kHalfPi) && (fETheta < kHalfPi) &&
                                      ((kHalfPi - fSTheta) > kAngTolerance) && ((kHalfPi - fETheta) > kTolerance)) ||
                                     ((fSTheta > kHalfPi && fETheta > kHalfPi) &&
                                      ((fSTheta - kHalfPi) > kAngTolerance) && ((fETheta - kHalfPi) > kTolerance)))));

    if (fSTheta < kHalfPi + halfAngTolerance) {
      if (fETheta < kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {

          Float_t tolAngMin2 = cone1Radius - 2 * kAngTolerance * 10.;
          Float_t tolAngMax2 = cone2Radius + 2 * kAngTolerance * 10.;

          completelyoutside |=
              (!completelyoutside && ((rho < tolAngMin2) || (rho > tolAngMax2) || (localPoint.z() < 0.)));
        }
      }

      if (fETheta > kHalfPi + halfAngTolerance) {
        if (fSTheta < fETheta) {
          Float_t tolAngMin2 = cone1Radius - 2 * kAngTolerance * 10.;
          Float_t tolAngMax2 = cone2Radius - 2 * kAngTolerance * 10.;
          completelyoutside |= (!completelyoutside && (((rho < tolAngMin2) && (localPoint.z() > 0.)) ||
                                                       ((rho < tolAngMax2) && (localPoint.z() < 0.))));
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

          Float_t tolAngMin2 = cone1Radius + 2 * kAngTolerance * 10.;
          Float_t tolAngMax2 = cone2Radius - 2 * kAngTolerance * 10.;
          completelyoutside |=
              (!completelyoutside && ((rho < tolAngMax2) || (rho > tolAngMin2) || (localPoint.z() > 0.)));
        }
      }
    }

    return completelyoutside;
  }

  template <typename Backend, bool ForInside>
  VECCORE_ATT_HOST_DEVICE void GenericKernelForContainsAndInside(
      Vector3D<typename Backend::precision_v> const &localPoint, typename Backend::bool_v &completelyinside,
      typename Backend::bool_v &completelyoutside) const
  {
    if (ForInside) completelyinside = IsCompletelyInside<Backend>(localPoint);

    completelyoutside = IsCompletelyOutside<Backend>(localPoint);
  }

}; // end of class ThetaCone
} // namespace VECGEOM_IMPL_NAMESPACE
} // namespace vecgeom

#endif /* VECGEOM_VOLUMES_THETACONE_H_ */
