/*
 * ConeImplementation.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_

#include "base/Global.h"
#include "base/Transformation3D.h"
#include "volumes/UnplacedCone.h"
#include "volumes/kernel/shapetypes/ConeTypes.h"
#include "volumes/kernel/TubeImplementation.h"
#include <cassert>
#include <stdio.h>

namespace vecgeom {

VECGEOM_DEVICE_DECLARE_CONV_TEMPLATE_2v_1t(ConeImplementation, TranslationCode,transCodeT, RotationCode,rotCodeT,typename,ConeType)

inline namespace VECGEOM_IMPL_NAMESPACE {

template <typename ConeType>
struct ConeHelper
{
/*
    template <typename VecType, typename OtherType typename BoolType>
    static
    inline
    typename BoolType IsInRightZInterval( VecType const & z, OtherType const & dz )
    {
       return Abs(z) <= dz;
    }

   template<typename ConeType>
   template<typename VectorType, typename MaskType>
   inline
   __attribute__((always_inline))
   typename MaskType determineRHit( UnplacedCone const & unplaced, VectorType const & x, VectorType const & y, VectorType const & z,
                                    VectorType const & dirx, VectorType const & diry, VectorType const & dirz,
                                    VectorType const & distanceR ) const
   {
      if( ! checkPhiTreatment<ConeType>( unplaced ) )
      {
         return distanceR > 0 && IsInRightZInterval( z+distanceR*dirz, unplaced->GetDz() );
      }
      else
      {
         // need to have additional look if hitting point on zylinder is not in empty phi range
         VectorType xhit = x + distanceR*dirx;
         VectorType yhit = y + distanceR*diry;
         return distanceR > 0 && IsInRightZInterval( z + distanceR*dirz, unplaced->GetDz() )
                      && ! GeneralPhiUtils::PointIsInPhiSector(
                         unplaced->normalPhi1.x,
                         unplaced->normalPhi1.y,
                         unplaced->normalPhi2.x,
                         unplaced->normalPhi2.y, xhit, yhit );
      }
   }
*/
};

class PlacedCone;

template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
struct ConeImplementation {

  static const int transC = transCodeT;
  static const int rotC   = rotCodeT;

  using PlacedShape_t = PlacedCone;
  using UnplacedShape_t = UnplacedCone;

  VECGEOM_CUDA_HEADER_BOTH
  static void PrintType() {
     printf("SpecializedCone<%i, %i>", transCodeT, rotCodeT);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedContains(
  UnplacedCone const &unplaced,
  Vector3D<typename Backend::precision_v> const &point,
  typename Backend::bool_v &contains) {

  typedef typename Backend::precision_v Float_t;
  typedef typename Backend::bool_v Bool_t;

//  std::cerr << point.x() << " " << point.y() << " " << point.z() << "\n";
 // Float_t unplaced.GetDz() = unplaced.GetDz();
  Bool_t inside = Abs( point.z() ) < unplaced.GetDz();

//  std::cerr << inside << "\n";

  // this could be used even for vector
  if( Backend::early_returns ){
      if( inside == Backend::kFalse ){
        // here all particles are outside
        contains = inside;
        return;
    }
  }

//  std::cerr << "after first return" << "\n";
//  std::cerr << inside << "\n";


  Float_t r2 = point.x()*point.x()+point.y()*point.y();
  // calculate cone radius at the z-height of position
  Float_t rh = unplaced.GetOuterSlope()*point.z()
          + unplaced.GetOuterOffset();
  inside &= ( r2 < rh*rh );

  if( Backend::early_returns ){
    if( inside == Backend::kFalse ){
        // here all particles are outside
        contains = inside;
        return;
    }
  }

//  std::cerr << "after second return" << "\n";
//  std::cerr << inside << "\n";


  if(ConeTypes::checkRminTreatment<ConeType>(unplaced)){
     Float_t rl = unplaced.GetInnerSlope()*point.z() + unplaced.GetInnerOffset();
     inside &= ( r2 > rl*rl );

     if( Backend::early_returns ){
        if( inside == Backend::kFalse ){
            // here all particles are outside
            contains = inside;
            return;
        }
      }
  }

//  std::cerr << "after third return" << "\n";
//  std::cerr << inside << "\n";

   // inside phi sector, then?
   Bool_t insector = Backend::kTrue;
   if( ConeTypes::checkPhiTreatment<ConeType>(unplaced)) {
      TubeUtilities::PointInCyclicalSector<Backend, ConeType, UnplacedCone, false>(unplaced, point.x(), point.y(), insector);
      inside &= insector;
   }

//   std::cerr << "after Phi treatment" << "\n";
//   std::cerr << inside << "\n";

   // this is not correct; look at tube stuff
   contains=inside;
}



  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Contains(
       UnplacedCone const &unplaced,
       Transformation3D const &transformation,
       Vector3D<typename Backend::precision_v> const &point,
       Vector3D<typename Backend::precision_v> &localPoint,
       typename Backend::bool_v &contains) {

      localPoint=transformation.Transform<transCodeT,rotCodeT>(point);
      return UnplacedContains<Backend>(unplaced,localPoint,contains);
  }


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void UnplacedInside(
      UnplacedCone const &unplaced,
      Vector3D<typename Backend::precision_v> const & point,
      typename Backend::int_v &location) {

      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;

      //  std::cerr << point.x() << " " << point.y() << " " << point.z() << "\n";
     // Float_t unplaced.GetDz() = unplaced.GetDz();
      // could cache a tolerant unplaced.GetDz();
      Float_t absz = Abs(point.z());
      Bool_t completelyinside = absz < unplaced.GetDz() - kTolerance;
      // could cache a tolerant unplaced.GetDz();
      Bool_t completelyoutside = absz > unplaced.GetDz() + kTolerance;

      if ( Backend::early_returns ){
        if ( completelyoutside == Backend::kTrue ){
                      // here all particles are outside
//           std::cerr << "ereturn 1\n";
           location = EInside::kOutside;
           return;
        }
      }

      Float_t r2 = point.x()*point.x()+point.y()*point.y();
      // calculate cone radius at the z-height of position
      Float_t rh = unplaced.GetOuterSlope()*point.z()
                + unplaced.GetOuterOffset();

      completelyinside &= ( r2 < (rh-kTolerance)*(rh-kTolerance) );
      // TODO: could we reuse the computation from the previous line??
      completelyoutside |= ( r2 > (rh+kTolerance)*(rh+kTolerance) );

      /** think about a suitable early return condition **/
      if (Backend::early_returns){
        if ( completelyoutside == Backend::kTrue ){
              location = EInside::kOutside;
//              std::cerr << "ereturn 2\n";
              return;
        }
      }

      // treat inner radius
      if (ConeTypes::checkRminTreatment<ConeType>(unplaced)){
         Float_t rl = unplaced.GetInnerSlope()*point.z() + unplaced.GetInnerOffset();
         completelyinside &= ( r2 > (rl+kTolerance)*(rl+kTolerance) );
         completelyoutside |= ( r2 < (rl-kTolerance)*(rl-kTolerance) );

         if( Backend::early_returns ){
            if( completelyoutside == Backend::kTrue ){
                // here all particles are outside
                location = EInside::kOutside;
 //               std::cerr << "ereturn 3\n";
                return;
              }
            }
      } /* end inner radius treatment */

      // inside phi sector, then?
      Bool_t insector = Backend::kTrue;
      if(ConeTypes::checkPhiTreatment<ConeType>(unplaced)) {
         TubeUtilities::PointInCyclicalSector<Backend, ConeType, UnplacedCone, false>
         (unplaced, point.x(), point.y(), insector);
         completelyinside &= insector;
         completelyoutside |= !insector;
      }
      // could do a final early return check for completely outside here

      // here we are done; do final assignment
      location = EInside::kSurface;
      MaskedAssign( completelyinside, EInside::kInside, &location );
      MaskedAssign( completelyoutside, EInside::kOutside, &location );
//      std::cerr << "final return\n";
      return;
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedCone const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> &localPoint,
                     typename Backend::int_v &inside) {
 //   localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
 //   UnplacedInside<Backend>(unplaced, localPoint, inside);
  }


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void Inside(UnplacedCone const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     typename Backend::int_v &inside) {
    // typename Backend::bool_v contains;
    Vector3D<typename Backend::precision_v> localPoint
        = transformation.Transform<transCodeT, rotCodeT>(point);
    //  typename Backend::int_v crosscheck;
    UnplacedInside<Backend>(unplaced,localPoint,inside);


//    UnplacedContains<Backend>(unplaced, localPoint, contains);
//    inside = EInside::kOutside;
//    MaskedAssign(contains, EInside::kInside, &inside);
//    // crosscheck
//    if( inside != crosscheck )
//    {
//        std::cerr << "diff " << inside << " " << crosscheck << "\n";
//    }
//    else
//    {
//        std::cerr << "same " << inside << " " << crosscheck << "\n";
//    }
 }

  // we need to provide the Contains functions for the boolean interface

  // the fall-back version from USolids
  // to take a short cut towards full functionality
  // this really only makes sense for Scalar and CUDA backend
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static typename Backend::precision_v DistanceToInUSolids
  ( UnplacedCone const &unplaced,
    Transformation3D const &transformation,
    Vector3D<typename Backend::precision_v> const &point,
    Vector3D<typename Backend::precision_v> const &direction,
    typename Backend::precision_v const &stepMax ) {

    // first of all: transform points and directions
    Vector3D<typename Backend::precision_v> p = transformation.Transform<transCodeT,rotCodeT>(point);
    Vector3D<typename Backend::precision_v> v = transformation.TransformDirection<rotCodeT>(direction);

    // now comes the original USolids code
    double snxt = VECGEOM_NAMESPACE::kInfinity;
    const double halfCarTolerance = VECGEOM_NAMESPACE::kHalfTolerance;
    const double halfRadTolerance = VECGEOM_NAMESPACE::kRadTolerance * 0.5;

    double rMaxAv, rMaxOAv; // Data for cones
    double rMinAv, rMinOAv;
    double rout, rin;

    double tolORMin, tolORMin2, tolIRMin, tolIRMin2; // `generous' radii squared
    double tolORMax2, tolIRMax, tolIRMax2;
    double tolODz, tolIDz;

    double Dist, sd, xi, yi, zi, ri = 0., risec, rhoi2, cosPsi; // Intersection point vars

    double t1, t2, t3, b, c, d; // Quadratic solver variables
    double nt1, nt2, nt3;
    double Comp;

    Vector3D<Precision> norm;

    // Cone Precalcs
    rMinAv  = (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;

    if (rMinAv > halfRadTolerance) {
          rMinOAv = rMinAv - halfRadTolerance;
    }
    else {
          rMinOAv = 0.0;
    }
    rMaxAv  = (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
    rMaxOAv = rMaxAv + halfRadTolerance;

    // Intersection with z-surfaces
    tolIDz = unplaced.GetDz() - halfCarTolerance;
    tolODz = unplaced.GetDz() + halfCarTolerance;

    if (Abs(p.z()) >= tolIDz)
    {
          if (p.z() * v.z() < 0)   // at +Z going in -Z or visa versa
          {
            sd = (Abs(p.z()) - unplaced.GetDz()) / Abs(v.z()); // Z intersect distance

            if (sd < 0.0)
            {
              sd = 0.0;  // negative dist -> zero
            }

            xi   = p.x() + sd * v.x(); // Intersection coords
            yi   = p.y() + sd * v.y();
            rhoi2 = xi * xi + yi * yi ;

            // Check validity of intersection
            // Calculate (outer) tolerant radi^2 at intersecion

            if (v.z() > 0)
            {
              tolORMin  = unplaced.GetRmin1() - halfRadTolerance * unplaced.fSecRMin;
              tolIRMin  = unplaced.GetRmin1() + halfRadTolerance * unplaced.fSecRMin;
              tolIRMax  = unplaced.GetRmax1() - halfRadTolerance * unplaced.fSecRMin;
              tolORMax2 = (unplaced.GetRmax1() + halfRadTolerance * unplaced.fSecRMax) *
                          (unplaced.GetRmax1() + halfRadTolerance * unplaced.fSecRMax);
            }
            else
            {
              tolORMin  = unplaced.GetRmin2() - halfRadTolerance * unplaced.fSecRMin;
              tolIRMin  = unplaced.GetRmin2() + halfRadTolerance * unplaced.fSecRMin;
              tolIRMax  = unplaced.GetRmax2() - halfRadTolerance * unplaced.fSecRMin;
              tolORMax2 = (unplaced.GetRmax2() + halfRadTolerance * unplaced.fSecRMax) *
                          (unplaced.GetRmax2() + halfRadTolerance * unplaced.fSecRMax);
            }
            if (tolORMin > 0)
            {
              tolORMin2 = tolORMin * tolORMin;
              tolIRMin2 = tolIRMin * tolIRMin;
            }
            else
            {
              tolORMin2 = 0.0;
              tolIRMin2 = 0.0;
            }
            if (tolIRMax > 0)
            {
              tolIRMax2 = tolIRMax * tolIRMax;
            }
            else
            {
              tolIRMax2 = 0.0;
            }

            if ((tolIRMin2 <= rhoi2) && (rhoi2 <= tolIRMax2))
            {
              if (! unplaced.IsFullPhi() && rhoi2)
              {
                // Psi = angle made with central (average) phi of shape

                cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi);

                if (cosPsi >= unplaced.fCosHDPhiIT * Sqrt(rhoi2))
                {
                  return sd;
                }
              }
              else
              {
                return sd;
              }
            }
          }
          else  // On/outside extent, and heading away  -> cannot intersect
          {
            return snxt;
          }
        }

      // ----> Can not intersect z surfaces


      // Intersection with outer cone (possible return) and
      //                   inner cone (must also check phi)
      //
      // Intersection point (xi,yi,zi) on line x=p.x()+t*v.x() etc.
      //
      // Intersects with x^2+y^2=(a*z+b)^2
      //
      // where a=unplaced.fTanRMax or unplaced.fTanRMin
      //       b=rMaxAv or rMinAv
      //
      // (vx^2+vy^2-(a*vz)^2)t^2+2t(pxvx+pyvy-a*vz(a*pz+b))+px^2+py^2-(a*pz+b)^2=0;
      //     t1                       t2                      t3
      //
      //  \--------u-------/       \-----------v----------/ \---------w--------/
      //

        t1   = 1.0 - v.z() * v.z();
        t2   = p.x() * v.x() + p.y() * v.y();
        t3   = p.x() * p.x() + p.y() * p.y();
        rin = unplaced.fTanRMin * p.z() + rMinAv;
        rout = unplaced.fTanRMax * p.z() + rMaxAv;

        // Outer Cone Intersection
        // Must be outside/on outer cone for valid intersection

        nt1 = t1 - (unplaced.fTanRMax * v.z()) * (unplaced.fTanRMax * v.z());
        nt2 = t2 - unplaced.fTanRMax * v.z() * rout;
        nt3 = t3 - rout * rout;

        if (Abs(nt1) > VECGEOM_NAMESPACE::kRadTolerance) // Equation quadratic => 2 roots
        {
          b = nt2 / nt1;
          c = nt3 / nt1;
          d = b * b - c ;
          if ( (rout<0) || (nt3 > rout * rout * kRadTolerance * kRadTolerance * unplaced.fSecRMax * unplaced.fSecRMax) )
          {
            // If outside real cone (should be rho-rout>kRadTolerance*0.5
            // NOT rho^2 etc) saves a Sqrt() at expense of accuracy

            if (d >= 0)
            {

              if ((rout < 0) && (nt3 <= 0))
              {
                // Inside `shadow cone' with -ve radius
                // -> 2nd root could be on real cone
                if (b > 0) {
                  sd = c / (-b - Sqrt(d));
                }
                else {
                  sd = -b + Sqrt(d);
                }
              }
              else
              {
                if ((b <= 0) && (c >= 0)) // both >=0, try smaller root
                {
                  sd = c / (-b + Sqrt(d));
                }
                else
                {
                  if (c <= 0)   // second >=0
                  {
                    sd = -b + Sqrt(d);
                  }
                  else  // both negative, travel away
                  {
                    return VECGEOM_NAMESPACE::kInfinity;
                  }
                }
              }
              if (sd > 0)    // If 'forwards'. Check z intersection
              {
                zi = p.z() + sd * v.z();
                if (Abs(zi) <= tolODz)
                {
                  // Z ok. Check phi intersection if reqd
                  if (unplaced.IsFullPhi())
                  {
                    return sd;
                  }
                  else
                  {
                    xi     = p.x() + sd * v.x();
                    yi     = p.y() + sd * v.y();
                    ri     = rMaxAv + zi * unplaced.fTanRMax;
                    cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;
                    if (cosPsi >= unplaced.fCosHDPhiIT)
                    {
                      return sd;
                    }
                  }
                }
              }               // end if (sd>0)
            }
          }
          else
          {
            // Inside outer cone
            // check not inside, and heading through UCons (-> 0 to in)
            if ((t3 > (rin + halfRadTolerance * unplaced.fSecRMin)*
                 (rin + halfRadTolerance * unplaced.fSecRMin))
                && (nt2 < 0) && (d >= 0) && (Abs(p.z()) <= tolIDz))
            {
              // Inside cones, delta r -ve, inside z extent
              // Point is on the Surface => check Direction using Normal.Dot(v)
              xi     = p.x();
              yi     = p.y()  ;
              risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
              norm = Vector3D<Precision>(xi / risec, yi / risec, -unplaced.fTanRMax * unplaced.fInvSecRMax);
              if (!unplaced.IsFullPhi())
              {
                cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi);
                if (cosPsi*cosPsi >= unplaced.fCosHDPhiIT*unplaced.fCosHDPhiIT*t3)
                {
                  if (norm.Dot(v) <= 0)
                  {
                    return 0.0;
                  }
                }
              }
              else
              {
                if (norm.Dot(v) <= 0)
                {
                  return 0.0;
                }
              }
            }
          }
        }
        else  //  Single root case
        {
          if (Abs(nt2) > VECGEOM_NAMESPACE::kRadTolerance)
          {
            sd = -0.5 * nt3 / nt2;
            if (sd < 0)
            {
              return VECGEOM_NAMESPACE::kInfinity;  // travel away
            }
            else  // sd >= 0, If 'forwards'. Check z intersection
            {
              zi = p.z() + sd * v.z();
              if ((Abs(zi) <= tolODz) && (nt2 < 0))
              {
                // Z ok. Check phi intersection if reqd
                if (unplaced.IsFullPhi())
                {
                  return sd;
                }
                else
                {
                  xi     = p.x() + sd * v.x();
                  yi     = p.y() + sd * v.y();
                  ri     = rMaxAv + zi * unplaced.fTanRMax;
                  cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                  if (cosPsi >= unplaced.fCosHDPhiIT)
                  {
                    return sd;
                  }
                }
              }
            }
          }
          else  //    travel || cone surface from its origin
          {
            sd = VECGEOM_NAMESPACE::kInfinity;
          }
        }

        // Inner Cone Intersection
        // o Space is divided into 3 areas:
        //   1) Radius greater than real inner cone & imaginary cone & outside
        //      tolerance
        //   2) Radius less than inner or imaginary cone & outside tolarance
        //   3) Within tolerance of real or imaginary cones
        //      - Extra checks needed for 3's intersections
        //        => lots of duplicated code
        if (rMinAv)
        {
          nt1 = t1 - (unplaced.fTanRMin * v.z()) * (unplaced.fTanRMin * v.z());
          nt2 = t2 - unplaced.fTanRMin * v.z() * rin;
          nt3 = t3 - rin * rin;
          if (nt1)
          {
            if (nt3 > rin * VECGEOM_NAMESPACE::kRadTolerance * unplaced.fSecRMin)
            {
              // At radius greater than real & imaginary cones
              // -> 2nd root, with zi check
              b = nt2 / nt1;
              c = nt3 / nt1;
              d = b * b - c;
              if (d >= 0)  // > 0
              {
                if (b > 0)
                {
                  sd = c / (-b - Sqrt(d));
                }
                else
                {
                  sd = -b + Sqrt(d);
                }
                if (sd >= 0)    // > 0
                {
                  zi = p.z() + sd * v.z();
                  if (Abs(zi) <= tolODz)
                  {
                    if (! unplaced.IsFullPhi() )
                    {
                      xi     = p.x() + sd * v.x();
                      yi     = p.y() + sd * v.y();
                      ri     = rMinAv + zi * unplaced.fTanRMin;
                      cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;
                      if (cosPsi >= unplaced.fCosHDPhiIT)
                      {
                        if (sd > halfRadTolerance)
                        {
                          snxt = sd;
                        }
                        else
                        {
                          // Calculate a normal vector in order to check Direction
                          risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                          norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin * unplaced.fInvSecRMin);
                          if (norm.Dot(v) <= 0)
                          {
                            snxt = sd;
                          }
                        }
                      }
                    }
                    else
                    {
                      if (sd > halfRadTolerance)
                      {
                        return sd;
                      }
                      else
                      {
                        // Calculate a normal vector in order to check Direction
                        xi     = p.x() + sd * v.x();
                        yi     = p.y() + sd * v.y();
                        risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                        norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin * unplaced.fInvSecRMin);
                        if (norm.Dot(v) <= 0)
                        {
                          return sd;
                        }
                      }
                    }
                  }
                }
              }
            }
            else  if (nt3 < -rin * VECGEOM_NAMESPACE::kRadTolerance * unplaced.fSecRMin)
            {
              // Within radius of inner cone (real or imaginary)
              // -> Try 2nd root, with checking intersection is with real cone
              // -> If check fails, try 1st root, also checking intersection is
              //    on real cone
              b = nt2 / nt1;
              c = nt3 / nt1;
              d = b * b - c;
              if (d >= 0)    // > 0
              {
                if (b > 0)
                {
                  sd = c / (-b - Sqrt(d));
                }
                else
                {
                  sd = -b + Sqrt(d);
                }
                zi = p.z() + sd * v.z();
                ri = rMinAv + zi * unplaced.fTanRMin;

                if (ri > 0)
                {
                  if ((sd >= 0) && (Abs(zi) <= tolODz))    // sd > 0
                  {
                    if (unplaced.IsFullPhi())
                    {
                      xi     = p.x() + sd * v.x();
                      yi     = p.y() + sd * v.y();
                      cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                      if (cosPsi >= unplaced.fCosHDPhiOT)
                      {
                        if (sd > halfRadTolerance)
                        {
                          snxt = sd;
                        }
                        else
                        {
                          // Calculate a normal vector in order to check Direction

                          risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                          norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin * unplaced.fInvSecRMin);
                          if (norm.Dot(v) <= 0)
                          {
                            snxt = sd;
                          }
                        }
                      }
                    }
                    else
                    {
                      if (sd > halfRadTolerance)
                      {
                        return sd;
                      }
                      else
                      {
                        // Calculate a normal vector in order to check Direction
                        xi     = p.x() + sd * v.x();
                        yi     = p.y() + sd * v.y();
                        risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                        norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin * unplaced.fInvSecRMin);
                        if (norm.Dot(v) <= 0)
                        {
                          return sd;
                        }
                      }
                    }
                  }
                }
                else
                {
                  if (b > 0)
                  {
                    sd = -b - Sqrt(d);
                  }
                  else
                  {
                    sd = c / (-b + Sqrt(d));
                  }
                  zi = p.z() + sd * v.z();
                  ri = rMinAv + zi * unplaced.fTanRMin;
                  if ((sd >= 0) && (ri > 0) && (Abs(zi) <= tolODz))   // sd>0
                  {
                    if (unplaced.IsFullPhi())
                    {
                      xi     = p.x() + sd * v.x();
                      yi     = p.y() + sd * v.y();
                      cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;
                      if (cosPsi >= unplaced.fCosHDPhiIT)
                      {
                        if (sd > halfRadTolerance)
                        {
                          snxt = sd;
                        }
                        else
                        {
                          // Calculate a normal vector in order to check Direction
                          risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                          norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin * unplaced.fInvSecRMin);
                          if (norm.Dot(v) <= 0)
                          {
                            snxt = sd;
                          }
                        }
                      }
                    }
                    else
                    {
                      if (sd > halfRadTolerance)
                      {
                        return sd;
                      }
                      else
                      {
                        // Calculate a normal vector in order to check Direction
                        xi     = p.x() + sd * v.x();
                        yi     = p.y() + sd * v.y();
                        risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                        norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin * unplaced.fInvSecRMin);
                        if (norm.Dot(v) <= 0)
                        {
                          return sd;
                        }
                      }
                    }
                  }
                }
              }
            }
            else
            {
              // Within kRadTol*0.5 of inner cone (real OR imaginary)
              // ----> Check not travelling through (=>0 to in)
              // ----> if not:
              //    -2nd root with validity check

              if (Abs(p.z()) <= tolODz)
              {
                if (nt2 > 0)
                {
                  // Inside inner real cone, heading outwards, inside z range

                  if (!unplaced.IsFullPhi())
                  {
                    cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi);

                    if (cosPsi*cosPsi >= unplaced.fCosHDPhiIT*unplaced.fCosHDPhiIT*t3 )
                    {
                      return 0.0;
                    }
                  }
                  else
                  {
                    return 0.0;
                  }
                }
                else
                {
                  // Within z extent, but not travelling through
                  // -> 2nd root or VECGEOM_NAMESPACE::kInfinity if 1st root on imaginary cone

                  b = nt2 / nt1;
                  c = nt3 / nt1;
                  d = b * b - c;

                  if (d >= 0)     // > 0
                  {
                    if (b > 0)
                    {
                      sd = -b - Sqrt(d);
                    }
                    else
                    {
                      sd = c / (-b + Sqrt(d));
                    }
                    zi = p.z() + sd * v.z();
                    ri = rMinAv + zi * unplaced.fTanRMin;

                    if (ri > 0)     // 2nd root
                    {
                      if (b > 0)
                      {
                        sd = c / (-b - Sqrt(d));
                      }
                      else
                      {
                        sd = -b + Sqrt(d);
                      }

                      zi = p.z() + sd * v.z();

                      if ((sd >= 0) && (Abs(zi) <= tolODz))    // sd>0
                      {
                        if (!unplaced.IsFullPhi())
                        {
                          xi     = p.x() + sd * v.x();
                          yi     = p.y() + sd * v.y();
                          ri     = rMinAv + zi * unplaced.fTanRMin;
                          cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                          if (cosPsi >= unplaced.fCosHDPhiIT)
                          {
                            snxt = sd;
                          }
                        }
                        else
                        {
                          return sd;
                        }
                      }
                    }
                    else
                    {
                      return VECGEOM_NAMESPACE::kInfinity;
                    }
                  }
                }
              }
              else   // 2nd root
              {
                b = nt2 / nt1;
                c = nt3 / nt1;
                d = b * b - c;

                if (d > 0)
                {
                  if (b > 0)
                  {
                    sd = c / (-b - Sqrt(d));
                  }
                  else
                  {
                    sd = -b + Sqrt(d);
                  }
                  zi = p.z() + sd * v.z();

                  if ((sd >= 0) && (Abs(zi) <= tolODz))    // sd>0
                  {
                    if (!unplaced.IsFullPhi())
                    {
                      xi     = p.x() + sd * v.x();
                      yi     = p.y() + sd * v.y();
                      ri     = rMinAv + zi * unplaced.fTanRMin;
                      cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / ri;

                      if (cosPsi >= unplaced.fCosHDPhiIT)
                      {
                        snxt = sd;
                      }
                    }
                    else
                    {
                      return sd;
                    }
                  }
                }
              }
            }
          }
        }

        // Phi segment intersection
        //
        // o Tolerant of points inside phi planes by up to VECGEOM_NAMESPACE::kTolerance*0.5
        //
        // o NOTE: Large duplication of code between sphi & ephi checks
        //         -> only diffs: sphi -> ephi, Comp -> -Comp and half-plane
        //            intersection check <=0 -> >=0
        //         -> Should use some form of loop Construct

        if (!unplaced.IsFullPhi())
        {
          // First phi surface (starting phi)

          Comp    = v.x() * unplaced.fSinSPhi - v.y() * unplaced.fCosSPhi;

          if (Comp < 0)      // Component in outwards normal dirn
          {
            Dist = (p.y() * unplaced.fCosSPhi - p.x() * unplaced.fSinSPhi);

            if (Dist < halfCarTolerance)
            {
              sd = Dist / Comp;

              if (sd < snxt)
              {
                if (sd < 0)
                {
                  sd = 0.0;
                }

                zi = p.z() + sd * v.z();

                if (Abs(zi) <= tolODz)
                {
                  xi        = p.x() + sd * v.x();
                  yi        = p.y() + sd * v.y();
                  rhoi2    = xi * xi + yi * yi;
                  tolORMin2 = (rMinOAv + zi * unplaced.fTanRMin) * (rMinOAv + zi * unplaced.fTanRMin);
                  tolORMax2 = (rMaxOAv + zi * unplaced.fTanRMax) * (rMaxOAv + zi * unplaced.fTanRMax);

                  if ((rhoi2 >= tolORMin2) && (rhoi2 <= tolORMax2))
                  {
                    // z and r intersections good - check intersecting with
                    // correct half-plane

                    if ((yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi) <= 0)
                    {
                      snxt = sd;
                    }
                  }
                }
              }
            }
          }

          // Second phi surface (Ending phi)

          Comp    = -(v.x() * unplaced.fSinEPhi - v.y() * unplaced.fCosEPhi);

          if (Comp < 0)     // Component in outwards normal dirn
          {
            Dist = -(p.y() * unplaced.fCosEPhi - p.x() * unplaced.fSinEPhi);
            if (Dist < halfCarTolerance)
            {
              sd = Dist / Comp;

              if (sd < snxt)
              {
                if (sd < 0)
                {
                  sd = 0.0;
                }

                zi = p.z() + sd * v.z();

                if (Abs(zi) <= tolODz)
                {
                  xi        = p.x() + sd * v.x();
                  yi        = p.y() + sd * v.y();
                  rhoi2    = xi * xi + yi * yi;
                  tolORMin2 = (rMinOAv + zi * unplaced.fTanRMin) * (rMinOAv + zi * unplaced.fTanRMin);
                  tolORMax2 = (rMaxOAv + zi * unplaced.fTanRMax) * (rMaxOAv + zi * unplaced.fTanRMax);

                  if ((rhoi2 >= tolORMin2) && (rhoi2 <= tolORMax2))
                  {
                    // z and r intersections good - check intersecting with
                    // correct half-plane

                    if ((yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi) >= 0.0)
                    {
                      snxt = sd;
                    }
                  }
                }
              }
            }
          }
        }
        if (snxt < halfCarTolerance)
        {
          snxt = 0.;
        }
        return snxt;
  }


  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToIn(
      UnplacedCone const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

   // TOBEIMPLEMENTED
    distance = DistanceToInUSolids<Backend>(unplaced, transformation, point, direction, stepMax);

//    typedef typename Backend::bool_v MaskType;
//    typedef typename Backend::precision_v VectorType;
//    typedef typename Vector3D<typename Backend::precision_v> Vector3D;
//
//    MaskType done_m(false); // which particles in the vector are ready to be returned == aka have been treated
//    distance = kInfinity; // initialize distance to infinity
//
//    // TODO: check that compiler is doing same thing
//    // as if we used a combined point + direction transformation
//    Vector3D localpoint;
//    localpoint=transformation.Transform<transCodeT,rotCodeT>(point);
//    Vector3D localdir; // VectorType dirx, diry, dirz;
//    localdir=transformation.TransformDirection<transCodeT,rotCodeT>(localdir);
//
//    // do some inside checks
//    // if safez is > 0 it means that particle is within z range
//    // if safez is < 0 it means that particle is outside z range
//
//    VectorType z = localpoint.z();
//    VectorType safez = unplaced.GetDz() - Abs(z);
//    MaskType inz_m = safez > Utils::fgToleranceVc;
//    VectorType dirz = localdir.z();
//    done_m = !inz_m && ( z*dirz >= 0 ); // particle outside the z-range and moving away
//
//    VectorType x=localpoint.x();
//    VectorType y=localpoint.y();
//    VectorType r2 = x*x + y*y; // use of Perp2
//    VectorType n2 = VectorType(1)-(unplaced.GetOuterSlopeSquare() + 1) *dirz*dirz; // dirx_v*dirx_v + diry_v*diry_v; ( dir is normalized !! )
//    VectorType dirx = localdir.x();
//    VectorType diry = localdir.y();
//    VectorType rdotnplanar = x*dirx + y*diry; // use of PerpDot
//
//    // T a = 1 - dir.z*dir.z*(1+m*m);
//    // T b = 2 * ( pos.x*dir.x + pos.y*dir.y - m*m*pos.z*dir.z - m*n*dir.z);
//    // T c = ( pos.x*pos.x + pos.y*pos.y - m*m*pos.z*pos.z - 2*m*n*pos.z - n*n );
//
//    // QUICK CHECK IF OUTER RADIUS CAN BE HIT AT ALL
//    // BELOW WE WILL SOLVE A QUADRATIC EQUATION OF THE TYPE
//    // a * t^2 + b * t + c = 0
//    // if this equation has a solution at all ( == hit )
//    // the following condition needs to be satisfied
//    // DISCRIMINANT = b^2 -  4 a*c > 0
//    //
//    // THIS CONDITION DOES NOT NEED ANY EXPENSIVE OPERATION !!
//    //
//    // then the solutions will be given by
//    //
//    // t = (-b +- SQRT(DISCRIMINANT)) / (2a)
//    //
//    // b = 2*(dirx*x + diry*y)  -- independent of shape
//    // a = dirx*dirx + diry*diry -- independent of shape
//    // c = x*x + y*y - R^2 = r2 - R^2 -- dependent on shape
//    VectorType c = r2 - unplaced.GetOuterSlopeSquare()*z*z - 2*unplaced.GetOuterSlope()*unplaced.GetOuterOffset() * z - unplaced.GetOuterOffsetSquare();
//
//    VectorType a = n2;
//
//    VectorType b = 2*(rdotnplanar - z*dirz*unplaced.GetOuterSlopeSquare() - unplaced.GetOuterSlope()*unplaced.GetOuterOffset()*dirz);
//    VectorType discriminant = b*b-4*a*c;
//    MaskType   canhitrmax = ( discriminant >= 0 );
//
//    done_m |= ! canhitrmax;
//
//    // this might be optional
//    if( done_m.isFull() )
//    {
//           // joint z-away or no chance to hit Rmax condition
//     #ifdef LOG_EARLYRETURNS
//           std::cerr << " RETURN1 IN DISTANCETOIN " << std::endl;
//     #endif
//           return;
//    }
//
//    // Check outer cylinder (only r>rmax has to be considered)
//    // this IS ALWAYS the MINUS (-) solution
//    VectorType distanceRmax( Utils::kInfinityVc );
//    distanceRmax( canhitrmax ) = (-b - Sqrt( discriminant ))/(2.*a);
//
//    // this determines which vectors are done here already
//    MaskType Rdone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmax );
//    distanceRmax( ! Rdone ) = Utils::kInfinityVc;
//    MaskType rmindone;
//    // **** inner tube ***** only compiled in for tubes having inner hollow cone -- or in case of a universal runtime shape ******/
//    if ( checkRminTreatment<ConeType>(unplaced) )
//    {
//       // in case of the Cone, generally all coefficients a, b, and c change
//       a = 1.-(unplaced.GetInnerSlopeSquare() + 1) *dirz*dirz;
//       c = r2 - unplaced.GetInnerSlopeSquare()*z*z - 2*unplaced.GetInnerSlope()*unplaced.GetInnerOffset() * z
//               - unplaced.GetInnerOffsetSquare();
//       b = 2*(rdotnplanar - dirz*(z*unplaced.GetInnerSlopeSquare + unplaced.GetInnerOffset* unplaced.GetInnerSlope()));
//       discriminant =  b*b-4*a*c;
//       MaskType canhitrmin = ( discriminant >= Vc::Zero );
//       VectorType distanceRmin ( Utils::kInfinityVc );
//       // this is always + solution
//       distanceRmin ( canhitrmin ) = (-b + Vc::sqrt( discriminant ))/(2*a);
//       rmindone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmin );
//       distanceRmin ( ! rmindone ) = Utils::kInfinity;
//
//       // reduction of distances
//       distanceRmax = Vc::min( distanceRmax, distanceRmin );
//       Rdone |= rmindone;
//     }
//        //distance( ! done_m && Rdone ) = distanceRmax;
//        //done_m |= Rdone;
//
//        /* might check early here */
//
//        // now do Z-Face
//        VectorType distancez = -safez/Vc::abs(dirz);
//        MaskType zdone = determineZHit(x,y,z,dirx,diry,dirz,distancez);
//        distance ( ! done_m && zdone ) = distancez;
//        distance ( ! done_m && ! zdone && Rdone ) = distanceRmax;
//        done_m |= ( zdone ) || (!zdone && (Rdone));
//
//        // now PHI
//
//        // **** PHI TREATMENT FOR CASE OF HAVING RMAX ONLY ***** only compiled in for cones having phi sektion ***** //
//        if ( ConeTraits::NeedsPhiTreatment<ConeType>::value )
//        {
//           // all particles not done until here have the potential to hit a phi surface
//           // phi surfaces require divisions so it might be useful to check before continuing
//
//           if( ConeTraits::NeedsRminTreatment<ConeType>::value || ! done_m.isFull() )
//           {
//              VectorType distphi;
//              ConeUtils::DistanceToPhiPlanes<ValueType,ConeTraits::IsPhiEqualsPiCase<ConeType>::value,ConeTraits::NeedsRminTreatment<ConeType>::value>(coneparams->dZ,
//                    coneparams->outerslope, coneparams->outeroffset,
//                    coneparams->innerslope, coneparams->inneroffset,
//                    coneparams->normalPhi1.x, coneparams->normalPhi1.y, coneparams->normalPhi2.x, coneparams->normalPhi2.y,
//                    coneparams->alongPhi1, coneparams->alongPhi2,
//                    x, y, z, dirx, diry, dirz, distphi);
//              if(ConeTraits::NeedsRminTreatment<ConeType>::value)
//              {
//                 // distance(! done_m || (rmindone && ! inrmin_m ) || (rmaxdone && ) ) = distphi;
//                 // distance ( ! done_m ) = distphi;
//                 distance = Vc::min(distance, distphi);
//              }
//              else
//              {
//                 distance ( ! done_m ) = distphi;
//              }
//           }
//        }

 }

  template <class Backend>
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   static Precision DistanceToOutUSOLIDS(
       UnplacedCone const &unplaced,
       Vector3D<typename Backend::precision_v> p,
       Vector3D<typename Backend::precision_v> v,
       typename Backend::precision_v const &stepMax
       ) {

       const double halfCarTolerance = VECGEOM_NAMESPACE::kHalfTolerance;
       const double halfRadTolerance = kRadTolerance * 0.5;
       const double halfAngTolerance = kAngTolerance * 0.5;

        double snxt, srd, sphi, pdist;

        double rMaxAv;  // Data for outer cone
        double rMinAv;  // Data for inner cone

        double t1, t2, t3, rout, rin, nt1, nt2, nt3;
        double b, c, d, sr2, sr3;

        // Vars for intersection within tolerance

      //  ESide   sidetol = kNull;
      //  double slentol = VECGEOM_NAMESPACE::kInfinity;

        // Vars for phi intersection:

        double pDistS, compS, pDistE, compE, sphi2, xi, yi, /*risec,*/ vphi;
        double zi, ri, deltaRoi2;

        // Z plane intersection

        if (v.z() > 0.0)
        {
          pdist = unplaced.GetDz() - p.z();

          if (pdist > halfCarTolerance)
          {
            snxt = pdist / v.z();
         //   side = kPZ;
          }
          else
          {
          //  aNormalVector        = UVector3(0, 0, 1);
          //  aConvex = true;
            return  snxt = 0.0;
          }
        }
        else if (v.z() < 0.0)
        {
          pdist = unplaced.GetDz() + p.z();

          if (pdist > halfCarTolerance)
          {
            snxt = -pdist / v.z();
           // side = kMZ;
          }
          else
          {
           // aNormalVector        = UVector3(0, 0, -1);
           // aConvex = true;
            return snxt = 0.0;
          }
        }
        else     // Travel perpendicular to z axis
        {
          snxt = VECGEOM_NAMESPACE::kInfinity;
         // side = kNull;
        }

        // Radial Intersections
        //
        // Intersection with outer cone (possible return) and
        //                   inner cone (must also check phi)
        //
        // Intersection point (xi,yi,zi) on line x=p.x()+t*v.x() etc.
        //
        // Intersects with x^2+y^2=(a*z+b)^2
        //
        // where a=unplaced.fTanRMax or unplaced.fTanRMin
        //       b=rMaxAv or rMinAv
        //
        // (vx^2+vy^2-(a*vz)^2)t^2+2t(pxvx+pyvy-a*vz(a*pz+b))+px^2+py^2-(a*pz+b)^2=0;
        //     t1                       t2                      t3
        //
        //  \--------u-------/       \-----------v----------/ \---------w--------/

        rMaxAv  = (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;

        t1   = 1.0 - v.z() * v.z();   // since v normalised
        t2   = p.x() * v.x() + p.y() * v.y();
        t3   = p.x() * p.x() + p.y() * p.y();
        rout = unplaced.fTanRMax * p.z() + rMaxAv;

        nt1 = t1 - (unplaced.fTanRMax * v.z()) * (unplaced.fTanRMax * v.z());
        nt2 = t2 - unplaced.fTanRMax * v.z() * rout;
        nt3 = t3 - rout * rout;

        if (v.z() > 0.0)
        {
          deltaRoi2 = snxt * snxt * t1 + 2 * snxt * t2 + t3
                      - unplaced.GetRmax2() * (unplaced.GetRmax2() + kRadTolerance * unplaced.fSecRMax);
        }
        else if (v.z() < 0.0)
        {
          deltaRoi2 = snxt * snxt * t1 + 2 * snxt * t2 + t3
                      - unplaced.GetRmax1() * (unplaced.GetRmax1() + kRadTolerance * unplaced.fSecRMax);
        }
        else
        {
          deltaRoi2 = 1.0;
        }

        if (nt1 && (deltaRoi2 > 0.0))
        {
          // Equation quadratic => 2 roots : second root must be leaving

          b = nt2 / nt1;
          c = nt3 / nt1;
          d = b * b - c;

          if (d >= 0)
          {
            // Check if on outer cone & heading outwards
            // NOTE: Should use rho-rout>-kRadTolerance*0.5

            if (nt3 > -halfRadTolerance && nt2 >= 0)
            {
	       //              risec     = Sqrt(t3) * unplaced.fSecRMax;
             // aConvex = true;
             // aNormalVector        = UVector3(p.x() / risec, p.y() / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
              return snxt = 0;
            }
            else
            {
             // sider = kRMax ;
              if (b > 0)
              {
                srd = -b - Sqrt(d);
              }
              else
              {
                srd = c / (-b + Sqrt(d));
              }

              zi    = p.z() + srd * v.z();
              ri    = unplaced.fTanRMax * zi + rMaxAv;

              if ((ri >= 0) && (-halfRadTolerance <= srd) && (srd <= halfRadTolerance))
              {
                // An intersection within the tolerance
                //   we will Store it in case it is good -
                //
               // slentol = srd;
               // sidetol = kRMax;
              }
              if ((ri < 0) || (srd < halfRadTolerance))
              {
                // Safety: if both roots -ve ensure that srd cannot `win'
                //         distance to out

                if (b > 0)
                {
                  sr2 = c / (-b - Sqrt(d));
                }
                else
                {
                  sr2 = -b + Sqrt(d);
                }
                zi  = p.z() + sr2 * v.z();
                ri  = unplaced.fTanRMax * zi + rMaxAv;

                if ((ri >= 0) && (sr2 > halfRadTolerance))
                {
                  srd = sr2;
                }
                else
                {
                  srd = VECGEOM_NAMESPACE::kInfinity;

                  if ((-halfRadTolerance <= sr2) && (sr2 <= halfRadTolerance))
                  {
                    // An intersection within the tolerance.
                    // Storing it in case it is good.

                 //   slentol = sr2;
                  //  sidetol = kRMax;
                  }
                }
              }
            }
          }
          else
          {
            // No intersection with outer cone & not parallel
            // -> already outside, no intersection

            //risec     = Sqrt(t3) * unplaced.fSecRMax;
           // aConvex = true;
           // aNormalVector        = UVector3(p.x() / risec, p.y() / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
            return snxt = 0.0;
          }
        }
        else if (nt2 && (deltaRoi2 > 0.0))
        {
          // Linear case (only one intersection) => point outside outer cone

	   //          risec     = Sqrt(t3) * unplaced.fSecRMax;
         // aConvex = true;
         // aNormalVector        = UVector3(p.x() / risec, p.y() / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
          return snxt = 0.0;
        }
        else
        {
          // No intersection -> parallel to outer cone
          // => Z or inner cone intersection

          srd = VECGEOM_NAMESPACE::kInfinity;
        }

        // Check possible intersection within tolerance

        /*
        if (slentol <= halfCarTolerance)
        {
          // An intersection within the tolerance was found.
          // We must accept it only if the momentum points outwards.
          //
          // UVector3 ptTol;  // The point of the intersection
          // ptTol= p + slentol*v;
          // ri=unplaced.fTanRMax*zi+rMaxAv;
          //
          // Calculate a normal vector, as below

          xi    = p.x() + slentol * v.x();
          yi    = p.y() + slentol * v.y();
          risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
          UVector3 norm = UVector3(xi / risec, yi / risec, -unplaced.fTanRMax / unplaced.fSecRMax);

          if (norm.Dot(v) > 0)     // We will leave the Cone immediatelly
          {
            aNormalVector        = norm.Unit();
            aConvex = true;
            return snxt = 0.0;
          }
          else // On the surface, but not heading out so we ignore this intersection
          {
            //                                        (as it is within tolerance).
            slentol = VECGEOM_NAMESPACE::kInfinity;
          }

        }
*/

        // Inner Cone intersection

        if (unplaced.GetRmin1() || unplaced.GetRmin2())
        {
          nt1    = t1 - (unplaced.fTanRMin * v.z()) * (unplaced.fTanRMin * v.z());

          if (nt1)
          {
            rMinAv  = (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;
            rin    = unplaced.fTanRMin * p.z() + rMinAv;
            nt2    = t2 - unplaced.fTanRMin * v.z() * rin;
            nt3    = t3 - rin * rin;

            // Equation quadratic => 2 roots : first root must be leaving

            b = nt2 / nt1;
            c = nt3 / nt1;
            d = b * b - c;

            if (d >= 0.0)
            {
              // NOTE: should be rho-rin<kRadTolerance*0.5,
              //       but using squared versions for efficiency

              if (nt3 < kRadTolerance * (rin + kRadTolerance * 0.25))
              {
                if (nt2 < 0.0)
                {
                 // aConvex = false;
                 // risec = Sqrt(p.x() * p.x() + p.y() * p.y()) * unplaced.fSecRMin;
                 // aNormalVector = UVector3(-p.x() / risec, -p.y() / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                  return          snxt      = 0.0;
                }
              }
              else
              {
                if (b > 0)
                {
                  sr2 = -b - Sqrt(d);
                }
                else
                {
                  sr2 = c / (-b + Sqrt(d));
                }
                zi  = p.z() + sr2 * v.z();
                ri  = unplaced.fTanRMin * zi + rMinAv;

                if ((ri >= 0.0) && (-halfRadTolerance <= sr2) && (sr2 <= halfRadTolerance))
                {
                  // An intersection within the tolerance
                  // storing it in case it is good.

                 // slentol = sr2;
                 // sidetol = kRMax;
                }
                if ((ri < 0) || (sr2 < halfRadTolerance))
                {
                  if (b > 0)
                  {
                    sr3 = c / (-b - Sqrt(d));
                  }
                  else
                  {
                    sr3 = -b + Sqrt(d);
                  }

                  // Safety: if both roots -ve ensure that srd cannot `win'
                  //         distancetoout

                  if (sr3 > halfRadTolerance)
                  {
                    if (sr3 < srd)
                    {
                      zi = p.z() + sr3 * v.z();
                      ri = unplaced.fTanRMin * zi + rMinAv;

                      if (ri >= 0.0)
                      {
                        srd = sr3;
                   //     sider = kRMin;
                      }
                    }
                  }
                  else if (sr3 > -halfRadTolerance)
                  {
                    // Intersection in tolerance. Store to check if it's good

               //     slentol = sr3;
                //    sidetol = kRMin;
                  }
                }
                else if ((sr2 < srd) && (sr2 > halfCarTolerance))
                {
                  srd  = sr2;
                //  sider = kRMin;
                }
                else if (sr2 > -halfCarTolerance)
                {
                  // Intersection in tolerance. Store to check if it's good

                //  slentol = sr2;
                //  sidetol = kRMin;
                }

                /*
                if (slentol <= halfCarTolerance)
                {
                  // An intersection within the tolerance was found.
                  // We must accept it only if  the momentum points outwards.

                  UVector3 norm;

                  // Calculate a normal vector, as below

                  xi     = p.x() + slentol * v.x();
                  yi     = p.y() + slentol * v.y();
                  if (sidetol == kRMax)
                  {
                    risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
                    norm = UVector3(xi / risec, yi / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
                  }
                  else
                  {
                    risec = Sqrt(xi * xi + yi * yi) * unplaced.fSecRMin;
                    norm = UVector3(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
                  }
                  if (norm.Dot(v) > 0)
                  {
                    // We will leave the cone immediately

                    aNormalVector        = norm.Unit();
                    aConvex = true;
                    return snxt = 0.0;
                  }
                  else
                  {
                    // On the surface, but not heading out so we ignore this
                    // intersection (as it is within tolerance).

                    slentol = VECGEOM_NAMESPACE::kInfinity;
                  }
                }*/
              }
            }
          }
        }

        // Linear case => point outside inner cone ---> outer cone intersect
        //
        // Phi Intersection

        if (!unplaced.IsFullPhi())
        {
          // add angle calculation with correction
          // of the difference in domain of atan2 and Sphi

          vphi = ATan2(v.y(), v.x());

          if (vphi < unplaced.GetSPhi() - halfAngTolerance)
          {
            vphi += 2 * VECGEOM_NAMESPACE::kPi;
          }
          else if (vphi > unplaced.GetSPhi() + unplaced.GetDPhi() + halfAngTolerance)
          {
            vphi -= 2 * VECGEOM_NAMESPACE::kPi;
          }

          if (p.x() || p.y())     // Check if on z axis (rho not needed later)
          {
            // pDist -ve when inside

            pDistS = p.x() * unplaced.fSinSPhi - p.y() * unplaced.fCosSPhi;
            pDistE = -p.x() * unplaced.fSinEPhi + p.y() * unplaced.fCosEPhi;

            // Comp -ve when in direction of outwards normal

            compS = -unplaced.fSinSPhi * v.x() + unplaced.fCosSPhi * v.y();
            compE = unplaced.fSinEPhi * v.x() - unplaced.fCosEPhi * v.y();

        //    sidephi = kNull;

            if (((unplaced.GetDPhi() <= VECGEOM_NAMESPACE::kPi) && ((pDistS <= halfCarTolerance)
                                            && (pDistE <= halfCarTolerance)))
                || ((unplaced.GetDPhi() > VECGEOM_NAMESPACE::kPi) && !((pDistS > halfCarTolerance)
                                               && (pDistE >  halfCarTolerance))))
            {
              // Inside both phi *full* planes
              if (compS < 0)
              {
                sphi = pDistS / compS;
                if (sphi >= -halfCarTolerance)
                {
                  xi = p.x() + sphi * v.x();
                  yi = p.y() + sphi * v.y();

                  // Check intersecting with correct half-plane
                  // (if not -> no intersect)
                  //
                  if ((Abs(xi) <= VECGEOM_NAMESPACE::kTolerance)
                      && (Abs(yi) <= VECGEOM_NAMESPACE::kTolerance))
                  {
                   // sidephi = kSPhi;
                    if ((unplaced.GetSPhi() - halfAngTolerance <= vphi)
                        && (unplaced.GetSPhi() + unplaced.GetDPhi() + halfAngTolerance >= vphi))
                    {
                      sphi = VECGEOM_NAMESPACE::kInfinity;
                    }
                  }
                  else if ((yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi) >= 0)
                  {
                    sphi = VECGEOM_NAMESPACE::kInfinity;
                  }
                  else
                  {
                   // sidephi = kSPhi;
                    if (pDistS > -halfCarTolerance)
                    {
                      sphi = 0.0; // Leave by sphi immediately
                    }
                  }
                }
                else
                {
                  sphi = VECGEOM_NAMESPACE::kInfinity;
                }
              }
              else
              {
                sphi = VECGEOM_NAMESPACE::kInfinity;
              }

              if (compE < 0)
              {
                sphi2 = pDistE / compE;

                // Only check further if < starting phi intersection
                //
                if ((sphi2 > -halfCarTolerance) && (sphi2 < sphi))
                {
                  xi = p.x() + sphi2 * v.x();
                  yi = p.y() + sphi2 * v.y();

                  // Check intersecting with correct half-plane

                  if ((Abs(xi) <= VECGEOM_NAMESPACE::kTolerance)
                      && (Abs(yi) <= VECGEOM_NAMESPACE::kTolerance))
                  {
                    // Leaving via ending phi

                    if (!((unplaced.GetSPhi() - halfAngTolerance <= vphi)
                          && (unplaced.GetSPhi() + unplaced.GetDPhi() + halfAngTolerance >= vphi)))
                    {
                    //  sidephi = kEPhi;
                      if (pDistE <= -halfCarTolerance)
                      {
                        sphi = sphi2;
                      }
                      else
                      {
                        sphi = 0.0;
                      }
                    }
                  }
                  else // Check intersecting with correct half-plane
                    if (yi * unplaced.fCosCPhi - xi * unplaced.fSinCPhi >= 0)
                    {
                      // Leaving via ending phi

                   //   sidephi = kEPhi;
                      if (pDistE <= -halfCarTolerance)
                      {
                        sphi = sphi2;
                      }
                      else
                      {
                        sphi = 0.0;
                      }
                    }
                }
              }
            }
            else
            {
              sphi = VECGEOM_NAMESPACE::kInfinity;
            }
          }
          else
          {
            // On z axis + travel not || to z axis -> if phi of vector direction
            // within phi of shape, Step limited by rmax, else Step =0

            if ((unplaced.GetSPhi() - halfAngTolerance <= vphi)
                && (vphi <= unplaced.GetSPhi() + unplaced.GetDPhi() + halfAngTolerance))
            {
              sphi = VECGEOM_NAMESPACE::kInfinity;
            }
            else
            {
             // sidephi = kSPhi ;  // arbitrary
              sphi    = 0.0;
            }
          }
          if (sphi < snxt)   // Order intersecttions
          {
            snxt = sphi;
        //    side = sidephi;
          }
        }
        if (srd < snxt)    // Order intersections
        {
          snxt = srd  ;
         // side = sider;
        }
        if (snxt < halfCarTolerance)
        {
          snxt = 0.;
        }

        return snxt;

  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  VECGEOM_INLINE
  static void DistanceToOut(
      UnplacedCone const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

   // TOBEIMPLEMENTED

   // has to be implemented in a way
   // as to be tolerant when particle is outside
    distance = DistanceToOutUSOLIDS<Backend>( unplaced, point, direction, stepMax );
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static Precision SafetyToInUSOLIDS(UnplacedCone const &unplaced,
                          Transformation3D const &transformation,
                          Vector3D<typename Backend::precision_v> const &point
                          ) {
      double safe = 0.0, rho, safeR1, safeR2, safeZ, safePhi, cosPsi;
      double pRMin, pRMax;

      // need a transformation
      Vector3D<Precision> p = transformation.Transform<transCodeT,rotCodeT>(point);

      rho  = Sqrt(p.x() * p.x() + p.y() * p.y());
      safeZ = Abs(p.z()) - unplaced.GetDz();
      safeR1 = 0; safeR2 = 0;

      if ( unplaced.GetRmin1() || unplaced.GetRmin2())
       {
         pRMin  = unplaced.fTanRMin * p.z() + (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;
         safeR1  = (pRMin - rho) * unplaced.fInvSecRMin;

         pRMax  = unplaced.fTanRMax * p.z() + (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
         safeR2  = (rho - pRMax) * unplaced.fInvSecRMax;

         if (safeR1 > safeR2)
         {
           safe = safeR1;
         }
         else
         {
           safe = safeR2;
         }
       }
       else
       {
         pRMax  = unplaced.fTanRMax * p.z() + (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
         safe    = (rho - pRMax) * unplaced.fInvSecRMax;
       }
       if (safeZ > safe)
       {
         safe = safeZ;
       }

       if (!unplaced.IsFullPhi() && rho)
       {
         // Psi=angle from central phi to point

         cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi);

         if (cosPsi < unplaced.fCosHDPhi*rho ) // Point lies outside phi range
         {
           if ((p.y() * unplaced.fCosCPhi - p.x() * unplaced.fSinCPhi) <= 0.0)
           {
             safePhi = Abs(p.x() * unplaced.fSinSPhi - p.y() * unplaced.fCosSPhi);
           }
           else
           {
             safePhi = Abs(p.x() * unplaced.fSinEPhi - p.y() * unplaced.fCosEPhi);
           }
           if (safePhi > safe)
           {
             safe = safePhi;
           }
         }
       }
       if (safe < 0.0)
       {
         safe = 0.0; return safe; //point is Inside
       }
       return safe;
       /*
       if (!aAccurate) return safe;

       double safsq = 0.0;
       int count = 0;
       if (safeR1 > 0)
       {
         safsq += safeR1 * safeR1;
         count++;
       }
       if (safeR2 > 0)
       {
         safsq += safeR2 * safeR2;
         count++;
       }
       if (safeZ > 0)
       {
         safsq += safeZ * safeZ;
         count++;
       }
       if (count == 1) return safe;
       return Sqrt(safsq);
*/

  }


  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedCone const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

    // TOBEIMPLEMENTED -- momentarily dispatching to USolids
    safety = SafetyToInUSOLIDS<Backend>(unplaced,transformation,point);
  }

  template<class Backend>
   VECGEOM_CUDA_HEADER_BOTH
   VECGEOM_INLINE
   static Precision SafetyToOutUSOLIDS(UnplacedCone const &unplaced,
                           Vector3D<typename Backend::precision_v> p) {
      double safe = 0.0, rho, safeR1, safeR2, safeZ, safePhi;
       double pRMin;
       double pRMax;


      rho = Sqrt(p.x() * p.x() + p.y() * p.y());
      safeZ = unplaced.GetDz() - Abs(p.z());

        if (unplaced.GetRmin1() || unplaced.GetRmin2())
        {
          pRMin  = unplaced.fTanRMin * p.z() + (unplaced.GetRmin1() + unplaced.GetRmin2()) * 0.5;
          safeR1  = (rho - pRMin) * unplaced.fInvSecRMin;
        }
        else
        {
          safeR1 = VECGEOM_NAMESPACE::kInfinity;
        }

        pRMax  = unplaced.fTanRMax * p.z() + (unplaced.GetRmax1() + unplaced.GetRmax2()) * 0.5;
        safeR2  = (pRMax - rho) * unplaced.fInvSecRMax;

        if (safeR1 < safeR2)
        {
          safe = safeR1;
        }
        else
        {
          safe = safeR2;
        }
        if (safeZ < safe)
        {
          safe = safeZ;
        }

        // Check if phi divided, Calc distances closest phi plane

        if (!unplaced.IsFullPhi())
        {
          // Above/below central phi of UCons?

          if ((p.y() * unplaced.fCosCPhi - p.x() * unplaced.fSinCPhi) <= 0)
          {
            safePhi = -(p.x() * unplaced.fSinSPhi - p.y() * unplaced.fCosSPhi);
          }
          else
          {
            safePhi = (p.x() * unplaced.fSinEPhi - p.y() * unplaced.fCosEPhi);
          }
          if (safePhi < safe)
          {
            safe = safePhi;
          }
        }
        if (safe < 0)
        {
          safe = 0;
        }

        return safe;
   }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedCone const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {

    safety = SafetyToOutUSOLIDS<Backend>(unplaced,point);
  }


  // normal kernel
  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void NormalKernel(
       UnplacedCone const &unplaced,
       Vector3D<typename Backend::precision_v> const &p,
       Vector3D<typename Backend::precision_v> &normal,
       typename Backend::bool_v &valid
      )
  {
      // TODO: provide generic vectorized implementation
      // TODO: tranform point p to local coordinates

      int noSurfaces = 0;
      double rho, pPhi;
      double distZ, distRMin, distRMax;
      double distSPhi = kInfinity, distEPhi = kInfinity;
      double pRMin, widRMin;
      double pRMax, widRMax;

      const double delta = 0.5 * kTolerance;
      const double dAngle = 0.5 * kAngTolerance;
      typedef Vector3D<typename Backend::precision_v> Vec3D_t;
      Vec3D_t norm, sumnorm(0., 0., 0.), nZ(0., 0., 1.);
      Vec3D_t nR, nr(0., 0., 0.), nPs, nPe;

      distZ = Abs(Abs(p.z()) - unplaced.GetDz());
      rho  = Sqrt(p.x() * p.x() + p.y() * p.y());

      pRMin   = rho - p.z() * unplaced.fTanRMin;
      widRMin = unplaced.GetRmin2() - unplaced.GetDz() * unplaced.fTanRMin;
      distRMin = Abs(pRMin - widRMin) * unplaced.fInvSecRMin;

      pRMax   = rho - p.z() * unplaced.fTanRMax;
      widRMax = unplaced.GetRmax2() - unplaced.GetDz() * unplaced.fTanRMax;
      distRMax = Abs(pRMax - widRMax) * unplaced.fInvSecRMax;

      if (! unplaced.IsFullPhi() )   // Protected against (0,0,z)
      {
          if (rho)
          {
            pPhi = ATan2(p.y(), p.x());

            if (pPhi  < unplaced.GetSPhi() - delta)
            {
              pPhi += kTwoPi;
            }
            else if (pPhi > unplaced.GetSPhi() + unplaced.GetDPhi() + delta)
            {
              pPhi -= kTwoPi;
            }

            distSPhi = Abs(pPhi - unplaced.GetSPhi());
            distEPhi = Abs(pPhi - unplaced.GetSPhi() - unplaced.GetDPhi());
          }
          else if (!(unplaced.GetRmin1()) || !(unplaced.GetRmin2()))
          {
            distSPhi = 0.;
            distEPhi = 0.;
          }
          nPs = Vec3D_t( unplaced.fSinSPhi, -unplaced.fCosSPhi, 0);
          nPe = Vec3D_t(-unplaced.fSinEPhi, unplaced.fCosEPhi, 0);
        }
        if (rho > delta)
        {
          nR = Vec3D_t(p.x() / rho / unplaced.fSecRMax, p.y() / rho / unplaced.fSecRMax,
                  -unplaced.fTanRMax * unplaced.fInvSecRMax);
          if (unplaced.GetRmin1() || unplaced.GetRmin2())
          {
            nr = Vec3D_t(-p.x() / rho / unplaced.fSecRMin, -p.y() / rho / unplaced.fSecRMin,
                    unplaced.fTanRMin * unplaced.fInvSecRMin);
          }
        }

        if (distRMax <= delta)
        {
          noSurfaces ++;
          sumnorm += nR;
        }
        if ((unplaced.GetRmin1() || unplaced.GetRmin2()) && (distRMin <= delta))
        {
          noSurfaces ++;
          sumnorm += nr;
        }
        if (!unplaced.IsFullPhi())
        {
          if (distSPhi <= dAngle)
          {
            noSurfaces ++;
            sumnorm += nPs;
          }
          if (distEPhi <= dAngle)
          {
            noSurfaces ++;
            sumnorm += nPe;
          }
        }
        if (distZ <= delta)
        {
          noSurfaces ++;
          if (p.z() >= 0.)
          {
            sumnorm += nZ;
          }
          else
          {
            sumnorm -= nZ;
          }
        }
        if (noSurfaces == 0)
        {
            Assert(false, "approximate surface normal not implemented");
            //          norm = ApproxSurfaceNormal(p);
        }
        else if (noSurfaces == 1)
        {
          norm = sumnorm;
        }
        else
        {
          norm = sumnorm.Unit();
        }

        normal = norm;

        valid = (bool) noSurfaces;
  }

}; // end struct

} } // End global namespace

#endif /* VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_ */
