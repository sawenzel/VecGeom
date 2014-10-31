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

namespace VECGEOM_NAMESPACE {

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

template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
struct ConeImplementation {

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
  Float_t fDz = unplaced.GetDz();
  Bool_t inside = Abs( point.z() ) < fDz;

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
      Float_t fDz = unplaced.GetDz();
      // could cache a tolerant fDz;
      Float_t absz = Abs(point.z());
      Bool_t completelyinside = absz < fDz - kTolerance;
      // could cache a tolerant fDz;
      Bool_t completelyoutside = absz > fDz + kTolerance;

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
    typename Backend::bool_v contains;
    Vector3D<typename Backend::precision_v> localPoint
        = transformation.Transform<transCodeT, rotCodeT>(point);
    typename Backend::int_v crosscheck;
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

                cosPsi = (xi * unplaced.fCosCPhi + yi * unplaced.fSinCPhi) / Sqrt(rhoi2);

                if (cosPsi >= unplaced.fCosHDPhiIT)
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

        if (std::fabs(nt1) > VECGEOM_NAMESPACE::kRadTolerance) // Equation quadratic => 2 roots
        {
          b = nt2 / nt1;
          c = nt3 / nt1;
          d = b * b - c ;
          if ((nt3 > rout * rout * kRadTolerance * kRadTolerance * unplaced.fSecRMax * unplaced.fSecRMax)
              || (rout < 0))
          {
            // If outside real cone (should be rho-rout>kRadTolerance*0.5
            // NOT rho^2 etc) saves a std::sqrt() at expense of accuracy

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
              risec = std::sqrt(xi * xi + yi * yi) * unplaced.fSecRMax;
              norm = Vector3D<Precision>(xi / risec, yi / risec, -unplaced.fTanRMax / unplaced.fSecRMax);
              if (!unplaced.IsFullPhi())
              {
                cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi) / Sqrt(t3);
                if (cosPsi >= unplaced.fCosHDPhiIT)
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
                          norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
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
                        norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
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
                          norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
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
                        norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
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
                          norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
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
                        norm = Vector3D<Precision>(-xi / risec, -yi / risec, unplaced.fTanRMin / unplaced.fSecRMin);
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
                    cosPsi = (p.x() * unplaced.fCosCPhi + p.y() * unplaced.fSinCPhi) / Sqrt(t3);

                    if (cosPsi >= unplaced.fCosHDPhiIT)
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
                  // -> 2nd root or UUtils::kInfinity if 1st root on imaginary cone

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

                      if ((sd >= 0) && (std::fabs(zi) <= tolODz))    // sd>0
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

                  if ((sd >= 0) && (std::fabs(zi) <= tolODz))    // sd>0
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
        // o Tolerant of points inside phi planes by up to VUSolid::Tolerance()*0.5
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
  static void DistanceToOut(
      UnplacedCone const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

   // TOBEIMPLEMENTED

   // has to be implemented in a way
   // as to be tolerant when particle is outside
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToIn(UnplacedCone const &unplaced,
                         Transformation3D const &transformation,
                         Vector3D<typename Backend::precision_v> const &point,
                         typename Backend::precision_v &safety) {

    typedef typename Backend::precision_v Float_t;

    // TOBEIMPLEMENTED
  }

  template<class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void SafetyToOut(UnplacedCone const &unplaced,
                          Vector3D<typename Backend::precision_v> point,
                          typename Backend::precision_v &safety) {

  // TOBEIMPLEMENTED
  }
}; // end struct

} // end namespace
#endif /* VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_ */
