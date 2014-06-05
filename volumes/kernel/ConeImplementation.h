/*
 * ConeImplementation.h
 *
 *  Created on: May 14, 2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_
#define VECGEOM_VOLUMES_KERNEL_CONEIMPLEMENTATION_H_

#include "base/global.h"
#include "base/transformation3d.h"
#include "volumes/UnplacedCone.h"
#include "volumes/specializations/ConeTraits.h"

namespace VECGEOM_NAMESPACE {

template <typename ConeType>
struct ConeHelper
{
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
}


template <TranslationCode transCodeT, RotationCode rotCodeT,
          typename ConeType>
struct ConeImplementation {

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void UnplacedInside(
      UnplacedCone const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      typename Backend::int_v &inside) {

    // TOBEIMPLEMENTED
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void Inside(UnplacedCone const &unplaced,
                     Transformation3D const &transformation,
                     Vector3D<typename Backend::precision_v> const &point,
                     Vector3D<typename Backend::precision_v> &localPoint,
                     typename Backend::int_v &inside) {
    localPoint = transformation.Transform<transCodeT, rotCodeT>(point);
    UnplacedInside<Backend>(unplaced, localPoint, inside);
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToIn(
      UnplacedCone const &unplaced,
      Transformation3D const &transformation,
      Vector3D<typename Backend::precision_v> const &point,
      Vector3D<typename Backend::precision_v> const &direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

   // TOBEIMPLEMENTED
    typedef typename Backend::bool_v MaskType;
    typedef typename Backend::precision_v VectorType;
    typedef typename Vector3D<typename Backend::precision_v> Vector3D;

    MaskType done_m(false); // which particles in the vector are ready to be returned == aka have been treated
    distance = kInfinity; // initialize distance to infinity

    // TODO: check that compiler is doing same thing as if we used a combined point + direction transformation
    Vector3D localpoint;
    localpoint=transformation.Transform<transCodeT,rotCodeT>(point);
    Vector3D localdir; // VectorType dirx, diry, dirz;
    localdir=transformation.TransformDirection<transCodeT,rotCodeT>(localdir);

    // do some inside checks
    // if safez is > 0 it means that particle is within z range
    // if safez is < 0 it means that particle is outside z range

    VectorType z = localpoint.z();
    VectorType safez = unplaced.GetDz() - Abs(z);
    MaskType inz_m = safez > Utils::fgToleranceVc;
    VectorType dirz = localdir.z();
    done_m = !inz_m && ( z*dirz >= 0 ); // particle outside the z-range and moving away

    VectorType x=localpoint.x();
    VectorType y=localpoint.y();
    VectorType r2 = x*x + y*y; // use of Perp2
    VectorType n2 = VectorType(1)-(unplaced.GetOuterSlopeSquare() + 1) *dirz*dirz; // dirx_v*dirx_v + diry_v*diry_v; ( dir is normalized !! )
    VectorType dirx = localdir.x();
    VectorType diry = localdir.y();
    VectorType rdotnplanar = x*dirx + y*diry; // use of PerpDot

    //   T a = 1 - dir.z*dir.z*(1+m*m);
    //   T b = 2 * ( pos.x*dir.x + pos.y*dir.y - m*m*pos.z*dir.z - m*n*dir.z);
    //   T c = ( pos.x*pos.x + pos.y*pos.y - m*m*pos.z*pos.z - 2*m*n*pos.z - n*n );

    // QUICK CHECK IF OUTER RADIUS CAN BE HIT AT ALL
    // BELOW WE WILL SOLVE A QUADRATIC EQUATION OF THE TYPE
    // a * t^2 + b * t + c = 0
    // if this equation has a solution at all ( == hit )
    // the following condition needs to be satisfied
    // DISCRIMINANT = b^2 -  4 a*c > 0
    //
    // THIS CONDITION DOES NOT NEED ANY EXPENSIVE OPERATION !!
    //
    // then the solutions will be given by
    //
        // t = (-b +- SQRT(DISCRIMINANT)) / (2a)
        //
        // b = 2*(dirx*x + diry*y)  -- independent of shape
        // a = dirx*dirx + diry*diry -- independent of shape
        // c = x*x + y*y - R^2 = r2 - R^2 -- dependent on shape
    VectorType c = r2 - unplaced.GetOuterSlopeSquare()*z*z - 2*unplaced.GetOuterSlope()*unplaced.GetOuterOffset() * z - unplaced.GetOuterOffsetSquare();

    VectorType a = n2;

    VectorType b = 2*(rdotnplanar - z*dirz*unplaced.GetOuterSlopeSquare() - unplaced.GetOuterSlope()*unplaced.GetOuterOffset()*dirz);
    VectorType discriminant = b*b-4*a*c;
    MaskType   canhitrmax = ( discriminant >= 0 );

    done_m |= ! canhitrmax;

    // this might be optional
    if( done_m.isFull() )
    {
           // joint z-away or no chance to hit Rmax condition
     #ifdef LOG_EARLYRETURNS
           std::cerr << " RETURN1 IN DISTANCETOIN " << std::endl;
     #endif
           return;
    }

    // Check outer cylinder (only r>rmax has to be considered)
    // this IS ALWAYS the MINUS (-) solution
    VectorType distanceRmax( Utils::kInfinityVc );
    distanceRmax( canhitrmax ) = (-b - Sqrt( discriminant ))/(2.*a);

    // this determines which vectors are done here already
    MaskType Rdone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmax );
    distanceRmax( ! Rdone ) = Utils::kInfinityVc;
    MaskType rmindone;
    // **** inner tube ***** only compiled in for tubes having inner hollow cone -- or in case of a universal runtime shape ******/
    if ( checkRminTreatment<ConeType>(unplaced) )
    {
       // in case of the Cone, generally all coefficients a, b, and c change
       a = 1.-(unplaced.GetInnerSlopeSquare() + 1) *dirz*dirz;
       c = r2 - unplaced.GetInnerSlopeSquare()*z*z - 2*unplaced.GetInnerSlope()*unplaced.GetInnerOffset() * z
               - unplaced.GetInnerOffsetSquare();
       b = 2*(rdotnplanar - dirz*(z*unplaced.GetInnerSlopeSquare + unplaced.GetInnerOffset* unplaced.GetInnerSlope()));
       discriminant =  b*b-4*a*c;
       MaskType canhitrmin = ( discriminant >= Vc::Zero );
       VectorType distanceRmin ( Utils::kInfinityVc );
       // this is always + solution
       distanceRmin ( canhitrmin ) = (-b + Vc::sqrt( discriminant ))/(2*a);
       rmindone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmin );
       distanceRmin ( ! rmindone ) = Utils::kInfinity;

       // reduction of distances
       distanceRmax = Vc::min( distanceRmax, distanceRmin );
       Rdone |= rmindone;
     }
        //distance( ! done_m && Rdone ) = distanceRmax;
        //done_m |= Rdone;

        /* might check early here */

        // now do Z-Face
        VectorType distancez = -safez/Vc::abs(dirz);
        MaskType zdone = determineZHit(x,y,z,dirx,diry,dirz,distancez);
        distance ( ! done_m && zdone ) = distancez;
        distance ( ! done_m && ! zdone && Rdone ) = distanceRmax;
        done_m |= ( zdone ) || (!zdone && (Rdone));

        // now PHI

        // **** PHI TREATMENT FOR CASE OF HAVING RMAX ONLY ***** only compiled in for cones having phi sektion ***** //
        if ( ConeTraits::NeedsPhiTreatment<ConeType>::value )
        {
           // all particles not done until here have the potential to hit a phi surface
           // phi surfaces require divisions so it might be useful to check before continuing

           if( ConeTraits::NeedsRminTreatment<ConeType>::value || ! done_m.isFull() )
           {
              VectorType distphi;
              ConeUtils::DistanceToPhiPlanes<ValueType,ConeTraits::IsPhiEqualsPiCase<ConeType>::value,ConeTraits::NeedsRminTreatment<ConeType>::value>(coneparams->dZ,
                    coneparams->outerslope, coneparams->outeroffset,
                    coneparams->innerslope, coneparams->inneroffset,
                    coneparams->normalPhi1.x, coneparams->normalPhi1.y, coneparams->normalPhi2.x, coneparams->normalPhi2.y,
                    coneparams->alongPhi1, coneparams->alongPhi2,
                    x, y, z, dirx, diry, dirz, distphi);
              if(ConeTraits::NeedsRminTreatment<ConeType>::value)
              {
                 // distance(! done_m || (rmindone && ! inrmin_m ) || (rmaxdone && ) ) = distphi;
                 // distance ( ! done_m ) = distphi;
                 distance = Vc::min(distance, distphi);
              }
              else
              {
                 distance ( ! done_m ) = distphi;
              }
           }
        }
  }

  template <class Backend>
  VECGEOM_CUDA_HEADER_BOTH
  static void DistanceToOut(
      UnplacedCone const &unplaced,
      Vector3D<typename Backend::precision_v> point,
      Vector3D<typename Backend::precision_v> direction,
      typename Backend::precision_v const &stepMax,
      typename Backend::precision_v &distance) {

   // TOBEIMPLEMENTED
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
