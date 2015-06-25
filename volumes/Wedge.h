/*
 * Wedge.h
 *
 *  Created on: 09.10.2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_WEDGE_H_
#define VECGEOM_VOLUMES_WEDGE_H_

#include "base/Global.h"
#include "volumes/kernel/GenericKernels.h"
#include "backend/Backend.h"

namespace vecgeom {
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
class Wedge{

    private:
        // Precision fSPhi; // starting angle
        Precision fDPhi; // delta angle representing/defining the wedge
        Vector3D<Precision> fAlongVector1; // vector along the first plane
        Vector3D<Precision> fAlongVector2; // vector aling the second plane

        Vector3D<Precision> fNormalVector1; // normal vector for first plane
        // convention is that it points inwards

        Vector3D<Precision> fNormalVector2; // normal vector for second plane
        // convention is that it points inwards

    public:
        VECGEOM_CUDA_HEADER_BOTH
        Wedge( Precision angle, Precision zeroangle = 0 );

        VECGEOM_CUDA_HEADER_BOTH
        ~Wedge(){}

        VECGEOM_CUDA_HEADER_BOTH
        Vector3D<Precision> GetAlong1() const {return fAlongVector1; }

        VECGEOM_CUDA_HEADER_BOTH
        Vector3D<Precision> GetAlong2() const {return fAlongVector2; }

        VECGEOM_CUDA_HEADER_BOTH
        Vector3D<Precision> GetNormal1() const {return fNormalVector1; }

        VECGEOM_CUDA_HEADER_BOTH
        Vector3D<Precision> GetNormal2() const {return fNormalVector2; }

        // very important:
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::bool_v Contains( Vector3D<typename Backend::precision_v> const& point ) const;

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::bool_v ContainsWithBoundary( Vector3D<typename Backend::precision_v> const& point ) const;

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::inside_v Inside( Vector3D<typename Backend::precision_v> const& point ) const;

        // static function determining if input points are on a plane surface which is part of a wedge
        // ( given by along and normal )
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        static typename Backend::bool_v IsOnSurfaceGeneric( Vector3D<Precision> const & alongVector,
                                                            Vector3D<Precision> const & normalVector,
                                                            Vector3D<typename Backend::precision_v> const& point );

        VECGEOM_CUDA_HEADER_BOTH
        bool IsOnSurface1( Vector3D<Precision> const& point ) const {
            return Wedge::IsOnSurfaceGeneric<kScalar>( fAlongVector1, fNormalVector1, point );
        }


        VECGEOM_CUDA_HEADER_BOTH
        bool IsOnSurface2( Vector3D<Precision> const& point ) const {
            return Wedge::IsOnSurfaceGeneric<kScalar>( fAlongVector2, fNormalVector2, point );
        }

        /**
         * estimate of the smallest distance to the Wedge boundary when
         * the point is located outside the Wedge
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::precision_v SafetyToIn( Vector3D<typename Backend::precision_v> const& point ) const;

        /**
         * estimate of the smallest distance to the Wedge boundary when
         * the point is located inside the Wedge ( within the defining phi angle )
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::precision_v SafetyToOut( Vector3D<typename Backend::precision_v> const& point ) const;

        /**
         * estimate of the distance to the Wedge boundary with given direction
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	void DistanceToIn(Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distWedge1,typename  Backend::precision_v &distWedge2 ) const;
          template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	void DistanceToOut(Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distWedge1,typename  Backend::precision_v &distWedge2 ) const;


        // this could be useful to be public such that other shapes can directly
        // use completelyinside + completelyoutside

        template<typename Backend, bool ForInside>
        VECGEOM_CUDA_HEADER_BOTH
        void GenericKernelForContainsAndInside(
                Vector3D<typename Backend::precision_v> const &localPoint,
                typename Backend::bool_v &completelyinside,
                typename Backend::bool_v &completelyoutside) const;

}; // end of class Wedge

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::inside_v Wedge::Inside( Vector3D<typename Backend::precision_v> const& point ) const
    {

        typedef typename Backend::bool_v      Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(
              point, completelyinside, completelyoutside);
        typename Backend::inside_v  inside=EInside::kSurface;
        MaskedAssign(completelyoutside, EInside::kOutside, &inside);
        MaskedAssign(completelyinside, EInside::kInside, &inside);
        return inside;
    }

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v Wedge::ContainsWithBoundary( Vector3D<typename Backend::precision_v> const& point ) const
    {
        typedef typename Backend::bool_v      Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(
              point, completelyinside, completelyoutside);
        return !completelyoutside;
    }

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v Wedge::Contains( Vector3D<typename Backend::precision_v> const& point ) const
    {
        typedef typename Backend::bool_v Bool_t;
        Bool_t unused;
        Bool_t outside;
        GenericKernelForContainsAndInside<Backend, false>(
           point, unused, outside);
        return !outside;
    }

    // Implementation follows
    template<typename Backend, bool ForInside>
    VECGEOM_CUDA_HEADER_BOTH
    void Wedge::GenericKernelForContainsAndInside(
                Vector3D<typename Backend::precision_v> const &localPoint,
                typename Backend::bool_v &completelyinside,
                typename Backend::bool_v &completelyoutside) const
    {
        typedef typename Backend::precision_v Real_v;

       // this part of the code assumes some symmetry knowledge and is currently only
        // correct for a PhiWedge assumed to be aligned along the z-axis.
        Real_v x = localPoint.x();
        Real_v y = localPoint.y();
        Real_v startx = fAlongVector1.x( );
        Real_v starty = fAlongVector1.y( );
        Real_v endx = fAlongVector2.x( );
        Real_v endy = fAlongVector2.y( );

        Real_v startCheck = (-x*starty + y*startx);
        Real_v endCheck   = (-endx*y   + endy*x);

        completelyoutside = startCheck < 0.;
        if(fDPhi<kPi)
            completelyoutside |= endCheck < 0.;
        else
            completelyoutside &= endCheck < 0.;
        if( ForInside ){
            // TODO: see if the compiler optimizes across these function calls sinc
            // a couple of multiplications inside IsOnSurfaceGeneric are already done preveously
            typename Backend::bool_v onSurface =
                    Wedge::IsOnSurfaceGeneric<Backend>(fAlongVector1, fNormalVector1, localPoint)
                    || Wedge::IsOnSurfaceGeneric<Backend>(fAlongVector2,fNormalVector2, localPoint);
            completelyoutside &= !onSurface;
            completelyinside = !onSurface && !completelyoutside;
        }
    }

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v Wedge::IsOnSurfaceGeneric( Vector3D<Precision> const & alongVector,
                                    Vector3D<Precision> const & normalVector,
                                    Vector3D<typename Backend::precision_v> const& point ) {
        // on right side of half plane ??
        typedef typename Backend::bool_v Bool_v;
        Bool_v condition1 = alongVector.x() * point.x() + alongVector.y()*point.y() >= 0.;
        if( IsEmpty(condition1) )
            return Bool_v(false);
        // within the right distance to the plane ??
        Bool_v condition2 = Abs( normalVector.x()*point.x() + normalVector.y()*point.y() ) < kTolerance;
        return condition1 && condition2;
    }



    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::precision_v Wedge::SafetyToOut(
            Vector3D<typename Backend::precision_v> const& point ) const{
        typedef typename Backend::precision_v Float_t;
        // algorithm: calculate projections to both planes
        // return minimum / maximum depending on fAngle < PI or not

        // assuming that we have z wedge and the planes pass through the origin
        Float_t dist1 = point.x()*fNormalVector1.x() + point.y()*fNormalVector1.y();
        Float_t dist2 = point.x()*fNormalVector2.x() + point.y()*fNormalVector2.y();

        // std::cerr << "d1 " << dist1<<"  "<<point << "\n";
	// std::cerr << "d2 " << dist2<<"  "<<point << "\n";

        if(fDPhi < kPi){
            return Min(dist1,dist2);
        }
        else{
	  //Float_t disttocorner = Sqrt(point.x()*point.x() + point.y()*point.y());
	  // return Max(dist1,Max(dist2,disttocorner));
           return Max(dist1,dist2);
        }
    }



    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::precision_v Wedge::SafetyToIn(
            Vector3D<typename Backend::precision_v> const& point ) const {
        typedef typename Backend::precision_v Float_t;
        // algorithm: calculate projections to both planes
        // return maximum / minimum depending on fAngle < PI or not
        // assuming that we have z wedge and the planes pass through the origin

        // actually we

        Float_t dist1 = point.x()*fNormalVector1.x() + point.y()*fNormalVector1.y();
        Float_t dist2 = point.x()*fNormalVector2.x() + point.y()*fNormalVector2.y();

       
        // std::cerr << "d1 " << dist1<<"  "<<point << "\n";
        // std::cerr << "d2 " << dist2<<"  "<<point << "\n";

        if(fDPhi < kPi){
           // Float_t disttocorner = Sqrt(point.x()*point.x() + point.y()*point.y());
           // commented out DistanceToCorner in order to not have a differences with Geant4 and Root
            return Max(-1*dist1,-1*dist2);
        }
        else{
            return Min(-1*dist1,-1*dist2);
        }
   }

   template <class Backend>
   VECGEOM_CUDA_HEADER_BOTH
   void Wedge::DistanceToIn(
           Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distWedge1,
           typename  Backend::precision_v &distWedge2 ) const {
      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;
      // algorithm::first calculate projections of direction to both planes,
      // then calculate real distance along given direction,
      // distance can be negative
      
      distWedge1 = kInfinity;
      distWedge2 = kInfinity;

      Float_t comp1 = dir.x()*fNormalVector1.x() + dir.y()*fNormalVector1.y();
      Float_t comp2 = dir.x()*fNormalVector2.x() + dir.y()*fNormalVector2.y();
       
      Bool_t cmp1 = comp1 > 0.;
      if( ! IsEmpty(cmp1))
      {
        Float_t tmp = -(point.x()*fNormalVector1.x()+point.y()*fNormalVector1.y())/comp1;
        MaskedAssign(cmp1 && tmp >0., tmp, &distWedge1);
      }
       Bool_t cmp2 = comp2 > 0.;
       if( ! IsEmpty(cmp2) )
      {
        Float_t tmp  =  -(point.x()*fNormalVector2.x()+point.y()*fNormalVector2.y())/comp2;
        MaskedAssign(cmp2&& tmp >0., tmp, &distWedge2);
      }
     
       //std::cerr << "c1 " << comp1 <<" d1="<<distWedge1<<" p="<<point<< "\n";
       //std::cerr << "c2 " << comp2 <<" d2="<<distWedge2<< "\n";
   }

   template <class Backend>
   VECGEOM_CUDA_HEADER_BOTH
   void Wedge::DistanceToOut(
           Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,
           typename  Backend::precision_v &distWedge1,
           typename  Backend::precision_v &distWedge2) const {

      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;

      // algorithm::first calculate projections of direction to both planes,
      // then calculate real distance along given direction,
      // distance can be negative

      Float_t comp1 = dir.x()*fNormalVector1.x() + dir.y()*fNormalVector1.y();
      Float_t comp2 = dir.x()*fNormalVector2.x() + dir.y()*fNormalVector2.y();

      //std::cerr << "c1 " << comp1 << "\n";
      //std::cerr << "c2 " << comp2 << "\n";
      distWedge1 = kInfinity;
      distWedge2 = kInfinity;

      Bool_t cmp1 = comp1 < 0.;
      if( ! IsEmpty(cmp1) )
      {
        Float_t tmp =  -(point.x()*fNormalVector1.x()+point.y()*fNormalVector1.y())/comp1;
        MaskedAssign( cmp1 && tmp>0., tmp, &distWedge1 );
      }

      Bool_t cmp2 = comp2 < 0.;
      if( ! IsEmpty(cmp2) )
      {
        Float_t tmp = -(point.x()*fNormalVector2.x()+point.y()*fNormalVector2.y())/comp2;
        MaskedAssign( cmp2 && tmp>0., tmp, &distWedge2 );
      }
     
      //std::cerr << "c1 " << comp1 <<" d1="<<distWedge1<<" "<<point<< "\n";
      //std::cerr << "c2 " << comp2 <<" d2=" <<distWedge2<<" "<<point<<"\n";
   }
          

} } // end of namespace


#endif /* VECGEOM_VOLUMES_WEDGE_H_ */
