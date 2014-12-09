/*
 * ThetaCone.h
 *
 *  Created on: 09.10.2014
 *      Author: swenzel
 */

#ifndef VECGEOM_VOLUMES_THETACONE_H_
#define VECGEOM_VOLUMES_THETACONE_H_

#include "base/Global.h"
#include "volumes/kernel/GenericKernels.h"
#include "backend/Backend.h"
#include <iostream>

namespace VECGEOM_NAMESPACE
{

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
class ThetaCone{

    private:
        Precision fSTheta; // starting angle
        Precision fDTheta; // delta angle representing/defining the wedge
        Precision kAngTolerance;
        Precision halfAngTolerance;
        Precision fETheta;
        
        //Vector3D<Precision> fAlongVector1; // vector along the first plane
        //Vector3D<Precision> fAlongVector2; // vector aling the second plane

        //Vector3D<Precision> fNormalVector1; // normal vector for first plane
        // convention is that it points inwards

        //Vector3D<Precision> fNormalVector2; // normal vector for second plane
        // convention is that it points inwards

    public:
        VECGEOM_CUDA_HEADER_BOTH
        ThetaCone( Precision sTheta, Precision dTheta ) :
            fSTheta(sTheta), fDTheta(dTheta), kAngTolerance(kSTolerance) {
            // check input
            //Assert( angle > 0., " wedge angle has to be larger than zero " );

            // initialize angles
            /*
            fAlongVector1.x() = std::cos(zeroangle);
            fAlongVector1.y() = std::sin(zeroangle);
            fAlongVector2.x() = std::cos(zeroangle+angle);
            fAlongVector2.y() = std::sin(zeroangle+angle);

            fNormalVector1.x() = -std::sin(zeroangle);
            fNormalVector1.y() = std::cos(zeroangle);  // not the + sign
            fNormalVector2.x() =  std::sin(zeroangle+angle);
            fNormalVector2.y() = -std::cos(zeroangle+angle); // note the - sign
             */ 
                fETheta = fSTheta + fDTheta;
                halfAngTolerance = (0.5 * kAngTolerance);
        }

        VECGEOM_CUDA_HEADER_BOTH
        ~ThetaCone(){}

        //Vector3D<Precision> GetAlong1() const {return fAlongVector1; }
        //Vector3D<Precision> GetAlong2() const {return fAlongVector2; }

        // very important:
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::bool_v Contains( Vector3D<typename Backend::precision_v> const& point ) const{
        
            typedef typename Backend::bool_v Bool_t;
            Bool_t unused;
            Bool_t outside;
            GenericKernelForContainsAndInside<Backend, false>(
                point, unused, outside);
            return !outside;
        
        }

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::bool_v ContainsWithBoundary( Vector3D<typename Backend::precision_v> const& point ) const{}

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::inside_v Inside( Vector3D<typename Backend::precision_v> const& point ) const{
        
            typedef typename Backend::bool_v      Bool_t;
            Bool_t completelyinside, completelyoutside;
            GenericKernelForContainsAndInside<Backend,true>(
                 point, completelyinside, completelyoutside);
            typename Backend::inside_v  inside=EInside::kSurface;
            MaskedAssign(completelyoutside, EInside::kOutside, &inside);
            MaskedAssign(completelyinside, EInside::kInside, &inside);
        }

        /**
         * estimate of the smallest distance to the ThetaCone boundary when
         * the point is located outside the Wegde
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::precision_v SafetyToIn( Vector3D<typename Backend::precision_v> const& point ) const {
        
            
            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;
            
            Float_t KPI(kPi);
            Float_t piby2(kPi/2);
    
            Float_t rad = point.Mag();
            Float_t dTheta1,dTheta2;
            Float_t pTheta,safeTheta1, safeTheta2;
            
        MaskedAssign((rad != 0.),(piby2 - asin(point.z() / rad)),&pTheta);
        MaskedAssign(((rad != 0.) && (pTheta < 0.)),(pTheta+KPI),&pTheta);
        MaskedAssign(((rad != 0.) ),(fSTheta - pTheta),&dTheta1);
        MaskedAssign(((rad != 0.) ),(pTheta - fETheta),&dTheta2);
        MaskedAssign(((rad != 0.) && (dTheta1 > dTheta2) && (dTheta1 >= 0.)),(rad * sin(dTheta1)),&safeTheta1);
        //MaskedAssign(((rad != zero) && (dTheta1 > dTheta2) && (dTheta1 >= zero) && (safety <= safeTheta) ),safeTheta,&safety);
        MaskedAssign(((rad != 0.) && (dTheta2 > dTheta1) && (dTheta2 >= 0.)),(rad * sin(dTheta2)),&safeTheta2);
        //MaskedAssign(((rad != zero) && !(dTheta1 > dTheta2) && (dTheta2 >= zero) && (safety <= safeTheta)),safeTheta,&safety);
        return Max(safeTheta1 , safeTheta2);
        }

        /**
         * estimate of the smallest distance to the ThetaCone boundary when
         * the point is located inside the ThetaCone ( within the defining phi angle )
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::precision_v SafetyToOut( Vector3D<typename Backend::precision_v> const& point ) const{
        
            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;
            
           // Float_t KPI(kPi);
           // Float_t piby2(kPi/2);
            
        Float_t pTheta(0.);
        Float_t dTheta1(0.);
        Float_t dTheta2(0.);
        Float_t safeTheta(0.);
        
    
        Float_t rad = point.Mag();
        MaskedAssign((rad != 0.),(kPi/2 - asin(point.z() / rad)),&pTheta);
        MaskedAssign( ((rad != 0.) && (pTheta < 0.) ),(pTheta+kPi),&pTheta);
        MaskedAssign( ((rad != 0.)),(pTheta - fSTheta),&dTheta1);
        MaskedAssign( ((rad != 0.)),(fETheta - pTheta),&dTheta2);
        CondAssign((dTheta1 < dTheta2),(rad * sin(dTheta1)),(rad * sin(dTheta2)),&safeTheta);
        return safeTheta;
        }

        /**
         * estimate of the distance to the ThetaCone boundary with given direction
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	void DistanceToIn(Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distThetaCone1,typename  Backend::precision_v &distThetaCone2 ) const{}
          template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	void DistanceToOut(Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distThetaCone1,typename  Backend::precision_v &distThetaCone2 ) const{}


        // this could be useful to be public such that other shapes can directly
        // use completelyinside + completelyoutside

        template<typename Backend, bool ForInside>
        VECGEOM_CUDA_HEADER_BOTH
        void GenericKernelForContainsAndInside(
                Vector3D<typename Backend::precision_v> const &localPoint,
                typename Backend::bool_v &completelyinside,
                typename Backend::bool_v &completelyoutside) const {
        
            typedef typename Backend::precision_v Float_t;
            Float_t pTheta = ATan2(Sqrt(localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y()), localPoint.z()); 
            Precision tolAngMin = fSTheta + halfAngTolerance;
            Precision tolAngMax = fETheta - halfAngTolerance;
            completelyinside = (pTheta <= tolAngMax) && (pTheta >= tolAngMin);
            Precision tolAngMin2 = fSTheta - halfAngTolerance;
            Precision tolAngMax2 = fETheta + halfAngTolerance;
            completelyoutside = (pTheta < tolAngMin2) || (pTheta > tolAngMax2);
            //if( IsFull(completelyoutside) )return;
        
    }
    
    

}; // end of class ThetaCone

/*
    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::inside_v ThetaCone::Inside( Vector3D<typename Backend::precision_v> const& point ) const
    {
        typedef typename Backend::bool_v      Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(
              point, completelyinside, completelyoutside);
        typename Backend::inside_v  inside=EInside::kSurface;
        MaskedAssign(completelyoutside, EInside::kOutside, &inside);
        MaskedAssign(completelyinside, EInside::kInside, &inside);
    }

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v ThetaCone::ContainsWithBoundary( Vector3D<typename Backend::precision_v> const& point ) const
    {
        typedef typename Backend::bool_v      Bool_t;
        Bool_t completelyinside, completelyoutside;
        GenericKernelForContainsAndInside<Backend,true>(
              point, completelyinside, completelyoutside);
	return !completelyoutside;
    }

    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::bool_v ThetaCone::Contains( Vector3D<typename Backend::precision_v> const& point ) const
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
    void ThetaCone::GenericKernelForContainsAndInside(
                Vector3D<typename Backend::precision_v> const &localPoint,
                typename Backend::bool_v &completelyinside,
                typename Backend::bool_v &completelyoutside) const
    {
        typedef typename Backend::precision_v Float_t;

       // this part of the code assumes some symmetry knowledge and is currently only
        // correct for a PhiThetaCone assumed to be aligned along the z-axis.
        Float_t x = localPoint.x();
        Float_t y = localPoint.y();
        Float_t startx = fAlongVector1.x( );
        Float_t starty = fAlongVector1.y( );
        Float_t endx = fAlongVector2.x( );
        Float_t endy = fAlongVector2.y( );

        Float_t startCheck = (-x*starty + y*startx);
        Float_t endCheck   = (-endx*y   + endy*x);

        // TODO: I think we need to treat the tolerance as a phi - tolerance
        // this will complicate things a little bit
        completelyoutside = startCheck < MakeMinusTolerant<ForInside>(0.);
        if(ForInside)
            completelyinside = startCheck > MakePlusTolerant<ForInside>(0.);

        if(fDPhi<kPi) {
            completelyoutside |= endCheck < MakeMinusTolerant<ForInside>(0.);
            if(ForInside)
                completelyinside &= endCheck > MakePlusTolerant<ForInside>(0.);
        }
        else {
            completelyoutside &= endCheck < MakeMinusTolerant<ForInside>(0.);
            if(ForInside)
               completelyinside |= endCheck > MakePlusTolerant<ForInside>(0.);
        }
    }


    template<typename Backend>
    VECGEOM_CUDA_HEADER_BOTH
    typename Backend::precision_v ThetaCone::SafetyToOut(
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
    typename Backend::precision_v ThetaCone::SafetyToIn(
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
   void ThetaCone::DistanceToIn(
           Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distThetaCone1,
           typename  Backend::precision_v &distThetaCone2 ) const {
      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;
      // algorithm::first calculate projections of direction to both planes,
      // then calculate real distance along given direction,
      // distance can be negative
      
      distThetaCone1 = kInfinity;
      distThetaCone2 = kInfinity;

      Float_t comp1 = dir.x()*fNormalVector1.x() + dir.y()*fNormalVector1.y();
      Float_t comp2 = dir.x()*fNormalVector2.x() + dir.y()*fNormalVector2.y();
       
      Bool_t cmp1 = comp1 > 0.;
      if( ! IsEmpty(cmp1))
      {
        Float_t tmp = -(point.x()*fNormalVector1.x()+point.y()*fNormalVector1.y())/comp1;
        MaskedAssign(cmp1 && tmp >0., tmp, &distThetaCone1);
      }
       Bool_t cmp2 = comp2 > 0.;
       if( ! IsEmpty(cmp2) )
      {
        Float_t tmp  =  -(point.x()*fNormalVector2.x()+point.y()*fNormalVector2.y())/comp2;
        MaskedAssign(cmp2&& tmp >0., tmp, &distThetaCone2);
      }
     
       //std::cerr << "c1 " << comp1 <<" d1="<<distThetaCone1<<" p="<<point<< "\n";
       //std::cerr << "c2 " << comp2 <<" d2="<<distThetaCone2<< "\n";
   }

    template <class Backend>
   VECGEOM_CUDA_HEADER_BOTH
   void ThetaCone::DistanceToOut(
           Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,
           typename  Backend::precision_v &distThetaCone1,
           typename  Backend::precision_v &distThetaCone2) const {

      typedef typename Backend::precision_v Float_t;
      typedef typename Backend::bool_v Bool_t;

      // algorithm::first calculate projections of direction to both planes,
      // then calculate real distance along given direction,
      // distance can be negative

      Float_t comp1 = dir.x()*fNormalVector1.x() + dir.y()*fNormalVector1.y();
      Float_t comp2 = dir.x()*fNormalVector2.x() + dir.y()*fNormalVector2.y();

      //std::cerr << "c1 " << comp1 << "\n";
      //std::cerr << "c2 " << comp2 << "\n";
      distThetaCone1 = kInfinity;
      distThetaCone2 = kInfinity;

      Bool_t cmp1 = comp1 < 0.;
      if( ! IsEmpty(cmp1) )
      {
        Float_t tmp =  -(point.x()*fNormalVector1.x()+point.y()*fNormalVector1.y())/comp1;
        MaskedAssign( cmp1 && tmp>0., tmp, &distThetaCone1 );
      }

      Bool_t cmp2 = comp2 < 0.;
      if( ! IsEmpty(cmp2) )
      {
        Float_t tmp = -(point.x()*fNormalVector2.x()+point.y()*fNormalVector2.y())/comp2;
        MaskedAssign( cmp2 && tmp>0., tmp, &distThetaCone2 );
      }
     
      //std::cerr << "c1 " << comp1 <<" d1="<<distThetaCone1<<" "<<point<< "\n";
      //std::cerr << "c2 " << comp2 <<" d2=" <<distThetaCone2<<" "<<point<<"\n";
   }
 */ 
          

} // end of namespace


#endif /* VECGEOM_VOLUMES_THETACONE_H_ */
