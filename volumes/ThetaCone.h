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
        Precision tanSTheta2;
        Precision tanETheta2;
        
        //Vector3D<Precision> fAlongVector1; // vector along the first plane
        //Vector3D<Precision> fAlongVector2; // vector aling the second plane

        Vector3D<Precision> fNormalVector1; // normal vector for first Cone
        // convention is that it points inwards

        Vector3D<Precision> fNormalVector2; // normal vector for second Cone
        // convention is that it points inwards

    public:
        VECGEOM_CUDA_HEADER_BOTH
        ThetaCone( Precision sTheta, Precision dTheta ) :
            fSTheta(sTheta), fDTheta(dTheta), kAngTolerance(kSTolerance) {
            
                fETheta = fSTheta + fDTheta;
                halfAngTolerance = (0.5 * kAngTolerance*10.);
                Precision tanSTheta = tan(fSTheta);
                tanSTheta2 = tanSTheta * tanSTheta;
                
                Precision tanETheta = tan(fETheta);
                tanETheta2 = tanETheta * tanETheta;
                
              
        }

        VECGEOM_CUDA_HEADER_BOTH
        ~ThetaCone(){}

       
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
        //typename Backend::precision_v SafetyToIn( Vector3D<typename Backend::precision_v> const& point ) const {
        void SafetyToIn( Vector3D<typename Backend::precision_v> const& point, typename Backend::precision_v &safety  ) const {
        
            
            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;
            
            //Float_t KPI(kPi);
            //Float_t piby2(kPi/2);
    
            Float_t rad = point.Mag();
            Float_t dTheta1,dTheta2;
            Float_t pTheta;//,safeTheta1, safeTheta2;
            Float_t safeTheta;
            //Float_t safety(0.);
            
            //ATan2(Sqrt(localPoint.x()*localPoint.x() + localPoint.y()*localPoint.y()), localPoint.z());
        //MaskedAssign((rad != 0.),(kPi/2 - asin(point.z() / rad)),&pTheta);
        MaskedAssign((rad != 0.),ATan2(Sqrt(point.x()*point.x() + point.y()*point.y()), point.z()),&pTheta);
        MaskedAssign(((rad != 0.) && (pTheta < 0.)),(pTheta+kPi),&pTheta);
        MaskedAssign(((rad != 0.) ),(fSTheta - pTheta),&dTheta1);
        MaskedAssign(((rad != 0.) ),(pTheta - fETheta),&dTheta2);
        MaskedAssign(((rad != 0.) && (dTheta1 > dTheta2) && (dTheta1 >= 0.)),(rad * sin(dTheta1)),&safeTheta);
        MaskedAssign(((rad != 0.) && (dTheta1 > dTheta2) && (dTheta1 >= 0.) && (safety <= safeTheta) ),safeTheta,&safety);
        MaskedAssign(((rad != 0.) && (dTheta2 > dTheta1) && (dTheta2 >= 0.)),(rad * sin(dTheta2)),&safeTheta);
        MaskedAssign(((rad != 0.) && !(dTheta1 > dTheta2) && (dTheta2 >= 0.) && (safety <= safeTheta)),safeTheta,&safety);
        //return Max(safeTheta1 , safeTheta2);
        //return safeTheta;
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
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distThetaCone1,typename  Backend::precision_v &distThetaCone2,
				    typename Backend::bool_v &intsect1, typename Backend::bool_v &intsect2/*, Vector3D<typename Backend::precision_v> &cone1IntSecPt,
                                    Vector3D<typename Backend::precision_v> &cone2IntSecPt*/) const{
	     
	    bool verbose=false;
            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;
            
            Float_t a,b,c,d2;
            Float_t a2,b2,c2,d22;
            
            Float_t firstRoot(kInfinity), secondRoot(kInfinity);
            
            Float_t pDotV2d = point.x() * dir.x() + point.y() * dir.y();
            Float_t rho2 = point.x() * point.x() + point.y() * point.y();
            
            b = pDotV2d - point.z() * dir.z() * tanSTheta2 ; // tan(fSTheta) * tan(fSTheta);
            a = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;  //tan(fSTheta) * tan(fSTheta);
            c = rho2 - point.z() * point.z() * tanSTheta2; //tan(fSTheta) * tan(fSTheta);
            d2 = b * b - a * c;
            
            MaskedAssign((d2 > 0.), (-1*b + Sqrt(d2))/a, &firstRoot);
            MaskedAssign(firstRoot < 0. ,kInfinity, &firstRoot);
            
            b2 = point.x() * dir.x() + point.y() * dir.y() - point.z() * dir.z() * tanETheta2; // tan(fETheta) * tan(fETheta);
            a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2; //tan(fETheta) * tan(fETheta);
            c2 = point.x() * point.x() + point.y() * point.y() - point.z() * point.z() * tanETheta2; //tan(fETheta) * tan(fETheta);
            d22 = b2 * b2 - a2 * c2;
            
            MaskedAssign((d22 > 0.), (-1*b2 - Sqrt(d22))/a2, &secondRoot);
            MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
            
              if(fSTheta < kPi/2)
              {
                  if(fETheta < kPi/2)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          distThetaCone2 = secondRoot;
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          
                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 > 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 > 0.));
                          
                      }
                  }
                  
                  if(fETheta > kPi/2)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          MaskedAssign((d22 > 0.), (-1*b2 + Sqrt(d22))/a2, &secondRoot);
                          MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;
                          
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          
                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 > 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));
                            
                      }
                  }
              }
              
              if(fSTheta > kPi/2)
              {
                  if(fETheta > kPi/2)
                  {
                      if(fSTheta < fETheta)
                      {
                          MaskedAssign((d2 > 0.), (-1*b - Sqrt(d2))/a, &firstRoot);
                          MaskedAssign(firstRoot < 0. ,kInfinity, &firstRoot);
                          distThetaCone1 = firstRoot;
                          MaskedAssign((d22 > 0.), (-1*b2 + Sqrt(d22))/a2, &secondRoot);
                          MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;
                          
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          
                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 < 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));
                          
                      }
                  }
              }
          
            //if(verbose) std::cout<<"DISTANCES : distThetaCone1 : "<<distThetaCone1<<"  :: distThetaCone2 : "<<distThetaCone2<<std::endl;
	  }
          template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	void DistanceToOut(Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distThetaCone1,typename  Backend::precision_v &distThetaCone2 ,
                            typename Backend::bool_v &intsect1, typename Backend::bool_v &intsect2) const{
              
              bool verbose=false;
            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;
            
            Float_t a,b,c,d2;
            Float_t a2,b2,c2,d22;
            
            Float_t firstRoot(kInfinity), secondRoot(kInfinity);
            
            Float_t pDotV2d = point.x() * dir.x() + point.y() * dir.y();
            Float_t rho2 = point.x() * point.x() + point.y() * point.y();
            
            b = pDotV2d - point.z() * dir.z() * tanSTheta2 ; // tan(fSTheta) * tan(fSTheta);
            a = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;  //tan(fSTheta) * tan(fSTheta);
            c = rho2 - point.z() * point.z() * tanSTheta2; //tan(fSTheta) * tan(fSTheta);
            d2 = b * b - a * c;
            
            MaskedAssign((d2 > 0.), (-1*b - Sqrt(d2))/a, &firstRoot);
            MaskedAssign(firstRoot < 0. ,kInfinity, &firstRoot);
            
            b2 = point.x() * dir.x() + point.y() * dir.y() - point.z() * dir.z() * tanETheta2; // tan(fETheta) * tan(fETheta);
            a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2; //tan(fETheta) * tan(fETheta);
            c2 = point.x() * point.x() + point.y() * point.y() - point.z() * point.z() * tanETheta2; //tan(fETheta) * tan(fETheta);
            d22 = b2 * b2 - a2 * c2;
            
            MaskedAssign((d22 > 0.), (-1*b2 + Sqrt(d22))/a2, &secondRoot);
            MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
            
              if(fSTheta < kPi/2)
              {
                  if(fETheta < kPi/2)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          distThetaCone2 = secondRoot;
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          
                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 > 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 > 0.));
                      }
                  }
                  
                  if(fETheta > kPi/2)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          MaskedAssign((d22 > 0.), (-1*b2 - Sqrt(d22))/a2, &secondRoot);
                          MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;
                          
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          
                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 > 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));
                          
                      }
                  }
              }
              
              if(fSTheta > kPi/2)
              {
                  if(fETheta > kPi/2)
                  {
                      if(fSTheta < fETheta)
                      {
                          MaskedAssign((d2 > 0.), (-1*b + Sqrt(d2))/a, &firstRoot);
                          MaskedAssign(firstRoot < 0. ,kInfinity, &firstRoot);
                          distThetaCone1 = firstRoot;
                          MaskedAssign((d22 > 0.), (-1*b2 - Sqrt(d22))/a2, &secondRoot);
                          MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;
                          
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          
                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 < 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));
                      }
                  }
              }
          
          }


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


} // end of namespace


#endif /* VECGEOM_VOLUMES_THETACONE_H_ */
