/*
 * ThetaCone.h
 *
 *      Author: Raman Sehgal
 */

#ifndef VECGEOM_VOLUMES_THETACONE_H_
#define VECGEOM_VOLUMES_THETACONE_H_

#include "base/Global.h"
#include "volumes/kernel/GenericKernels.h"
#include "backend/Backend.h"
#include <iostream>
#include <iomanip>
namespace vecgeom {
inline namespace VECGEOM_IMPL_NAMESPACE {

/**
 * A class representing a ThetaCone (basically a double cone) which is represented by an angle theta ( 0 < theta < Pi). It
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
 *DistanceToIn and DistanceToOut provides distances with the First and Second ThetaCone in "distThetaCone1" and "distThetaCone2" reference variables.
 *Reference bool variable "intsect1" and "intsect2" is used to detect the real intersection cone, i.e. whether the point really intersects with a ThetaCone or not.
 */
class ThetaCone{

    private:

        Precision fSTheta; // starting angle
        Precision fDTheta; // delta angle representing/defining the wedge
        Precision kAngTolerance;
        Precision halfAngTolerance;
        Precision fETheta; // ending angle
	Precision tanSTheta;
	Precision tanETheta;
	Precision tanBisector;
	Precision slope1,slope2;
        Precision tanSTheta2;
        Precision tanETheta2;

	//Precision cone1Radius,cone2Radius;


    public:
        VECGEOM_CUDA_HEADER_BOTH
        ThetaCone( Precision sTheta, Precision dTheta) :
            fSTheta(sTheta), fDTheta(dTheta), kAngTolerance(kTolerance) {
               // std::cout<<"Entered Constructor of ThetaCone\n";
            // check input
            //Assert( angle > 0., " wedge angle has to be larger than zero " );

            // initialize angles
            fETheta = fSTheta + fDTheta;
            halfAngTolerance = (0.5 * kAngTolerance);
	    Precision tempfSTheta=fSTheta;
	    Precision tempfETheta=fETheta;

	    if(fSTheta > kPi/2)
		tempfSTheta = kPi - fSTheta;
	    if(fETheta > kPi/2)
		tempfETheta = kPi - fETheta;


            tanSTheta = tan(tempfSTheta);
            tanSTheta2 = tanSTheta * tanSTheta;
            tanETheta = tan(tempfETheta);
            tanETheta2 = tanETheta * tanETheta;
	    tanBisector = tan(tempfSTheta+(fDTheta/2));
	    if(fSTheta > kPi/2 && fETheta > kPi/2)
		tanBisector = tan(tempfSTheta-(fDTheta/2));
	    slope1=tan(kPi/2-fSTheta);
	    slope2=tan(kPi/2-fETheta);

        }

        VECGEOM_CUDA_HEADER_BOTH
        ~ThetaCone(){}

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::bool_v Contains( Vector3D<typename Backend::precision_v> const& point ) const{

            typedef typename Backend::bool_v Bool_t;
            Bool_t unused(false);
            Bool_t outside(false);
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
            //std::cout<<"Entered INSIDE of ThetaCone\n";
            typedef typename Backend::bool_v      Bool_t;
            Bool_t completelyinside(false), completelyoutside(false);
            GenericKernelForContainsAndInside<Backend,true>(
                 point, completelyinside, completelyoutside);
            typename Backend::inside_v  inside=EInside::kSurface;
            MaskedAssign(completelyoutside, EInside::kOutside, &inside);
            MaskedAssign(completelyinside, EInside::kInside, &inside);
			return inside;
        }

        /**
         * estimate of the smallest distance to the ThetaCone boundary when
         * the point is located outside the ThetaCone
         */
		template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::precision_v SafetyToIn( Vector3D<typename Backend::precision_v> const& point ) const {


            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;

			Float_t safeTheta(0.);
			Float_t pointRad = Sqrt(point.x()*point.x() + point.y()*point.y());
			Float_t sfTh1 = DistanceToLine<Backend>(slope1,pointRad, point.z());
			Float_t sfTh2 = DistanceToLine<Backend>(slope2,pointRad, point.z());

			safeTheta = Min(sfTh1,sfTh2);
			Bool_t done=Contains<Backend>(point);
			MaskedAssign(done,0., &safeTheta);
			if(IsFull(done)) return safeTheta;

			//Case 1 : Both cones are in Positive Z direction
			if(fSTheta < kPi/2 + halfAngTolerance)
              	{
                  if(fETheta < kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
						MaskedAssign((!done && point.z()<0.),sfTh2, &safeTheta);
		      		  }
		  		 }

				//Case 2 : First Cone is in Positive Z direction and Second is in Negative Z direction
				if(fETheta > kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
						MaskedAssign((!done && point.z()>0.),sfTh1, &safeTheta);
						MaskedAssign((!done && point.z()<0.),sfTh2, &safeTheta);
					  }
		 		 }
				}

			//Case 3 : Both cones are in Negative Z direction
			if(fETheta > kPi/2 + halfAngTolerance)
             {
				if(fSTheta > kPi/2 + halfAngTolerance)
                     {
			 	if(fSTheta < fETheta)
                      		{
								MaskedAssign((!done && point.z()>0.),sfTh1, &safeTheta);
	   						}
		     		}
	     	}


	return safeTheta;

  }

        /**
         * estimate of the smallest distance to the ThetaCone boundary when
         * the point is located inside the ThetaCone ( within the defining phi angle )
         */
        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
        typename Backend::precision_v SafetyToOut( Vector3D<typename Backend::precision_v> const& point ) const{

            typedef typename Backend::precision_v Float_t;
            //typedef typename Backend::bool_v      Bool_t;

	Float_t safeTheta(0.);
	Float_t pointRad = Sqrt(point.x()*point.x() + point.y()*point.y());
	Float_t bisectorRad = Abs(point.z() * tanBisector);

	//Case 1 : Both cones are in Positive Z direction
	if(fSTheta < kPi/2 + halfAngTolerance)
              {
                  if(fETheta < kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                          MaskedAssign(fSTheta==0.,DistanceToLine<Backend>(slope2,pointRad, point.z()),&safeTheta);
			CondAssign((pointRad < bisectorRad) && (fSTheta!=Float_t(0.)) ,DistanceToLine<Backend>(slope1,pointRad, point.z()),DistanceToLine<Backend>(slope2,pointRad, point.z()),&safeTheta);

		     }
		  }

	//Case 2 : First Cone is in Positive Z direction and Second is in Negative Z direction
		if(fETheta > kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
			Float_t sfTh1 = DistanceToLine<Backend>(slope1,pointRad, point.z());
			Float_t sfTh2 = DistanceToLine<Backend>(slope2,pointRad, point.z());
			safeTheta = sfTh1;
			MaskedAssign((sfTh2 < sfTh1),sfTh2,&safeTheta);
		      }
		 }

	     }

	//Case 3 : Both cones are in Negative Z direction
	if(fETheta > kPi/2 + halfAngTolerance)
             {
		if(fSTheta > kPi/2 + halfAngTolerance)
                     {
			 if(fSTheta < fETheta)
                      		{
                      		    //MaskedAssign(fSTheta==0.,DistanceToLine<Backend>(slope2,pointRad, point.z()),&safeTheta);
                      		    MaskedAssign(fETheta==kPi,DistanceToLine<Backend>(slope1,pointRad, point.z()),&safeTheta);
					CondAssign((pointRad < bisectorRad) && /*(fSTheta!=0.) && */(fETheta!=Float_t(kPi)),DistanceToLine<Backend>(slope2,pointRad, point.z()),DistanceToLine<Backend>(slope1,pointRad, point.z()),&safeTheta);
	   			}
		     }
	     }

		return safeTheta;

	


        }

	template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	//typename Backend::precision_v DistanceToLine(typename Backend::precision_v const& slope, typename Backend::precision_v const& x, typename Backend::precision_v const& y ) const{
	typename Backend::precision_v DistanceToLine(Precision const& slope, typename Backend::precision_v const& x, typename Backend::precision_v const& y ) const{

		typedef typename Backend::precision_v Float_t;
		Float_t dist = (y - slope*x)/Sqrt(1.+ slope*slope);
		return Abs(dist);

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

            {
            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;


            Bool_t done(false);
            Bool_t fal(false);


            Float_t a,b,c,d2;
            Float_t a2,b2,c2,d22;


            Float_t firstRoot(kInfinity), secondRoot(kInfinity);

            Float_t pDotV2d = point.x() * dir.x() + point.y() * dir.y();
            Float_t rho2 = point.x() * point.x() + point.y() * point.y();

            b = pDotV2d - point.z() * dir.z() * tanSTheta2 ;
            a = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;
            c = rho2 - point.z() * point.z() * tanSTheta2;
            d2 = b * b - a * c;

            MaskedAssign((d2 > 0.), (-1*b + Sqrt(d2))/a, &firstRoot);
            done |= (Abs(firstRoot) < 3.0*kTolerance);
            MaskedAssign( ( (Abs(firstRoot) < 3.0*kTolerance)),0., &firstRoot);
            MaskedAssign( ( !done && (firstRoot < 0.)) ,kInfinity,&firstRoot);

            b2 = pDotV2d - point.z() * dir.z() * tanETheta2;
            a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2;
            c2 = rho2 - point.z() * point.z() * tanETheta2;
            d22 = b2 * b2 - a2 * c2;

            MaskedAssign((d22 > 0.), (-1*b2 - Sqrt(d22))/a2, &secondRoot);
            //done = fal;
            //done |= (Abs(secondRoot) < 3.0*kSTolerance*10.);
            MaskedAssign( (!done && (Abs(secondRoot) < 3.0*kTolerance)),0., &secondRoot);
	    done |= (Abs(secondRoot) < 3.0*kTolerance);
            MaskedAssign(!done && (secondRoot < 0.) ,kInfinity, &secondRoot);

              if(fSTheta < kPi/2 + halfAngTolerance)
              {
                  if(fETheta < kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          distThetaCone2 = secondRoot;
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

                          intsect1 = ((d2 > 0) /*&& (distThetaCone1!=kInfinity)*/ && (zOfIntSecPtCone1 > 0.));
                          intsect2 = ((d22 > 0) /*&& (distThetaCone2!=kInfinity)*/ && (zOfIntSecPtCone2 > 0.));

                      }
                  }

                  if(fETheta >= kPi/2 - halfAngTolerance && fETheta <= kPi/2 + halfAngTolerance)
                  {
                      MaskedAssign((dir.z() > 0.),-1. * point.z() / dir.z() , &distThetaCone2);
                      Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                      intsect2 = ((distThetaCone2!=kInfinity) && (Abs(zOfIntSecPtCone2) < halfAngTolerance));

                  }

                  if(fETheta > kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          MaskedAssign((d22 > 0.), (-1*b2 + Sqrt(d22))/a2, &secondRoot);

                          done = fal;
                          done |= (Abs(secondRoot) < 3.0*kTolerance);
                          MaskedAssign( ( (Abs(secondRoot) < 3.0*kTolerance)),0., &secondRoot);
                          MaskedAssign(!done && (secondRoot < 0.) ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;

                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

                          intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 > 0.));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));

                      }
                  }

                  }

              if(fSTheta >= kPi/2 - halfAngTolerance)
              {
                  if(fETheta > kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                        MaskedAssign((d2 > 0.), (-1*b - Sqrt(d2))/a, &firstRoot);
                        done = fal;
                        done |= (Abs(firstRoot) < 3.0*kTolerance);
                        MaskedAssign( ( (Abs(firstRoot) < 3.0*kTolerance)),0., &firstRoot);
                        MaskedAssign(!done && (firstRoot < 0.) ,kInfinity, &firstRoot);
                        distThetaCone1 = firstRoot;

                        MaskedAssign((d22 > 0.), (-1*b2 + Sqrt(d22))/a2, &secondRoot);
                        done = fal;
                        done |= (Abs(secondRoot) < 3.0*kTolerance);
                        MaskedAssign( ( (Abs(secondRoot) < 3.0*kTolerance)),0., &secondRoot);
                        MaskedAssign(!done && (secondRoot < 0.) ,kInfinity, &secondRoot);
                        distThetaCone2 = secondRoot;

                        Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                        Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

                        intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 < 0.));
                        intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));

                      }
                  }

              }

           if(fSTheta >= kPi/2 - halfAngTolerance && fSTheta <= kPi/2 + halfAngTolerance)
                  {
                      MaskedAssign((dir.z() < 0.),-1. * point.z() / dir.z() , &distThetaCone1);
                      Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                      intsect1 = ((distThetaCone1!=kInfinity) && (Abs(zOfIntSecPtCone1) < halfAngTolerance));

                  }

		//std::cout<<"DistThetaCone-1 : "<<distThetaCone1<<"  :: DistThetaCone-2 : "<<distThetaCone2<<std::endl;
            MaskedAssign((distThetaCone1 < halfAngTolerance), 0.,&distThetaCone1);
            MaskedAssign((distThetaCone2 < halfAngTolerance), 0.,&distThetaCone2);

            }



	}

        template<typename Backend>
        VECGEOM_CUDA_HEADER_BOTH
	void DistanceToOut(Vector3D<typename Backend::precision_v> const &point,
           Vector3D<typename Backend::precision_v> const &dir,typename  Backend::precision_v &distThetaCone1,typename  Backend::precision_v &distThetaCone2 ,
                            typename Backend::bool_v &intsect1, typename Backend::bool_v &intsect2) const{

            //  bool verbose=false;


            typedef typename Backend::precision_v Float_t;
            typedef typename Backend::bool_v      Bool_t;
			Bool_t tr(true);
            Float_t a,b,c,d2;
            Float_t a2,b2,c2,d22;

            Float_t inf(kInfinity);

            Float_t firstRoot(kInfinity), secondRoot(kInfinity);

            Float_t pDotV2d = point.x() * dir.x() + point.y() * dir.y();
            Float_t rho2 = point.x() * point.x() + point.y() * point.y();

            b = pDotV2d - point.z() * dir.z() * tanSTheta2 ;
            a = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanSTheta2;
            c = rho2 - point.z() * point.z() * tanSTheta2;
            d2 = b * b - a * c;
            MaskedAssign(d2<0. && Abs(d2)<kHalfTolerance,0.,&d2);

            MaskedAssign((d2 >= 0.) && b>=0. && a!=0. ,( (-b - Sqrt(d2))/(a) ),&firstRoot);
            MaskedAssign((d2 >= 0.) && b<0.,((c)/(-b + Sqrt(d2)) ),&firstRoot);

            MaskedAssign(firstRoot < 0. ,kInfinity, &firstRoot);

            b2 = point.x() * dir.x() + point.y() * dir.y() - point.z() * dir.z() * tanETheta2;
            a2 = dir.x() * dir.x() + dir.y() * dir.y() - dir.z() * dir.z() * tanETheta2; ;
            c2 = point.x() * point.x() + point.y() * point.y() - point.z() * point.z() * tanETheta2;
            d22 = (b2 * b2) - (a2 * c2);
            MaskedAssign(d22<0. && Abs(d22)<kHalfTolerance,0.,&d22);

            MaskedAssign((d22 >= 0.) && b2>=0. ,((c2)/(-b2 - Sqrt(d22)) ),&secondRoot);
            MaskedAssign((d22 >= 0.) && b2<0. && a2!=0.,((-b2 + Sqrt(d22))/a2 ),&secondRoot);

            MaskedAssign(secondRoot < 0. && Abs(secondRoot) > kTolerance ,kInfinity, &secondRoot);
            MaskedAssign(Abs(secondRoot) < kTolerance,0.,&secondRoot);

			//std::cout<<std::setprecision(15);
			//std::cout<<"A : "<<a<<"  :: B : "<<b<<"  :: C : "<<c<<"  :: D2 : "<<d2<<std::endl;
			//std::cout<<"A2 : "<<a2<<"  :: B2 : "<<b2<<"  :: C2 : "<<c2<<"  :: D22 : "<<d22<<std::endl;

              if(fSTheta < kPi/2 + halfAngTolerance)
              {
                  if(fETheta < kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          distThetaCone2 = secondRoot;
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

                          //intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 >= 0.));
                          //intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 >= 0.));
						  intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && ((zOfIntSecPtCone1) > -kHalfTolerance));
                          intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && ((zOfIntSecPtCone2) > -kHalfTolerance));

                          Float_t dirRho2 = dir.x()*dir.x() + dir.y()*dir.y();
                          Float_t zs(kInfinity);
                          if(fSTheta)
                            zs=dirRho2/tanSTheta;
                          Float_t ze(kInfinity);
                          if(fETheta)
                            ze=dirRho2/tanETheta;
                          Bool_t cond = (point.x()==0. && point.y()==0. && point.z()==0. && dir.z()<zs && dir.z()<ze);
                          MaskedAssign(cond,0.,&distThetaCone1);
                          MaskedAssign(cond,0.,&distThetaCone2);
                          MaskedAssign(cond,tr,&intsect1); //intsect1=cond;
                          MaskedAssign(cond,tr,&intsect2); //intsect2=cond;

                      }
                  }

                  if(fETheta >= kPi/2 - halfAngTolerance && fETheta <= kPi/2 + halfAngTolerance)
                  {
                      distThetaCone1 = firstRoot;
                      distThetaCone2 = inf;
                      MaskedAssign((dir.z() < 0.),-1. * point.z() / dir.z() , &distThetaCone2);
                      Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                      Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                      intsect2 = ((d22 >= 0) && (distThetaCone2!=kInfinity) && (Abs(zOfIntSecPtCone2) < kHalfTolerance) && !(dir.z()==0.));
                      intsect1 = ((d2 >= 0) && (distThetaCone1!=kInfinity) && (Abs(zOfIntSecPtCone1) < kHalfTolerance) && !(dir.z()==0.));
                  }

                  if(fETheta > kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                          distThetaCone1 = firstRoot;
                          //MaskedAssign((d22 > 0.), (-1*b2 - Sqrt(d22))/a2, &secondRoot);
                          MaskedAssign((d22 >= 0.) && b2>0. && a2!=0.,( (-b2 - Sqrt(d22))/(a2) ),&secondRoot);
                          MaskedAssign((d22 >= 0.) && b2<=0.,((c2)/(-b2 + Sqrt(d22)) ),&secondRoot);
						  //std::cout<<"FirstRoot : "<<firstRoot<<"  :: SecondRoot : "<<secondRoot<<std::endl;
                          MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;
		
						  //std::cout<<"DistThetaCone1 : "<<distThetaCone1<<std::endl;
						  //std::cout<<"DistThetaCone2 : "<<distThetaCone2<<std::endl;
						  //std::cout<<"--------------------------------------------------------\n";
                          Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                          Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

						  //std::cout<<"zOfIntSecPtCone1 : "<<zOfIntSecPtCone1<<std::endl;
						  //std::cout<<"zOfIntSecPtCone2 : "<<zOfIntSecPtCone2<<std::endl;
						  //std::cout<<"--------------------------------------------------------\n";

                          //intsect1 = ((d2 >= 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 >= 0.));
                          //intsect2 = ((d22 >= 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 <= 0.));
						  intsect1 = ((d2 >= 0) && (distThetaCone1!=kInfinity) && ((zOfIntSecPtCone1) > -kHalfTolerance));
                          intsect2 = ((d22 >= 0) && (distThetaCone2!=kInfinity) && ((zOfIntSecPtCone2) < kHalfTolerance));
						
						//std::cout<<"Insect-1 : "<<intsect1<<std::endl;
						//std::cout<<"Insect-2 : "<<intsect2<<std::endl;
	
            
                      }
                  }
              }

                  if(fETheta > kPi/2 + halfAngTolerance)
                  {
                      if(fSTheta < fETheta)
                      {
                          MaskedAssign((d22 >= 0.) && b2>0. && a2!=0.,( (-b2 - Sqrt(d22))/(a2) ),&secondRoot);
                            MaskedAssign((d22 >= 0.) && b2<=0.,((c2)/(-b2 + Sqrt(d22)) ),&secondRoot);
                          MaskedAssign(secondRoot < 0. ,kInfinity, &secondRoot);
                          distThetaCone2 = secondRoot;

                          if(fSTheta > kPi/2 + halfAngTolerance)
                          {
                            MaskedAssign((d2 >= 0.) && b>0. ,( (c)/(-b - Sqrt(d2))),&firstRoot);
                            MaskedAssign((d2 >= 0.) && b<=0. && a!=0.,((-b + Sqrt(d2))/(a) ),&firstRoot);
                            MaskedAssign(firstRoot < 0. ,kInfinity, &firstRoot);
                            distThetaCone1 = firstRoot;
                            Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());
                            //intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && (zOfIntSecPtCone1 < 0.));
						  intsect1 = ((d2 > 0) && (distThetaCone1!=kInfinity) && ((zOfIntSecPtCone1) < kHalfTolerance));
            
			              Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());
                          //intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && (zOfIntSecPtCone2 < 0.));
						  intsect2 = ((d22 > 0) && (distThetaCone2!=kInfinity) && ((zOfIntSecPtCone2) < kHalfTolerance));

                          Float_t dirRho2 = dir.x()*dir.x() + dir.y()*dir.y();
                          Float_t zs(-kInfinity);
                          if(tanSTheta)
                            zs=-dirRho2/tanSTheta;
                          Float_t ze(-kInfinity);
                          if(tanETheta)
                            ze=-dirRho2/tanETheta;
                          Bool_t cond = (point.x()==0. && point.y()==0. && point.z()==0. && dir.z()>zs && dir.z()>ze);
                          MaskedAssign(cond,0.,&distThetaCone1);
                          MaskedAssign(cond,0.,&distThetaCone2);
                          MaskedAssign(cond,tr,&intsect1); //intsect1=cond;
                          MaskedAssign(cond,tr,&intsect2); //intsect2=cond;
                         }
                      }
                  }

            if(fSTheta >= kPi/2 - halfAngTolerance && fSTheta <= kPi/2 + halfAngTolerance)
                  {
                      distThetaCone2 = secondRoot;
                      distThetaCone1 = inf;
                      MaskedAssign((dir.z() > 0.),-1. * point.z() / dir.z() , &distThetaCone1);
                      Float_t zOfIntSecPtCone1 = (point.z() + distThetaCone1 * dir.z());

                      Float_t zOfIntSecPtCone2 = (point.z() + distThetaCone2 * dir.z());

                      intsect1 = ((d2 >= 0) && (distThetaCone1!=kInfinity) && (Abs(zOfIntSecPtCone1) < kHalfTolerance) && !(dir.z()==0.));
                      intsect2 = ((d22 >= 0) && (distThetaCone2!=kInfinity) && (Abs(zOfIntSecPtCone2) < kHalfTolerance) && !(dir.z()==0.));


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
            typedef typename Backend::bool_v Bool_t;
	Float_t pi(kPi),zero(0.);
	Float_t rad = Sqrt(localPoint.Mag2() - (localPoint.z() * localPoint.z()));
	Float_t cone1Radius = Abs(localPoint.z()*tanSTheta);
	Float_t cone2Radius = Abs(localPoint.z()*tanETheta);
	Bool_t isPointOnZAxis = localPoint.z()!=zero && localPoint.x()==zero && localPoint.y()==zero;
	
	Bool_t isPointOnXYPlane = localPoint.z()==zero && (localPoint.x()!=zero || localPoint.y()!=zero) ;
	
	Float_t startTheta(fSTheta),endTheta(fETheta);
														
	
	completelyoutside = (isPointOnZAxis && ((startTheta!=zero && endTheta!=pi) || (localPoint.z()>zero && startTheta!=zero ) || (localPoint.z()<zero && endTheta!=pi))) ;
	
	completelyinside = (isPointOnZAxis && ((startTheta==zero && endTheta==pi) || (localPoint.z()>zero && startTheta==zero) || (localPoint.z()<zero && endTheta==pi) ));
	
	MaskedAssign(!completelyoutside,(isPointOnXYPlane && ((startTheta<pi/2 && endTheta<pi/2 && (pi/2 - startTheta)>kAngTolerance && (pi/2 - endTheta)>kTolerance)
		|| (startTheta>pi/2 && endTheta>pi/2 && (startTheta-pi/2)>kAngTolerance && (endTheta-pi/2)>kTolerance))),&completelyoutside);
	
		
	MaskedAssign(!completelyinside, (isPointOnXYPlane && (startTheta<pi/2 && endTheta>pi/2 && (pi/2 - startTheta)>kAngTolerance && (endTheta-pi/2)>kTolerance)),&completelyinside);	
		 
	if(fSTheta < kPi/2 + halfAngTolerance )
              {
                  if(fETheta < kPi/2 + halfAngTolerance)
                  {  
                      if(fSTheta < fETheta)
                      {
			Float_t tolAngMin = cone1Radius + 2*kAngTolerance*10.;
            		Float_t tolAngMax = cone2Radius - 2*kAngTolerance*10.;
			if(ForInside)
            {
    
                	MaskedAssign(!completelyinside,( ((rad <= tolAngMax) && (rad >= tolAngMin) && (localPoint.z() > zero) && (fSTheta!=zero)) || ((rad <= tolAngMax) && (fSTheta==zero) && (localPoint.z() > zero) )  ),&completelyinside);
    

            }

            		Float_t tolAngMin2 = cone1Radius - 2*kAngTolerance*10.;
            		Float_t tolAngMax2 = cone2Radius + 2*kAngTolerance*10.;
    
			MaskedAssign(!completelyoutside,((rad < tolAngMin2) || (rad > tolAngMax2) || (localPoint.z() < 0.)),&completelyoutside);
	
					
            }
		  }

		if(fETheta > kPi/2 + halfAngTolerance)
                  {   if(fSTheta < fETheta)
                      {
			Float_t tolAngMin = cone1Radius + 2*kAngTolerance*10.;
            		Float_t tolAngMax = cone2Radius + 2*kAngTolerance*10.;
					if(ForInside)
            		
					MaskedAssign(!completelyinside,(((rad >= tolAngMin) && (localPoint.z() > 0.)) || ((rad >= tolAngMax) && (localPoint.z() < 0.))),&completelyinside);
            		Float_t tolAngMin2 = cone1Radius - 2*kAngTolerance*10.;
            		Float_t tolAngMax2 = cone2Radius - 2*kAngTolerance*10.;
            		MaskedAssign(!completelyoutside,(((rad < tolAngMin2) && (localPoint.z() > 0.))  || ((rad < tolAngMax2) && (localPoint.z() < 0.))),&completelyoutside);
            		
		      }
		  }
			
            if(fETheta >= kPi/2 - halfAngTolerance && fETheta <= kPi/2 + halfAngTolerance)
            {
    
                    MaskedAssign(Abs(localPoint.z())<halfAngTolerance ,Bool_t(false) ,&completelyinside);
                    MaskedAssign(Abs(localPoint.z())<halfAngTolerance ,Bool_t(false) ,&completelyoutside);
            }
            
	      }

	if(fETheta > kPi/2 + halfAngTolerance )
             {
             if(fSTheta >= kPi/2 - halfAngTolerance && fSTheta <= kPi/2 + halfAngTolerance)
            {
    
                    MaskedAssign(Abs(localPoint.z())<halfAngTolerance ,Bool_t(false) ,&completelyinside);
                    MaskedAssign(Abs(localPoint.z())<halfAngTolerance ,Bool_t(false) ,&completelyoutside);
            }
            
		if(fSTheta > kPi/2 + halfAngTolerance)
                     { 
			 if(fSTheta < fETheta)
                      		{
					Float_t tolAngMin = cone1Radius - 2*kAngTolerance*10.;
            				Float_t tolAngMax = cone2Radius + 2*kAngTolerance*10.;
					if(ForInside)
                    {
                    
                    MaskedAssign(!completelyinside,(((rad <= tolAngMin) && (rad >= tolAngMax) && (localPoint.z() < zero) && (fETheta!=pi)) || ((rad <= tolAngMin) && (localPoint.z() < zero) && (fETheta==pi))),&completelyinside);
                       
                    }

            				Float_t tolAngMin2 = cone1Radius + 2*kAngTolerance*10.;
            				Float_t tolAngMax2 = cone2Radius - 2*kAngTolerance*10.;
           				MaskedAssign(!completelyoutside,((rad < tolAngMax2) || (rad > tolAngMin2) || (localPoint.z() > 0.)),&completelyoutside);
           				

				}

		     }
	     }


    }



}; // end of class ThetaCone

} } // end of namespace


#endif /* VECGEOM_VOLUMES_THETACONE_H_ */
