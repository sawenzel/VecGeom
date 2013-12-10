/*
 * PhysicalCone.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */
#ifndef PHYSICALCONE_H_
#define PHYSICALCONE_H_

#include "PhysicalVolume.h"
#include "ConeTraits.h"

template <typename T=double>
// T is a floating point type
class ConeParameters // : ShapeParameters
{
private:
	T dRmin1; // inner radius at z=-dz
	T dRmax1; // outer radius
	T dRmin2; // inner radius at z=dz
	T dRmax2;
	T dZ;    // halfLength in z direction
	T dSPhi; // starting angle in radians
	T dDPhi; // delta angle of segment in radians

	// for caching
	T cacheRminSqr; // rminsquared
	T cacheRmaxSqr; // rmaxsquared
	T cacheTolORminSqr; // tolerant outer radius of rmin
	T cacheTolORmaxSqr; // tolerant outer radius of rmax

	T cacheTolIRminSqr; // tolerant inner radius of rmin
	T cacheTolIRmaxSqr; // tolerant inner radius of rmax

	//**** we save normals to phi - planes *****//
	Vector3D normalPhi1;
	Vector3D normalPhi2;

public:
	ConeParameters(T pRmin1, T pRmax1, T pRmin2, T pRmax2, T pDZ, T pPhiMin, T pPhiMax) :
		dRmin1(pRmin1),
		dRmax1(pRmax1),
		dRmin2(pRmin2),
		dRmax2(pRmax2),
		dZ(pDZ),
		dSPhi(pPhiMin),
		dDPhi(pPhiMax) {
			// calculate caches
			cacheRminSqr=dRmin1*dRmin1;
			cacheRmaxSqr=dRmax1*dRmax1;

			if ( dRmin1 > Utils::GetRadHalfTolerance() )
			{
				// CHECK IF THIS CORRECT ( this seems to be inversed with tolerance for ORmax
				cacheTolORminSqr = (dRmin1 - Utils::GetRadHalfTolerance()) * (dRmin1 - Utils::GetRadHalfTolerance());
				cacheTolIRminSqr = (dRmin1 + Utils::GetRadHalfTolerance()) * (dRmin1 + Utils::GetRadHalfTolerance());
			}
			else
			{
				cacheTolORminSqr = 0.0;
				cacheTolIRminSqr = 0.0;
			}

			cacheTolORmaxSqr = (dRmax1 + Utils::GetRadHalfTolerance()) * (dRmax1 + Utils::GetRadHalfTolerance());
			cacheTolIRmaxSqr = (dRmax1 - Utils::GetRadHalfTolerance()) * (dRmax1 - Utils::GetRadHalfTolerance());

			// calculate normals
			GeneralPhiUtils::GetNormalVectorToPhiPlane(dSPhi, normalPhi1, true);
			GeneralPhiUtils::GetNormalVectorToPhiPlane(dSPhi + dDPhi, normalPhi2, false);

			normalPhi1.print();
			normalPhi2.print();
	};

	//	virtual void inspect() const;
	inline T GetRmin1() const {return dRmin1;}
	inline T GetRmax1() const {return dRmax1;}
	inline T GetRmin2() const {return dRmin2;}
	inline T GetRmax2() const {return dRmax2;}
	inline T GetDZ() const {return dZ;}
	inline T GetSPhi() const {return dSPhi;}
	inline T GetDPhi() const {return dDPhi;}

	virtual ~ConeParameters(){};
	// The placed boxed can easily access the private members
	template<int,int,class,class> friend class PlacedCone;

	// template<int,int,int> friend class PlacedRootTube;
};



template<TranslationIdType tid, RotationIdType rid, class ConeType=ConeTraits::HollowConeWithPhi, class T=double>
class
PlacedCone : public PhysicalVolume
{
	ConeParameters<T> const * coneparams;

public:
	PlacedCone( ConeParameters<T> const * _cp, TransformationMatrix const *m ) : PhysicalVolume(m), coneparams(_cp) {
		this->bbox = new PlacedBox<1,0>( new BoxParameters( coneparams->dRmax2, coneparams->dRmax2, coneparams->dZ), new TransformationMatrix(0,0,0,0,0,0) );
	};

	// ** functions to implement
	__attribute__((always_inline))
	virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const {return 0.;}


	virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const {return 0.;}
	virtual bool   Contains( Vector3D const & ) const {return false;}

	__attribute__((always_inline))
	inline
	virtual bool   UnplacedContains( Vector3D const & ) const {return false;}

	virtual double SafetyToIn( Vector3D const & ) const {return 0.;}
	virtual double SafetyToOut( Vector3D const & ) const {return 0.;}

	// for basket treatment (supposed to be dispatched to particle parallel case)
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};

	// for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
	virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {} ;
	virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const {};


	// a template version for T = Vc or T = Boost.SIMD or T= double
	template<typename VectorType>
	inline
	__attribute__((always_inline))
	void DistanceToIn( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
					   VectorType const & /*dx-vec*/, VectorType const & /*dy-vec*/, VectorType const & /*dz-vec*/, VectorType const & /*step*/, VectorType & /*result*/ ) const {};


};




#endif /* PHYSICALCONE_H_ */
