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

	// some cone specific stuff
	T innerslope; // "gradient" of inner surface in z direction
	T outerslope; // "gradient" of outer surface in z direction
	T inneroffset;
	T outeroffset;

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
		// the possible caches are one major difference between tube and cone

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


	// some helper function (mainly useful to debug)
	Vector3D RHit( Vector3D const & /*pos*/,
				   Vector3D const & /*dir*/,
				   T & /* distance */,
				   T /* radiusatminusz */,
				   T /* radiusatplusz */,
				   bool /* wantMinusSolution */ ) const ;

	bool isInRRange( Vector3D const & x ) const;
	bool isInPhiRange( Vector3D const & x ) const;
	bool isInZRange( Vector3D const & x ) const;
	Vector3D PhiHit( Vector3D const &x, Vector3D const &y, double & distance, bool isPhi1Solution ) const;
	Vector3D ZHit( Vector3D const &x, Vector3D const &y, double & distance ) const;

	Vector3D RHit( Vector3D const &x, Vector3D const &y, T & distance, T radius, bool wantMinusSolution ) const;
	void printInfoHitPoint( char const * c, Vector3D const & vec, double distance ) const;
};


template<int tid, int rid, class TubeType, typename T>
void
PlacedCone<tid,rid,TubeType,T>::printInfoHitPoint( char const * c, Vector3D const & vec, double distance ) const
{
	std::cout << c << " " << vec << "\t\t at dist " << distance << " in z " << GeneralPhiUtils::IsInRightZInterval( vec, coneparams->dZ )
					<< " inPhi " << isInPhiRange( vec )
					<< " inR " << isInRRange( vec )
					<< std::endl;
}

template<int tid, int rid, class TubeType, typename T>
bool
PlacedCone<tid,rid, TubeType,T>::isInRRange( Vector3D const & pos) const
{
	double d = Vector3D::scalarProductInXYPlane(pos, pos);

	// we also need z coordinate for the cone
	T outerradius = coneparams->outerslope*pos.z+coneparams->outeroffset;
	T innerradius = coneparams->innerslope*pos.z+coneparams->inneroffset;
	// we need rad

	return ( innerradius <= d && d <= outerradius );
}

template <int tid, int rid, class ConeType, typename T>
Vector3D PlacedCone<tid,rid, ConeType, T>::RHit( Vector3D const & pos, Vector3D const & dir,
												 T & distance,
												 T m,
												 T n,
												 bool wantMinusSolution ) const
{
	// m = z-gradient of cone surface
	// T m = -( radiusatminusz - radiusatplusz )/(2*coneparams->dZ);
	// T n = radiusatplusz - m*coneparams->dZ;

	T a = 1 - dir.z*dir.z*(1-m*m);
	T b = 2 * ( pos.x*dir.x + pos.y*dir.y - pos.z*dir.z*m*m - 2*m*n * dir.z);
	T c = (pos.x*pos.x + pos.y*pos.y - m*m*pos.z*pos.z - 2*m*n*pos.z + n*n);

	T discriminant = b*b - 4*a*c;

	if(discriminant >= 0)
	{
		if( wantMinusSolution ) distance = ( -b - sqrt(discriminant) );
		if( ! wantMinusSolution ) distance = (-b + sqrt(discriminant) );
	}
	else
	{
		distance = Utils::kInfinity;
	}
	Vector3D vec( pos.x + distance*dir.x, pos.y + distance*dir.y, pos.z + distance*dir.z );
	return vec;
}

template<int tid, int rid, class TubeType, typename T>
bool
PlacedCone<tid,rid, TubeType,T>::isInPhiRange( Vector3D const & pos) const
{
	return GeneralPhiUtils::PointIsInPhiSector<T>( coneparams->normalPhi1, coneparams->normalPhi2, pos );
}

template<int tid, int rid, class TubeType, typename T>
Vector3D
PlacedCone<tid,rid, TubeType,T>::PhiHit ( Vector3D const & pos, Vector3D const & dir, double & distance, bool isPhi1Solution ) const
{
	Vector3D const & normal = ( isPhi1Solution ) ? coneparams->normalPhi1 : coneparams->normalPhi2;
	T scalarproduct1 = normal.x*pos.x + normal.y*pos.y;
	T N1dotDir = normal.x*dir.x + normal.y*dir.y;

	distance = -scalarproduct1/N1dotDir;
	Vector3D vec( pos.x + distance*dir.x, pos.y + distance*dir.y, pos.z + distance*dir.z );
	return vec;
}

template<int tid, int rid, class TubeType, typename T>
Vector3D
PlacedCone<tid,rid, TubeType, T>::ZHit( Vector3D const & pos, Vector3D const & dir, double & distance ) const
{
	distance = - (coneparams->dZ - std::abs(pos.z))/std::abs(dir.z);
	Vector3D vec( pos.x + distance*dir.x, pos.y + distance*dir.y, pos.z + distance*dir.z );
	return vec;
}

template<int tid, int rid, class TubeType, typename T>
void
PlacedCone<tid,rid,TubeType,T>::DebugPointAndDirDistanceToIn( Vector3D const & x, Vector3D const & y ) const
{
	std::cout << "INSPECTING POINT - TUBE INTERACTION ALONG FLIGHT-PATH " << std::endl;
	std::cout << "PARTICLE POSITION " << std::endl;
	double distance=0;
	printInfoHitPoint("ParticlePos", x, distance );
	std::cout << "PARTICLE DIRECTION " << std::endl;
	std::cout << y << std::endl;
	std::cout << "RMAX INTERACTION " << std::endl;
	Vector3D vmax = this->RHit( x, y, distance, coneparams->outerslope, coneparams->outeroffset, true);
	printInfoHitPoint("hitpointrmax", vmax, distance);

	if( ConeTraits::NeedsRminTreatment<TubeType>::value )
	{
		std::cout << "RMIN INTERACTION " << std::endl;
		Vector3D vmin = this->RHit( x, y, distance, coneparams->innerslope, coneparams->inneroffset, false );
		printInfoHitPoint("hitpointrmin", vmin, distance);
	}

	if( ConeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		std::cout << "PHI INTERACTION " << std::endl;
		Vector3D vphi1 = this->PhiHit( x, y, distance, true);
		printInfoHitPoint("hitpointphi1", vphi1, distance);
		Vector3D vphi2 = this->PhiHit( x, y, distance, false);
		printInfoHitPoint("hitpointphi2", vphi2, distance);
	}

	std::cout << "Z INTERACTION " << std::endl;
	Vector3D vZ = this->ZHit( x, y, distance );
	printInfoHitPoint("hitpointZ", vZ, distance);
}


#endif /* PHYSICALCONE_H_ */
