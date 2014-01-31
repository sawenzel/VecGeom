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
#include "GlobalDefs.h"
#include "Vector3D.h"
#include "Utils.h"
#include "PhysicalBox.h"

// for stuff from USolids
#include "VUSolid.hh"
#include "UCons.hh"
#include "TGeoCone.h"

//Member Data:
//
//   fRmin1  inside radius at  -fDz
//   fRmin2  inside radius at  +fDz
//   fRmax1  outside radius at -fDz
//   fRmax2  outside radius at +fDz
//   fDz half length in z
//
//   fSPhi starting angle of the segment in radians
//   fDPhi delta angle of the segment in radians


template <typename T=double>
class
ConeParameters
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
	T cacheRmaxSqr;
	T cacheRmslORminSqr; // tolerant outer radius of rmin
	T cacheTolORmaxSqr; // tolerant outer radius of rmax
	T cacheTolORminSqr;

	T cacheTolIRminSqr; // tolerant inner radius of rmin
	T cacheTolIRmaxSqr; // tolerant inner radius of rmax


	// some cone specific stuff
	T innerslope; // "gradient" of inner surface in z direction
	T outerslope; // "gradient" of outer surface in z direction
	T inneroffset;
	T outeroffset;
	T outerslopesquare;
	T innerslopesquare;
	T outeroffsetsquare;
	T inneroffsetsquare;


public:
	//**** we save normals to phi - planes *****//
	Vector3D normalPhi1;
	Vector3D normalPhi2;
	//**** vectors along radial direction of phi-planes
	Vector3D alongPhi1;
	Vector3D alongPhi2;

	ConeParameters(T pRmin1, T pRmax1, T pRmin2, T pRmax2, T pDZ, T pPhiMin, T pPhiMax) :
		dRmin1(pRmin1),
		dRmax1(pRmax1),
		dRmin2(pRmin2),
		dRmax2(pRmax2),
		dZ(pDZ),
		dSPhi(pPhiMin),
		dDPhi(pPhiMax) {

			// check this very carefully
			innerslope = -(dRmin1 - dRmin2)/(2.*dZ);
			outerslope = -(dRmax1 - dRmax2)/(2.*dZ);
			inneroffset = dRmin2 - innerslope*dZ;
			outeroffset = dRmax2 - outerslope*dZ;
			outerslopesquare = outerslope*outerslope;
			innerslopesquare = innerslope*innerslope;
			outeroffsetsquare = outeroffset*outeroffset;
			inneroffsetsquare = inneroffset*inneroffset;

			// calculate caches
			// the possible caches are one major difference between tube and cone

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

			// get alongs
			GeneralPhiUtils::GetAlongVectorToPhiPlane(dSPhi, alongPhi1);
			GeneralPhiUtils::GetAlongVectorToPhiPlane(dSPhi + dDPhi, alongPhi2);

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
	inline T GetInnerSlope() const {return innerslope;}
	inline T GetOuterSlope() const {return outerslope;}
	inline T GetInnerOffset() const {return inneroffset;}
	inline T GetOuterOffset() const {return outeroffset;}


	virtual ~ConeParameters(){};
	// The placed boxed can easily access the private members
	template<int,int,class,class> friend class PlacedCone;

	// template<int,int,int> friend class PlacedRootTube;
};


// TODO: this is currently almost a one to one copy from the tube
// should be ideally the same kernels ( requiring some more template techniques )
struct ConeKernels
{

// this kernel could be used for the Cone two
__attribute__((always_inline))
inline
static
GlobalTypes::SurfaceEnumType
InsideZ( ConeParameters<> const * tp, Vector3D const & x )
{
	if( std::abs(x.z) - tp->GetDZ() > Utils::frHalfTolerance )
		return GlobalTypes::kOutside;

	if( std::abs(x.z) - tp->GetDZ() < -Utils::frHalfTolerance )
		return GlobalTypes::kInside;

	return GlobalTypes::kOnSurface;
}

template<typename Float_t=double>
__attribute__((always_inline))
inline
static
Float_t
GetInnerRAtZ(ConeParameters<Float_t> const * cp, Float_t z)
{
	return cp->GetInnerSlope() *z + cp->GetInnerOffset();
}

template<typename Float_t=double>
__attribute__((always_inline))
inline
static
Float_t
GetOuterRAtZ(ConeParameters<Float_t> const * cp, Float_t z)
{
	return cp->GetOuterSlope() *z + cp->GetOuterOffset();
}

template<typename Float_t=double>
__attribute__((always_inline))
inline
static
Float_t
GetRminSqrAtZ(ConeParameters<Float_t> const * cp, Float_t z)
{
	Float_t t = GetInnerRAtZ(cp, z);
	return t*t;
}


template<typename Float_t=double>
__attribute__((always_inline))
inline
static
Float_t
GetRmaxSqrAtZ(ConeParameters<Float_t> const * cp, Float_t z)
{
	Float_t t = GetOuterRAtZ(cp, z);
	return t*t;
}

// TODO: tolerances are likely wrong because we are treating squared variables
template<typename ConeType, typename Float_t=double>
__attribute__((always_inline))
inline
static
GlobalTypes::SurfaceEnumType
InsideR(ConeParameters<Float_t> const * tp, Vector3D const & x)
{
	Float_t r2 = x.x*x.x + x.y*x.y;

	Float_t rmaxsqr = GetRmaxSqrAtZ(tp, x.z);

	if( ! ConeTraits::NeedsRminTreatment<ConeType>::value )
	{
		if( r2 > rmaxsqr + Utils::frHalfTolerance )
		{
			return GlobalTypes::kOutside;
		}
		if( r2 < rmaxsqr - Utils::frHalfTolerance )
		{
			return GlobalTypes::kInside;
		}
		return GlobalTypes::kOnSurface;
	}
	else
	{
		Float_t rminsqr = GetRminSqrAtZ<Float_t>( tp, x.z );

		if( r2 > rmaxsqr + Utils::frHalfTolerance || r2 < rminsqr - Utils::frHalfTolerance )
		{
			return GlobalTypes::kOutside;
		}
		if( r2 < rmaxsqr - Utils::frHalfTolerance && r2 > rminsqr + Utils::frHalfTolerance )
		{
			return GlobalTypes::kInside;
		}
		return GlobalTypes::kOnSurface;
	}
}

template<typename ConeType, typename Float_t=double>
__attribute__((always_inline))
inline
static
GlobalTypes::SurfaceEnumType
// this kernel could be used for the cone two
InsidePhi( ConeParameters<Float_t> const * tp, Vector3D const & x )
{
	// we should catch here the case when we do not need phi treatment at all ( or assert on it )
	if( ! ConeTraits::NeedsPhiTreatment<ConeType>::value )
		return GlobalTypes::kInside;

	// a bit tricky; how can we
	// method based on calculating the scalar product of position vectors with the normals of the (empty) phi sektor
	// avoids taking the atan2

	// this method could be template specialized in case DeltaPhi = 180^o
	Float_t scalarproduct1 = Vector3D::scalarProductInXYPlane( tp->normalPhi1, x );

	// template specialize on some interesting cases
	if( ConeTraits::IsPhiEqualsPiCase<ConeType>::value )
	{
		// here we only have one plane
		if( scalarproduct1 > Utils::frHalfTolerance )
			return GlobalTypes::kOutside;

		// we are on the side of the plane where we might find a tube
		if ( scalarproduct1 < -Utils::frHalfTolerance )
			return GlobalTypes::kInside;

		return GlobalTypes::kOnSurface;
	}
	else // more general case for which we need to treat a second plane
	{
		Float_t scalarproduct2 = Vector3D::scalarProductInXYPlane( tp->normalPhi2, x );
		if ( tp->normalPhi1.x*tp->normalPhi2.x + tp->normalPhi1.y*tp->normalPhi2.y >= 0 )
		{
			// here angle between planes is < 180 degree
			if (scalarproduct1 > Utils::frHalfTolerance && scalarproduct2 > Utils::frHalfTolerance )
				return GlobalTypes::kOutside;
			if (scalarproduct1 < -Utils::frHalfTolerance && scalarproduct2 < -Utils::frHalfTolerance )
				return GlobalTypes::kInside;
			return GlobalTypes::kOnSurface;
		}
		else
		{
			// here angle between planes is > 180 degree
			if (scalarproduct1 > Utils::frHalfTolerance || scalarproduct2 > Utils::frHalfTolerance )
				return GlobalTypes::kOutside;
			if (scalarproduct1 < -Utils::frHalfTolerance || scalarproduct2 < -Utils::frHalfTolerance )
				return GlobalTypes::kOutside;

			return GlobalTypes::kOnSurface;
		}
	}
}


};

template<TranslationIdType tid, RotationIdType rid, typename ConeType=ConeTraits::HollowConeWithPhi, typename PrecType=double>
class
PlacedCone : public PhysicalVolume
{
private:
	ConeParameters<PrecType> const * coneparams;
	PlacedCone<0,1296,ConeType,PrecType> * unplacedcone;

public:
	inline PrecType GetRmin1() const { return coneparams->GetRmin2(); }
	inline PrecType GetRmax1() const { return coneparams->GetRmax2(); }
	inline PrecType GetRmin2() const { return coneparams->GetRmin2(); }
	inline PrecType GetRmax2() const { return coneparams->GetRmax2(); }
	inline PrecType GetDZ()   const { return coneparams->GetDZ();   }
	inline PrecType GetSPhi() const { return coneparams->GetSPhi(); }
	inline PrecType GetDPhi() const { return coneparams->GetDPhi(); }

	PlacedCone( ConeParameters<PrecType> const * _cp, TransformationMatrix const *m ) : PhysicalVolume(m), coneparams(_cp) {
		this->bbox = new PlacedBox<1,1296>( new BoxParameters( std::max(GetRmax2(), GetRmax1() ),
				std::max(GetRmax2(), GetRmax1()), GetDZ()), new TransformationMatrix(0,0,0,0,0,0) );

	    // initialize the equivalent usolid shape
		VUSolid * s = new UCons("internalucons", GetRmin1(), GetRmax2(),
				GetRmin2(), GetRmax2(), GetDZ(), GetSPhi(), GetDPhi());

		this->SetUnplacedUSolid( s );

		if( ConeTraits::NeedsPhiTreatment<ConeType>::value )
		{
			analogousrootsolid = new TGeoConeSeg(
									GetDZ(), GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2(),
									GetSPhi()*360/(2.*M_PI), GetSPhi()+360*GetDPhi()/(2.*M_PI) );
		}
		else
		{
			analogousrootsolid = new TGeoCone( GetDZ(), GetRmin1(), GetRmax1(), GetRmin2(), GetRmax2() );
		}

		if( ! (tid==0 && rid==1296) )
			unplacedcone = new  PlacedCone<0,1296,ConeType,PrecType>(_cp,m);
	};

	// ** functions to implement
	__attribute__((always_inline))
	virtual double DistanceToIn( Vector3D const & aPoint, Vector3D const & aDir, PrecType step ) const
	{
		// do coordinate trafo
		Vector3D tPoint;
		Vector3D tDir;
		matrix->MasterToLocal<tid,rid>(aPoint, tPoint);
		matrix->MasterToLocalVec<rid>(aDir, tDir);
		return analogoususolid->DistanceToIn( reinterpret_cast<UVector3 const &>(tPoint),
											  reinterpret_cast<UVector3 const &>(tDir), step );
	}


	virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const {return 0.;}
	virtual bool   Contains( Vector3D const & ) const;
	__attribute__((always_inline))
	inline
	virtual bool   UnplacedContains( Vector3D const & ) const;

	__attribute__((always_inline))
	inline
	virtual
	GlobalTypes::SurfaceEnumType UnplacedContains_WithSurface( Vector3D const & x ) const;
	__attribute__((always_inline))
	inline
	virtual
	GlobalTypes::SurfaceEnumType Contains_WithSurface( Vector3D const & x ) const;


	virtual double SafetyToIn( Vector3D const & ) const {return 0.;}
	virtual double SafetyToOut( Vector3D const & ) const {return 0.;}

	// for basket treatment (supposed to be dispatched to particle parallel case)
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const;

	// for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
	virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {} ;
	virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const {};


	// a template version for T = Vc or T = Boost.SIMD or T= double
	template<typename VectorType>
	inline
	__attribute__((always_inline))
	void DistanceToIn( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
					   VectorType const & /*dx-vec*/, VectorType const & /*dy-vec*/, VectorType const & /*dz-vec*/,
					   VectorType const & /*step*/, VectorType & /*result*/ ) const;


	// some helper function (mainly useful to debug)
	Vector3D RHit( Vector3D const & /*pos*/,
				   Vector3D const & /*dir*/,
				   PrecType & /* distance */,
				   PrecType /* radiusatminusz */,
				   PrecType /* radiusatplusz */,
				   bool /* wantMinusSolution */ ) const ;

	bool isInRRange( Vector3D const & x ) const;
	bool isInPhiRange( Vector3D const & x ) const;
	bool isInZRange( Vector3D const & x ) const;
	Vector3D PhiHit( Vector3D const &x, Vector3D const &y, double & distance, bool isPhi1Solution ) const;
	Vector3D ZHit( Vector3D const &x, Vector3D const &y, double & distance ) const;

	Vector3D RHit( Vector3D const &x, Vector3D const &y, PrecType & distance, PrecType radius, bool wantMinusSolution ) const;
	void printInfoHitPoint( char const * c, Vector3D const & vec, double distance ) const;
	virtual void DebugPointAndDirDistanceToIn( Vector3D const & x, Vector3D const & y) const;

	template<typename VectorType = Vc::Vector<PrecType> >
	inline
	__attribute__((always_inline))
	typename VectorType::Mask determineRHit( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
												VectorType const & /*dirx-vec*/, VectorType const & /*diry-vec*/, VectorType const & /*dirz-vec*/, VectorType const & /**/ ) const;

	template<typename VectorType = Vc::Vector<PrecType> >
	inline
	__attribute__((always_inline))
	typename VectorType::Mask determineZHit( VectorType const & /*x-vec*/, VectorType const & /*y-vec*/, VectorType const & /*z-vec*/,
													VectorType const & /*dirx-vec*/, VectorType const & /*diry-vec*/, VectorType const & /*dirz-vec*/, VectorType const & /**/ ) const;

	virtual PhysicalVolume const * GetAsUnplacedVolume() const
		{
			if (! ( tid==0 && rid==1296) )
				{ return unplacedcone;}
			else
				return this;
		}
};


template<int tid, int rid, class ConeType, typename T>
void
PlacedCone<tid,rid,ConeType,T>::printInfoHitPoint( char const * c, Vector3D const & vec, double distance ) const
{
	std::cout << c << " " << vec << "\t\t at dist " << distance << " in z " << GeneralPhiUtils::IsInRightZInterval<T>( vec, coneparams->dZ )
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

	return ( innerradius*innerradius <= d && d <= outerradius*outerradius );
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

	T a = 1 - dir.z*dir.z*(1+m*m);
	T b = 2 * ( pos.x*dir.x + pos.y*dir.y - m*m*pos.z*dir.z - m*n*dir.z);
	T c = ( pos.x*pos.x + pos.y*pos.y - m*m*pos.z*pos.z - 2*m*n*pos.z - n*n );

	T discriminant = b*b - 4*a*c;

	if( discriminant >= 0 )
	{
		if( wantMinusSolution ) distance = ( -b - sqrt(discriminant) )/(2.*a);
		if( ! wantMinusSolution ) distance = (-b + sqrt(discriminant) )/(2.*a);
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

template<int tid, int rid, typename ConeType, typename ValueType>
inline
bool PlacedCone<tid,rid,ConeType,ValueType>::UnplacedContains( Vector3D const & point ) const
{
	ValueType fDz = coneparams->dZ;
	// test if point is inside this cone
	if ( std::abs(point.z) > coneparams->dZ ) return false;

	ValueType r2 = point.x*point.x+point.y*point.y;

	// calculate cone stuff at the height of
	ValueType rh = coneparams->outerslope*point.z + coneparams->outeroffset;
	if( r2 > rh*rh ) return false;

	if( ConeTraits::NeedsRminTreatment<ConeType>::value )
	{
		ValueType rl = coneparams->innerslope*point.z + coneparams->inneroffset;
		if( r2 < rl*rl ) return false;
	}
	if ( ConeTraits::NeedsPhiTreatment<ConeType>::value )
	{
		if( GeneralPhiUtils::PointIsInPhiSector<ValueType>( coneparams->normalPhi1 , coneparams->normalPhi2, point )) return false;
	}
	return true;
}

template<int tid, int rid, typename ConeType, typename ValueType>
inline
bool PlacedCone<tid,rid,ConeType,ValueType>::Contains( Vector3D const & pointmaster ) const
{
	Vector3D pointlocal;
	matrix->MasterToLocal<tid,rid>( pointmaster, pointlocal );
	return this->template UnplacedContains( pointlocal );
}


template<int tid, int rid, typename TubeType, typename ValueType>
template<typename VectorType>
inline
__attribute__((always_inline))
typename VectorType::Mask PlacedCone<tid,rid,TubeType, ValueType>::determineRHit( VectorType const & x, VectorType const & y, VectorType const & z,
											VectorType const & dirx, VectorType const & diry, VectorType const & dirz, VectorType const & distanceR ) const
{
	if( ! ConeTraits::NeedsPhiTreatment<TubeType>::value )
	{
		return distanceR > 0 && GeneralPhiUtils::IsInRightZInterval<ValueType>( z+distanceR*dirz, coneparams->dZ );
	}
	else
	{
		// need to have additional look if hitting point on zylinder is not in empty phi range
		VectorType xhit = x + distanceR*dirx;
		VectorType yhit = y + distanceR*diry;
		return distanceR > 0 && GeneralPhiUtils::IsInRightZInterval<ValueType>( z + distanceR*dirz, coneparams->dZ )
							 && ! GeneralPhiUtils::PointIsInPhiSector<ValueType>(
								 coneparams->normalPhi1.x, coneparams->normalPhi1.y, coneparams->normalPhi2.x, coneparams->normalPhi2.y, xhit, yhit );
	}
}


template<int tid, int rid, typename ConeType, typename ValueType>
template<typename VectorType>
inline
__attribute__((always_inline))
typename VectorType::Mask PlacedCone<tid,rid,ConeType, ValueType>::determineZHit( VectorType const & x, VectorType const & y, VectorType const & z,
											VectorType const & dirx, VectorType const & diry, VectorType const & dirz, VectorType const & distancez ) const
{
	//TODO: THIS IMPLEMENTATION IS ONLY TEMPORAY; WE CAN MAKE THIS MUCHER NICER


	if( ! ConeTraits::NeedsRminTreatment<ConeType>::value &&  ! ConeTraits::NeedsPhiTreatment<ConeType>::value )
	{
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		// this is big difference to tube
		VectorType zhit = z + distancez*dirz;

		VectorType ro =  coneparams->outerslope*zhit + Vc::One*coneparams->outeroffset; // actually this should just be Rmax1 or Rmax2
        // we only need a way to find this out

		return distancez > 0 && ((xhit*xhit + yhit*yhit) <= ro*ro);
	}

	if( ! ConeTraits::NeedsRminTreatment<ConeType>::value &&  ConeTraits::NeedsPhiTreatment<ConeType>::value )
	{
		// need to have additional look if hitting point on zylinder is not in empty phi range
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;

		VectorType zhit = z + distancez*dirz;
		VectorType ro =  coneparams->outerslope*zhit + Vc::One*coneparams->outeroffset;

		return distancez > 0 && ((xhit*xhit + yhit*yhit) <= ro*ro) && ! GeneralPhiUtils::PointIsInPhiSector<ValueType>(
					coneparams->normalPhi1.x, coneparams->normalPhi1.y, coneparams->normalPhi2.x, coneparams->normalPhi2.y, xhit, yhit );
	}

	if( ConeTraits::NeedsRminTreatment<ConeType>::value &&  ! ConeTraits::NeedsPhiTreatment<ConeType>::value )
	{
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		VectorType hitradiussquared = xhit*xhit + yhit*yhit;

		VectorType zhit = z + distancez*dirz;
		VectorType ro =  coneparams->outerslope*zhit + Vc::One*coneparams->outeroffset;
		VectorType ri =  coneparams->innerslope*zhit + Vc::One*coneparams->inneroffset;

		return distancez > 0 && hitradiussquared <= ro*ro && hitradiussquared >= ri*ri;
	}

	else // this should be totally general case ( Rmin and Rmax and general phi )
	{
		VectorType xhit = x + distancez*dirx;
		VectorType yhit = y + distancez*diry;
		VectorType hitradiussquared = xhit*xhit + yhit*yhit;
		VectorType zhit = z + distancez*dirz;
		VectorType ro =  coneparams->outerslope*zhit + Vc::One*coneparams->outeroffset;
		VectorType ri =  coneparams->innerslope*zhit + Vc::One*coneparams->inneroffset;

		return distancez > 0 && hitradiussquared <= ro*ro
							 && hitradiussquared >= ri*ri
							 && ( ! GeneralPhiUtils::PointIsInPhiSector<ValueType>(
									coneparams->normalPhi1.x, coneparams->normalPhi1.y, coneparams->normalPhi2.x, coneparams->normalPhi2.y, xhit, yhit ) );
	}
}



template<int tid, int rid, typename ConeType, typename ValueType>
template<typename VectorType>
inline
void PlacedCone<tid,rid,ConeType,ValueType>::DistanceToIn( VectorType const & xm, VectorType const & ym, VectorType const & zm,
					   VectorType const & dirxm, VectorType const & dirym, VectorType const & dirzm, VectorType const & stepmax, VectorType & distance ) const
{
	typedef typename VectorType::Mask MaskType;

	MaskType done_m(false); // which particles in the vector are ready to be returned == aka have been treated
	distance = Utils::kInfinityVc; // initialize distance to infinity

	VectorType x,y,z;
	matrix->MasterToLocal<tid,rid,VectorType>(xm,ym,zm,x,y,z);
	VectorType dirx, diry, dirz;
	matrix->MasterToLocalVec<tid,rid,VectorType>(dirxm,dirym,dirzm,dirx,diry,dirz);

	// do some inside checks
	// if safez is > 0 it means that particle is within z range
	// if safez is < 0 it means that particle is outside z range

	VectorType safez = coneparams->dZ - Vc::abs(z);
	MaskType inz_m = safez > Utils::fgToleranceVc;
	done_m = !inz_m && ( z*dirz >= Vc::Zero ); // particle outside the z-range and moving away

	VectorType r2 = x*x + y*y;
	VectorType n2 = Vc::One-(coneparams->outerslopesquare + 1) *dirz*dirz; // dirx_v*dirx_v + diry_v*diry_v; ( dir is normalized !! )
	VectorType rdotnplanar = x*dirx + y*diry;

	//	T a = 1 - dir.z*dir.z*(1+m*m);
	//	T b = 2 * ( pos.x*dir.x + pos.y*dir.y - m*m*pos.z*dir.z - m*n*dir.z);
	//	T c = ( pos.x*pos.x + pos.y*pos.y - m*m*pos.z*pos.z - 2*m*n*pos.z - n*n );

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
	VectorType c = r2 - coneparams->outerslopesquare*z*z - 2*coneparams->outerslope*coneparams->outeroffset * z - coneparams->outeroffsetsquare;

	VectorType a = n2;

	VectorType b = 2*(rdotnplanar - z*dirz*coneparams->outerslopesquare - coneparams->outeroffset* coneparams->outerslope*dirz);
	VectorType discriminant = b*b-4*a*c;
	MaskType   canhitrmax = ( discriminant >= Vc::Zero );

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
	distanceRmax( canhitrmax ) = (-b - Vc::sqrt( discriminant ))/(2.*a);

	// this determines which vectors are done here already
	MaskType Rdone = determineRHit( x, y, z, dirx, diry, dirz, distanceRmax );
	distanceRmax( ! Rdone ) = Utils::kInfinityVc;
	MaskType rmindone;
	// **** inner tube ***** only compiled in for tubes having inner hollow tube ******/
	if ( ConeTraits::NeedsRminTreatment<ConeType>::value )
	{
		// case of the Cone, generally all coefficients a, b, and c change
		a = Vc::One-(coneparams->innerslopesquare + 1) *dirz*dirz;
		c = r2 -  coneparams->innerslopesquare*z*z - 2*coneparams->innerslope*coneparams->inneroffset * z - coneparams->inneroffsetsquare;
		b = 2*(rdotnplanar - dirz*(z*coneparams->innerslopesquare + coneparams->inneroffset* coneparams->innerslope));
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


template<int tid, int rid, typename TubeType, typename ValueType>
inline
void PlacedCone<tid,rid,TubeType,ValueType>::DistanceToIn( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double const * steps, double * distance ) const
{
	int i=0;
	typedef typename Vc::Vector<ValueType> VectorType;
	for( i=0; i < points_v.size; i += Vc::Vector<ValueType>::Size )
		{
			VectorType x( &points_v.x[i] );
			VectorType y( &points_v.y[i] );
			VectorType z( &points_v.z[i] );
			VectorType xd( &dirs_v.x[i] );
			VectorType yd( &dirs_v.y[i] );
			VectorType zd( &dirs_v.z[i] );
			VectorType step( &steps[i] );
			VectorType dist;
			DistanceToIn< VectorType >(x, y, z, xd, yd, zd, step, dist);

			// store back result
			dist.store( &distance[i] );
		}
}

// this is now just the same high-level implementation as the one for the Tube --> potential further code reduction
template<int tid, int rid, typename ConeType, typename T>
inline
GlobalTypes::SurfaceEnumType PlacedCone<tid,rid,ConeType,T>::UnplacedContains_WithSurface( Vector3D const & x ) const
{
	GlobalTypes::SurfaceEnumType inside_z;
	GlobalTypes::SurfaceEnumType inside_r;
	GlobalTypes::SurfaceEnumType inside_phi;

	inside_z = ConeKernels::InsideZ( coneparams, x );
	if ( inside_z == GlobalTypes::kOutside ) return GlobalTypes::kOutside;

	inside_r = ConeKernels::InsideR<ConeType>( coneparams, x );
	if ( inside_r == GlobalTypes::kOutside ) return GlobalTypes::kOutside;

	inside_phi = ConeKernels::InsidePhi<ConeType>( coneparams, x );
	if ( inside_phi == GlobalTypes::kOutside ) return GlobalTypes::kOutside;

	// at this point we are either inside or on the surface
    // we are on surface if at least one of the surfaces is touched
	if( inside_z == GlobalTypes::kOnSurface || inside_r == GlobalTypes::kOnSurface || inside_phi == GlobalTypes::kOnSurface )
		return GlobalTypes::kOnSurface;

	// we are inside
	return GlobalTypes::kInside;
}

template<int tid, int rid, typename ConeType, typename T>
inline
GlobalTypes::SurfaceEnumType PlacedCone<tid,rid,ConeType,T>::Contains_WithSurface( Vector3D const & x ) const
{
	// same pattern over and over again ...
	Vector3D xp;
	matrix->MasterToLocal<tid,rid>(x,xp);
	return this->PlacedCone<tid,rid,ConeType,T>::UnplacedContains_WithSurface(xp);
}



#endif /* PHYSICALCONE_H_ */
