/*
 * PhysicalPolycone.h
 *
 *  Created on: Jan 8, 2014
 *      Author: swenzel
 */

#ifndef PHYSICALPOLYCONE_H_
#define PHYSICALPOLYCONE_H_

#include <vector>
#include <algorithm>
#include <map>

#include "UPolycone.hh"
#include "TGeoPcon.h"

#include "ShapeFactories.h"
#include "GeoManager.h"

// class encapsulating the defining properties of a Polycone
template<typename Float_t=double>
class
PolyconeParameters
{
public:
	enum PconSectionType { kTube, kCone };

private:
	bool has_rmin; // this is true if a least one section has an inner radius
	bool is_full_phi; // true if DPhi =

	Float_t fSPhi; // start angle
	Float_t fDPhi; // total phi angle
	int numZPlanes; // number of z planes ( minimum 2 )
	std::vector<Float_t> zPlanes; // position of z planes
	std::vector<Float_t> rInner; // inner radii at position of z planes
	std::vector<Float_t> rOuter; // outer radii at position of z planes
	std::vector<PconSectionType> pconsectype;
	std::vector<PhysicalVolume const *> pconsection; // maps a section to a concrete tube or cone; mapping is to PhysicalVolume pointer, concrete information is stored in pconsectype )

	void DetermineAndSetRminProperty( int numZPlanes, Float_t const rinner[] );
	PconSectionType DetermineSectionType( Float_t rinner1, Float_t rinner2, Float_t router1, Float_t router2 ) const;
	void Init(Float_t sphi, Float_t dphi, int numZPlanes, Float_t const zplanes[], Float_t const rinner[], Float_t const router[]); // see USolids

public:
	// a constructor
	PolyconeParameters( Float_t sphi,
						Float_t dphi,
						int numzplanes,
						Float_t const zplanes[],
						Float_t const rinner[],
						Float_t const router[] ) : zPlanes(numZPlanes), rInner(numZPlanes), rOuter(numZPlanes),
						numZPlanes(numzplanes), fSPhi(sphi), fDPhi(dphi) {
		Init(sphi, dphi, numzplanes, zplanes, rinner, router);
	}

	// the following methods will be used by the factory to determine the polyconetype
	bool HasRmin() const { return has_rmin; }
	bool IsFullPhi() const { return is_full_phi; }
	Float_t GetDPhi() const { return fDPhi; }
	Float_t GetSPhi() const { return fSPhi; }
	Float_t GetDZ() const { return (zPlanes[numZPlanes-1] - zPlanes[0])/2.; }
	Float_t GetMaxR() const { return *std::max_element(rOuter.begin(), rOuter.end()); }


	std::vector<Float_t> const & GetZPlanes() const { return zPlanes; }
	std::vector<Float_t> const & GetInnerRs() const { return rInner; }
	std::vector<Float_t> const & GetOuterRs() const { return rOuter; }
	Float_t GetZPlane( int i ) const { return zPlanes[i]; }
	PconSectionType GetPConSectionType( int i ) const { return pconsection[i]; }
	int GetNZ() const { return numZPlanes; }
	int GetNumSections() const { return numZPlanes-1; }
	PhysicalVolume const * GetSectionVolume(int i) const { return pconsection[i]; }

	inline
	int FindZSection( Float_t zcoord ) const {
		for(auto i=0;i<numZPlanes;i++){
			if( zPlanes[i] <= zcoord && zcoord <= zPlanes[i+1] ) return i;
		}
		return -1;
	};

};

// this might be placed into a separate header later
template<typename Float_t>
void PolyconeParameters<Float_t>::Init(  Float_t sphi, Float_t dphi, int numZPlanes, Float_t const zplanes[], Float_t const rinner[], Float_t const router[]  )
{
	DetermineAndSetRminProperty( numZPlanes, rinner );

	// check validity !!
	// TOBEDONE

    // copy data ( might not be necessary )
	for(auto i=0; i<numZPlanes-1; i++)
	{
		zPlanes[i]=zplanes[i];
		rInner[i]=rinner[i];
		rOuter[i]=router[i];
	}

	for(auto i=0; i<numZPlanes-1; i++ )
	{
		// analyse rmin / rmax at planes i and i+1
		pconsectype[i] =  DetermineSectionType( rinner[i], rinner[i+1], router[i], router[i+1] );

		if( pconsectype[i] == kTube )
		{
			// creates a tube without translation nor rotation
			pconsection[i] = TubeFactory::template Create<0, 1296>(
					new TubeParameters<Float_t>(rinner[i], router[i], (zplanes[i+1] - zplanes[i])/2., sphi, dphi),
					IdentityTransformationMatrix,
					!has_rmin);
		}
		if( pconsectype[i] == kCone )
		{
			// creates a cone without translation nor rotation
			pconsection[i] = ConeFactory::template Create<0, 1296>(
					new ConeParameters<Float_t>(rinner[i], router[i], rinner[i+1], router[i+1], (zplanes[i+1] - zplanes[i])/2., sphi, dphi),
					IdentityTransformationMatrix,
					!has_rmin);
		}
	}
}

template<typename Float_t>
void PolyconeParameters<Float_t>::DetermineAndSetRminProperty( int numZPlanes, Float_t const rinner[] )
{
	has_rmin = false;
	for(auto i=0; i<numZPlanes; i++)
	{
		if( rinner[i] > Utils::frHalfTolerance ) has_rmin = true;
	}
}



template<TranslationIdType tid, RotationIdType rid, typename ConeType=ConeTraits::HollowConeWithPhi, typename Float_t=double>
class
PlacedPolycone : public PhysicalVolume
{
private:
	PolyconeParameters<Float_t> * pconparams;

public:
	PlacedPolycone( PolyconeParameters<Float_t> const * pcp, TransformationMatrix const * tm ) : PhysicalVolume(tm), pconparams(pcp)
	{
		this->bbox = new PlacedBox<1,1296>(
				new BoxParameters( pconparams->GetMaxR(), pconparams->GetMaxR(), pconparams->GetDZ()),
				IdentityTransformationMatrix );

		analogoususolid = new UPolycone( "internal_usolid", pconparams->GetSPhi(),
				pconparams->GetDPhi(),
				pconparams->GetNumZPlanes(),
				pconparams->GetZPlanes(),
				pconparams->GetInnerRs(),
				pconparams->GetOuterRs());

		analogousrootsolid = 0; // new TGeoPcon("internal_tgeopcon", GetRmin(), GetRmax(), GetDZ());

		/*
		if(! (tid==0 && rid==1296) )
		{
			unplacedtube = new PlacedUSolidsTube<0,1296,TubeType,T>( _tb, m );
		}
		 */
	}


	virtual double DistanceToIn( Vector3D const &, Vector3D const &, double ) const {return 0.;}
	virtual double DistanceToOut( Vector3D const &, Vector3D const &, double ) const {return 0.;}

	virtual bool   Contains( Vector3D const & x) const {Vector3D xprime;matrix->MasterToLocal<tid,rid>(x,xprime); return UnplacedContains(xprime);}

	__attribute__((always_inline))
	inline
	virtual bool   UnplacedContains( Vector3D const & x) const {return false;}

	// the contains function knowing about the surface
	virtual GlobalTypes::SurfaceEnumType   UnplacedContains_WithSurface( Vector3D const & ) const;
	virtual GlobalTypes::SurfaceEnumType   Contains_WithSurface( Vector3D const & ) const;

	virtual double SafetyToIn( Vector3D const & ) const {return 0.;}
	virtual double SafetyToOut( Vector3D const & ) const {return 0.;}

	// for basket treatment (supposed to be dispatched to particle parallel case)
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};
	virtual void DistanceToOut( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};


	// for basket treatment (supposed to be dispatched to loop case over (SIMD) optimized 1-particle function)
	virtual void DistanceToInIL( Vectors3DSOA const &, Vectors3DSOA const &, double const * /*steps*/, double * /*result*/ ) const {};
	//	virtual void DistanceToInIL( std::vector<Vector3D> const &, std::vector<Vector3D> const &, double const * /*steps*/, double * /*result*/ ) const;
	virtual void DistanceToInIL( Vector3D const *, Vector3D const *, double const * /*steps*/, double * /*result*/, int /*size*/ ) const {};
};


template<TranslationIdType tid, RotationIdType rid, typename ConeType, typename Float_t>
GlobalTypes::SurfaceEnumType PlacedPolycone<tid,rid,ConeType,Float_t>::Contains_WithSurface(Vector3D const & x ) const
{
	Vector3D xp;
	matrix->MasterToLocal<tid,rid>(x,xp);
	return this->UnplacedContains_WithSurface( xp );
}


template<TranslationIdType tid, RotationIdType rid, typename ConeType, typename Float_t>
GlobalTypes::SurfaceEnumType PlacedPolycone<tid,rid,ConeType,Float_t>::UnplacedContains_WithSurface(Vector3D const & x ) const
{
	// algorithm:
    // check if point is outsize z range
	// find z section of point
	// call Contains function of that section ( no virtual call!! will be inlined )
	// if point found on z-surface of that section, need to check neighbouring sections as well
	if( x.z < pconparams->GetZPlane(0)-Utils::frHalfTolerance
			|| x.z > pconparams->GetZPlane(pconparams->GetNZ()) + Utils::frHalfTolerance )
		return GlobalTypes::kOutside;

	int zsection = pconparams->FindZSection( x.z );

	GlobalTypes::SurfaceEnumType result;

	// section is a tube
	PhysicalVolume const * section = pconparams->GetSectionVolume( zsection );
	if( pconparams->GetPConSectionType( zsection ) == PolyconeParameters<Float_t>::kTube )
	{
		// now comes a bit of template magic to achieve inlining of optimized function / ( no virtual function call )
		result=section->PlacedUSolidsTube<tid,rid, ConeTraits::ConeTypeToTubeType<ConeType>, Float_t>::Contains_WithSurface( x );
	}
	// section is a cone
	if( pconparams->GetPConSectionType( zsection ) == PolyconeParameters<Float_t>::kCone )
	{
		// now comes a bit of template magic to achieve inlining of optimized function / ( no virtual function call )
		result=section->PlacedCone<tid,rid, ConeType,Float_t>::Contains_WithSurface( x );
	}

	// TODO: case where point is on a z-plane surface

	return result;
}

#endif /* PHYSICALPOLYCONE_H_ */
