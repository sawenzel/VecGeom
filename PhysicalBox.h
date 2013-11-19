/*
 * PhysicalBox.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef PHYSICALBOX_H_
#define PHYSICALBOX_H_

#include "PhysicalVolume.h"
#include "LogicalVolume.h"
#include "Vector3D.h"
#include "TransformationMatrix.h"
#include <iostream>
#include "Vc/vector.h"

class BoxParameters // : ShapeParameters
{
private:
	double dX; // half distances in x-y-z direction
	double dY;
	double dZ;

public:
	BoxParameters(double x,double y, double z) : dX(x), dY(y), dZ(z) {}

	virtual void inspect() const;

	double GetDX() const {return dX;}
	double GetDY() const {return dY;}
	double GetDZ() const {return dZ;}

	virtual ~BoxParameters(){};
	// The placed boxed can easily access the private members
	template<int,int> friend class PlacedBox;
};


//template<TranslationIdType tid, RotationIdType rid>
template<TranslationIdType tid, RotationIdType rid>
class PlacedBox : public PhysicalVolume
{

private:
	BoxParameters const * boxparams;
	//friend class GeoManager;
	template<bool in>
	double Safety( Vector3D const &) const;

public:
	void foo() const;
	double GetDX() const {return boxparams->GetDX();}
	double GetDY() const {return boxparams->GetDY();}
	double GetDZ() const {return boxparams->GetDZ();}

	//will provide a private constructor
	PlacedBox(BoxParameters const * bp, TransformationMatrix const *m) : PhysicalVolume(m), boxparams(bp) {};

	virtual double DistanceToIn( Vector3D const &, Vector3D const &, double cPstep ) const;
	virtual double DistanceToOut( Vector3D const &, Vector3D const &, double cPstep ) const {return 0;}
	virtual bool Contains( Vector3D const & ) const;
	virtual bool UnplacedContains( Vector3D const & ) const;
	virtual double SafetyToIn( Vector3D const & ) const;
	virtual double SafetyToOut( Vector3D const &) const;

	// for the basket treatment
	virtual void DistanceToIn( Vectors3DSOA const &, Vectors3DSOA const &, double, double * ) const;

	template<typename T>
	inline
	void DistanceToInT(Vectors3DSOA const &, Vectors3DSOA const &, double, double * ) const;

	// a template version for T = Vc or T = Boost.SIMD or T= double
	template<typename T>
	inline
	void DistanceToIn( T const & /*x-vec*/, T const & /*y-vec*/, T const & /*z-vec*/,
					   T const & /*dx-vec*/, T const & /*dy-vec*/, T const & /*dz-vec*/, double step, T & /*result*/ ) const;


	virtual ~PlacedBox(){};

	//a factory method that produces a specialized box based on params and transformations
	//static PhysicalBox* MakeBox( BoxParameters *param, TransformationMatrix *m );
};


struct UUtils
{
	static constexpr double kInfinity=1E30;
};


template <TranslationIdType tid, RotationIdType rid>
double
PlacedBox<tid,rid>::DistanceToIn(Vector3D const &x, Vector3D const &y, double cPstep ) const
{
	// Computes distance from a point presumably outside the solid to the solid
	// surface. Ignores first surface if the point is actually inside. Early return
	// infinity in case the safety to any surface is found greater than the proposed
	// step aPstep.
	// The normal vector to the crossed surface is filled only in case the box is
	// crossed, otherwise aNormal.IsNull() is true.

	// Compute safety to the closest surface on each axis.
	// Early exits if safety bigger than proposed step.
	const double delta = 1E-9;

	// here we do the point transformation
	Vector3D aPoint;
	matrix->MasterToLocal<tid,rid>(x, aPoint);

	//   aNormal.SetNull();
	double safx = std::abs(aPoint.x) - boxparams->dX;
	double safy = std::abs(aPoint.y) - boxparams->dY;
	double safz = std::abs(aPoint.z) - boxparams->dZ;

	if ((safx > cPstep) || (safy > cPstep) || (safz > cPstep))
		return 1E30;

	// only here we do the directional transformation
	Vector3D aDirection;
	matrix->MasterToLocalVec<rid>(y, aDirection);

	// Check if numerical inside
	bool outside = (safx > 0) || (safy > 0) || (safz > 0);
	if ( !outside ) {
		// If point close to this surface, check against the normal
		if ( safx > -delta ) {
			return ( aPoint.x * aDirection.x > 0 ) ? UUtils::kInfinity : 0.0;
		}
		if ( safy > -delta ) {
			return ( aPoint.y * aDirection.y > 0 ) ? UUtils::kInfinity : 0.0;
		}
		if ( safz > -delta ) {
			return ( aPoint.z * aDirection.z > 0 ) ? UUtils::kInfinity : 0.0;
		}
		// Point actually "deep" inside, return zero distance, normal un-defined
		return 0.0;
	}

	// The point is really outside. Only axis with positive safety to be
	// considered. Early exit on each axis if point and direction components
	// have the same .sign.
	double dist = 0.0;
	double coordinate = 0.0;
	if ( safx > 0 ) {
		if ( aPoint.x * aDirection.x >= 0 ) return UUtils::kInfinity;
		dist = safx/std::abs(aDirection.x);
		coordinate = aPoint.y + dist*aDirection.y;
		if ( std::abs(coordinate) < boxparams->dY ) {
			coordinate = aPoint.z + dist*aDirection.z;
			if ( std::abs(coordinate) < boxparams->dZ ) {
				return dist;
			}
		}
	}
	if ( safy > 0 ) {
		if ( aPoint.y * aDirection.y >= 0 ) return UUtils::kInfinity;
		dist = safy/std::abs(aDirection.y);
		coordinate = aPoint.x + dist*aDirection.x;
		if ( std::abs(coordinate) < boxparams->dX ) {
			coordinate = aPoint.z + dist*aDirection.z;
			if ( std::abs(coordinate) < boxparams->dZ ) {
				return dist;
			}
		}
	}
	if ( safz > 0 ) {
		if ( aPoint.z * aDirection.z >= 0 ) return UUtils::kInfinity;
		dist = safz/std::abs(aDirection.z);
		coordinate = aPoint.x + dist*aDirection.x;
		if ( std::abs(coordinate) < boxparams->dX ) {
			coordinate = aPoint.y + dist*aDirection.y;
			if ( std::abs(coordinate) < boxparams->dY ) {
				return dist;
			}
		}
	}
	return UUtils::kInfinity;
}

// a template version for T = Vc or T = Boost.SIMD or T= double
// this is the kernel operating on type T
template<int tid, int rid>
template<typename T>
inline
void PlacedBox<tid,rid>::DistanceToIn( T const & x, T const & y, T const & z,
					   T const & dirx, T const & diry, T const & dirz, double stepmax, T & distance ) const
{
	   T in(1.);
	   T saf[3];
	   T newpt[3];
	   T tiny(1e-20);
	   T big(UUtils::kInfinity);
	   T faraway(0.); // initializing all components to zero
	   T par[3]={ boxparams->dX, boxparams->dY, boxparams->dZ }; // very convenient

	   // new thing: do coordinate transformation in place here
	   T localx, localy, localz;
	   matrix->MasterToLocal<tid,rid,T>(x,y,z,newpt[0],newpt[1],newpt[2]);
	   //
	   saf[0] = Vc::abs(newpt[0])-par[0];
	   saf[1] = Vc::abs(newpt[1])-par[1];
	   saf[2] = Vc::abs(newpt[2])-par[2];
	   faraway(saf[0]>=stepmax || saf[1]>=stepmax || saf[2]>=stepmax)=1;
	   in(saf[0]<0. && saf[1]<0. && saf[2]<0.)=0;
	   distance=big;

	   if( faraway > T(0.) )
	       return; // return big

	   // new thing:  do coordinate transformation for directions here
	   T localdirx, localdiry, localdirz;
	   matrix->MasterToLocalVec<tid,rid,T>(dirx, diry, dirz, localdirx, localdiry, localdirz);
	   //

	   // proceed to analysis of hits
	   T snxt[3];
	   T hit0=T(0.);
	   snxt[0] = saf[0]/(Vc::abs(localdirx)+tiny); // distance to y-z face
	   T coord1=newpt[1]+snxt[0]*localdiry; // calculate new y and z coordinate
	   T coord2=newpt[2]+snxt[0]*localdirz;
	   hit0( saf[0] > 0 && newpt[0]*localdirx < 0 && ( Vc::abs(coord1) <= par[1] && Vc::abs(coord2) <= par[2] ) ) = 1; // if out and right direction

	   T hit1=T(0.);
	   snxt[1] = saf[1]/(Vc::abs(localdiry)+tiny); // distance to x-z face
	   coord1=newpt[0]+snxt[1]*localdirx; // calculate new x and z coordinate
	   coord2=newpt[2]+snxt[1]*localdirz;
	   hit1( saf[1] > 0 && newpt[1]*localdiry < 0 && ( Vc::abs(coord1) <= par[0] && Vc::abs(coord2) <= par[2] ) ) = 1; // if out and right direction

	   T hit2=T(0.);
	   snxt[2] = saf[2]/(Vc::abs(localdirz)+tiny); // distance to x-y face
	   coord1=newpt[0]+snxt[2]*localdirx; // calculate new x and y coordinate
	   coord2=newpt[1]+snxt[2]*localdiry;
	   hit2( saf[2] > 0 && newpt[2]*localdirz < 0 && ( Vc::abs(coord1) <= par[0] && Vc::abs(coord2) <= par[1] ) ) = 1; // if out and right direction

	   distance( hit0>0 || hit1>0 || hit2>0 ) = (hit0*snxt[0] + hit1*snxt[1] + hit2*snxt[2]);
	   distance=in*distance;
	   return;
}


// for the basket treatment
template<int tid, int rid>
template<typename T>
inline
void PlacedBox<tid,rid>::DistanceToInT( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double step, double * distance ) const
{
	for( int i=0; i < points_v.size; i += T::Size )
		{
			T x( &points_v.x[i] );
			T y( &points_v.y[i] );
			T z( &points_v.z[i] );
			T xd( &dirs_v.x[i] );
			T yd( &dirs_v.y[i] );
			T zd( &dirs_v.z[i] );
			T dist;
			DistanceToIn<T>(x, y, z, xd, yd, zd, step, dist);

			// store back result
			dist.store( &distance[i] );
		}
}

// dispatching the virtual method to some concrete method
template<int tid, int rid>
void PlacedBox<tid,rid>::DistanceToIn( Vectors3DSOA const & points_v, Vectors3DSOA const & dirs_v, double step, double * distance ) const
{
	// PlacedBox<tid,rid>::DistanceToInT< Vc::double_v, Vc >(points_v, dirs_v, step, distance);
	this->template DistanceToInT<Vc::double_v>( points_v, dirs_v, step, distance);
}


template<int tid, int rid>
bool PlacedBox<tid,rid>::Contains( Vector3D const & point ) const
{
	// here we do the point transformation
	Vector3D localPoint;
	matrix->MasterToLocal<tid,rid>(point, localPoint);

	// this could be vectorized also
	if ( std::abs(localPoint.x) > boxparams->dX ) return false;
	if ( std::abs(localPoint.y) > boxparams->dY ) return false;
	if ( std::abs(localPoint.z) > boxparams->dZ ) return false;
	return true;
}


template<int tid, int rid>
bool PlacedBox<tid,rid>::UnplacedContains( Vector3D const & point ) const
{
	// here we do the point transformation
	Vector3D localPoint;
	matrix->MasterToLocal<0,0>(point, localPoint);

	// this could be vectorized also
	if ( std::abs(localPoint.x) > boxparams->dX ) return false;
	if ( std::abs(localPoint.y) > boxparams->dY ) return false;
	if ( std::abs(localPoint.z) > boxparams->dZ ) return false;
	return true;
}

template<int tid, int rid>
template<bool in>
double PlacedBox<tid, rid>::Safety( Vector3D const & point ) const
{

   double safe, safy, safz;
   if (in) {
	   Vector3D localPoint;
	   matrix->MasterToLocal<tid,rid>(point, localPoint);
	   safe = boxparams->dX - std::fabs(localPoint.x);
	   safy = boxparams->dY - std::fabs(localPoint.y);
	   safz = boxparams->dZ - std::fabs(localPoint.z);
	   if (safy < safe) safe = safy;
	   if (safz < safe) safe = safz;
	   }
   else {
	   Vector3D localPoint;
	   matrix->MasterToLocal<tid,rid>(point, localPoint);
	   safe = -boxparams->dX + std::fabs(localPoint.x);
	   safy = -boxparams->dY + std::fabs(localPoint.y);
	   safz = -boxparams->dZ + std::fabs(localPoint.z);
	   if (safy > safe) safe = safy;
	   if (safz > safe) safe = safz;
   }
   return safe;
}

template<int tid, int rid>
double PlacedBox<tid,rid>::SafetyToIn( Vector3D const & point ) const
{
	return Safety<false>( point );
}

template<int tid, int rid>
double PlacedBox<tid,rid>::SafetyToOut( Vector3D const & point ) const
{
	return Safety<true>( point );
}


#endif /* PHYSICALBOX_H_ */
