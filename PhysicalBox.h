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

class BoxParameters : ShapeParameters
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

public:
	//will provide a private constructor
	PlacedBox(BoxParameters const * bp, TransformationMatrix const *m) : PhysicalVolume(m), boxparams(bp) {};

	virtual double DistanceToIn( Vector3D const &, Vector3D const &, double cPstep ) const;
	virtual double DistanceToOut( Vector3D const &, Vector3D const &, double cPstep ) const {return 0;}

	virtual ~PlacedBox(){};

	//a factory method that produces a specialized box based on params and transformations
	//static PhysicalBox* MakeBox( BoxParameters *param, TransformationMatrix *m );
};

struct UUtils
{
	static const double kInfinity=1E30;
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



#endif /* PHYSICALBOX_H_ */
