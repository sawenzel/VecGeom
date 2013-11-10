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

	// The placed boxed can easily access the private members
	friend class PhysicalBox;
};


//template <PlacementIdType id>
class PhysicalBox : PhysicalVolume
{
private:
	BoxParameters const * boxparams;

	// will provide a private constructor

public:
	virtual double DistanceToIn( Vector3D const &, Vector3D const &) const;
	virtual double DistanceToOut( Vector3D const &, Vector3D const &) const;

// a factory method that produces a specialized box based on params and transformations
	static PhysicalBox* MakeBox( BoxParameters *param, TransformationMatrix *m );
};
#endif /* PHYSICALBOX_H_ */
