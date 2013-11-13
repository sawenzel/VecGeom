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

	// The placed boxed can easily access the private members
	// friend class PhysicalBox;
};

template<int tid, int rid>
class A : public PhysicalVolume
{
public:
	A(TransformationMatrix *m) : PhysicalVolume(m) {};
	virtual double DistanceToIn( Vector3D const &, Vector3D const &) const { return 0.; }
	virtual double DistanceToOut( Vector3D const &, Vector3D const &) const { return 0.; }
	virtual ~A(){};
};

//template<TranslationIdType tid, RotationIdType rid>

template<TranslationIdType tid, RotationIdType rid>
class PlacedBox : public PhysicalVolume
{

private:
	BoxParameters const * boxparams;
//	friend class GeoManager;

public:
	// will provide a private constructor
	PlacedBox(BoxParameters const * bp, TransformationMatrix const *m) : PhysicalVolume(m), boxparams(bp) {};


	virtual double DistanceToIn( Vector3D const &, Vector3D const &) const;
	virtual double DistanceToOut( Vector3D const &, Vector3D const &) const {return 0;}

	virtual ~PlacedBox(){};

// a factory method that produces a specialized box based on params and transformations
//	static PhysicalBox* MakeBox( BoxParameters *param, TransformationMatrix *m );
};


template <TranslationIdType tid, RotationIdType rid>
double
PlacedBox<tid,rid>::DistanceToIn(Vector3D const &x, Vector3D const &y) const
{
	// this is just playing around
	//double m = boxparams->dX * matrix->foo<1>(); // foo can be called because Box is a friend

	// inline transformations can easily be placed like this
	//matrix->foo<2>();
	Vector3D yp;
	// inline the correct stuff
	matrix->MasterToLocal<tid,rid>(x, yp);
	return boxparams->GetDX()*yp.z;
}



#endif /* PHYSICALBOX_H_ */
