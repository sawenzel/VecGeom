/*
 * PhysicalBox.cpp
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#include <iostream>
#include "PhysicalBox.h"
#include "TransformationMatrix.h"


void BoxParameters::inspect() const
{
	std::cout << " This is a box " << std::endl;
	std::cout << " dX : " << dX << std::endl;
	std::cout << " dY : " << dY << std::endl;
	std::cout << " dZ : " << dZ << std::endl;
}


//template <PlacementIdType id>
double
PhysicalBox::DistanceToIn(Vector3D const &x, Vector3D const &y) const
{
	// this is just playing around
	double m = boxparams->dX * matrix->foo<1>(); // foo can be called because Box is a friend

	// inline transformations can easily be placed like this
	matrix->foo<2>();

	return 1.;
}
