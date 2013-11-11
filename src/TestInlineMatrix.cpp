/*
 * TestInlineMatrix.cpp
 *
 *  Created on: Nov 11, 2013
 *      Author: swenzel
 */

#include "../TransformationMatrix.h"
#include "../Vector3D.h"
#include <cstdlib>

int main()
{
	Vector3D x,y;
	double r[9]={1,2,3,4,5,6,7,8,9};
	double t[3]={1,2,3};

	TransformationMatrix m(t,r);

	x.x=rand();
	x.y=rand();
	x.z=rand();


	m.MasterToLocal<1008>(x,y);

	volatile double d=y.x+y.y+y.z;
	return d;
}



