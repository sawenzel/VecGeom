/*
 * Vector3D.h
 *
 *  Created on: Nov 10, 2013
 *      Author: swenzel
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

struct Vector3D
{
	double x,y,z;
	Vector3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {};
	Vector3D() {};
};

struct Vectors3DSOA
{
	// this needs to be aligned
	double *x,*y,*z;
	int size;
};


#endif /* VECTOR3D_H_ */
