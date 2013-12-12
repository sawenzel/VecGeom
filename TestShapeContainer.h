/*
 * TestShapeContainer.h
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

// a class which creates a couple of test instances for various shapes and methods to access those shapes

#ifndef TESTSHAPECONTAINER_H_
#define TESTSHAPECONTAINER_H_


class TubeParameters;
class BoxParameters;

#include <vector>
#include <cmath>
#include "PhysicalCone.h"

class TestShapeContainer
{
private:
	// a simple vector storing various cones as sets of parameters
	const std::vector<std::vector<double> > coneparams { /* rmin1, rmin2, rmax1, rmax2, dz, phistart, deltaphi */
											{ 0., 0., 10., 15., 30., 0, 2*M_PI}, // a full cone without inner radius
											{ 0., 0., 10., 15., 30., 0, M_PI/2.}, // a half cone without inner radius
											{ 8., 8., 10., 15., 30., 0, 3.*M_PI/2.}, //
											{ 0., 0., 10., 15., 30., 0, M_PI},  // a half cone without inner radius
											{ 8., 8., 10., 15., 30., 0, 2*M_PI}, //
											{ 8., 8., 10., 15., 30., 0, M_PI}, //
											{ 8., 0., 10., 0., 30., 0, M_PI}, //
											{ 8., 7., 10., 15., 30., 0, 3.21*M_PI/2.} // some weird angle phi
										 };

public:
	int GetNumberOfConeTestShapes()
	{
		return coneparams.size();
	}

	ConeParameters<double> const * GetConeParams(int i)
	{
		return new ConeParameters<double>( coneparams[i][0],
										   coneparams[i][2],
										   coneparams[i][1],
										   coneparams[i][3],
										   coneparams[i][4],
										   coneparams[i][5],
										   coneparams[i][6] );
	}

};


#endif /* TESTSHAPECONTAINER_H_ */
