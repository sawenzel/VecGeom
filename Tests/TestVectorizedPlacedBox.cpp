/*
 * TestVectorizedPlacedBox.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */



// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "TGeoMatrix.h"
#include "../Utils.h"
#include <iostream>

const std::vector<std::vector<double>> TransCases  {{0,0,0},
    {10, 0, 0},
      {10, -10, 0},
	{10,-100,100}};

// for the rotations ( C++11 feature )
const std::vector<std::vector<double> > EulerAngles  {{0.,0.,0.},
    //    {30.,0, 0.},
    //  {0, 45., 0.},
    //	{0, 0, 67.5}};
    {0, 0, 180.}};
	//	{30, 48, 0.},
	//	{78.2, 81.0, -10.}};

int main()
{



	return 1;
}
