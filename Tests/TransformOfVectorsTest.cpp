/*
 * TransformOfVectorsTest.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: swenzel
 */

// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#include "TGeoMatrix.h"
#include <iostream>

const std::vector<std::vector<double>> TransCases  {{0,0,0},
			   				   {10, 0, 0},
								{10, -10, 0},
								{10,-100,100}};

// for the rotations ( C++11 feature )
const std::vector<std::vector<double> > EulerAngles  {{0.,0.,0.},
										{30.,0, 0.},
										{0, 45., 0.},
										{0, 0, 67.5},
										{0, 0, 180.},
										{30, 48, 0.},
										{78.2, 81.0, -10.}};

int main()
{
	Vectors3DSOA points, transformedpoints;
	StructOfCoord rpoints, transformedpoints2;

	points.alloc(1024);
	rpoints.alloc(1024);
	transformedpoints.alloc(1024);
	transformedpoints2.alloc(1024);

	for(int i=0;i<1024;++i)
	{
		points.x[i]=i;
		points.y[i]=i/10;
		points.z[i]=-i;
		rpoints.x[i]=i;
		rpoints.y[i]=i/10;
		rpoints.z[i]=-i;
	}


	for( int r=0; r< EulerAngles.size(); ++r ) // rotation cases
		for( int t=0; t<TransCases.size(); ++t ) // translation cases
		{
			TransformationMatrix const *tm = TransformationMatrix::createSpecializedMatrix(TransCases[t][0], TransCases[t][1], TransCases[t][2],
														 EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]);

			tm->MasterToLocal(points,transformedpoints);
			delete tm;


			TGeoMatrix *tmr = new TGeoCombiTrans(TransCases[t][0], TransCases[t][1], TransCases[t][2],
														new TGeoRotation("rot1",EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]));

			tmr->MasterToLocal_v(rpoints, transformedpoints2, 1024);
			delete tmr;

			std::cerr << transformedpoints.x[345] << " " << transformedpoints2.x[345] << std::endl;
		}
}
