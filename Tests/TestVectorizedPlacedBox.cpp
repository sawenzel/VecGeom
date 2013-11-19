/*
 * TestVectorizedPlacedBox.cpp
 *
 *  Created on: Nov 18, 2013
 *      Author: swenzel
 */



// testing matrix transformations on vectors
#include "../TransformationMatrix.h"
#//include "TGeoMatrix.h"
#include "../Utils.h"
#include <iostream>
#include "mm_malloc.h"
#include "../GlobalDefs.h"

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
	Vectors3DSOA points, dirs;
	int np=1024;
	points.alloc(np);
	dirs.alloc(np);
	double *distances = _mm_malloc(np*sizeof(double), ALIGNMENT_BOUNDARY);

	StopWatch timer;

    // generate benchmark cases
    for( int r=0; r< EulerAngles.size(); ++r ) // rotation cases
    		for( int t=0; t<TransCases.size(); ++t ) // translation cases
    		  {
    			PhysicalVolume * world = GeoManager::MakePlacedBox( new BoxParameters(100,100,100), new TransformationMatrix(0,0,0,0,0,0));

    			PhysicalVolume * daughter = GeoManager::MakePlacedBox( new BoxParameters(10,15,20),
    			    		new TransformationMatrix(TransCases[t][0], TransCases[t][1], TransCases[t][2],
    			    									EulerAngles[r][0], EulerAngles[r][1], EulerAngles[r][2]));

    			world->AddDaughter(daughter);

    			world->fillWithRandomPoints(points,np);
    			world->fillWithBiasedDirections(points, dirs, np, 1./3);


    			// time performance for this placement ( we should probably include some random physical steps )
    			timer.Start();
    			for(int r=0;r<10;r++)
    			{
    				daughter->DistanceToIn(points,dirs,1E30,distances);
    			}
    			timer.Stop();
    		  }
	return 1;
}
